# coding:utf-8
import os
import argparse
import time
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_TII import BiSeNet
from TaskFusion_dataset import Fusion_dataset
from FusionNet import FusionNet
# from tqdm import tqdm
from torch.autograd import Variable
from PIL import Image
from FusionNet1 import FusionNet1
from FusionNet1 import FusionNet2
from FusionNet1 import FusionNet3
from PIL import Image
from torchvision import transforms
# To run, set the fused_dir, and the val path in the TaskFusionDataset.py
import cv2
def char_act(x_visualize, it, type='tset'):
    fused_dir = os.path.join('.\\tzt\\', type)
    x_visualize = x_visualize.clone().detach().cpu().numpy()[-1, :, :, :]  # 用Numpy处理返回的[1,256,513,513]特征图
    x_visualize = np.expand_dims(x_visualize, 0)
    x_visualize = np.max(x_visualize, axis=1)  # shape为[513,513]，二维
    x_visualize = (((x_visualize - np.min(x_visualize)) / (
            np.max(x_visualize) - np.min(x_visualize))) * 255).astype(np.uint8)  # 归一化并映射到0-255的整数，方便伪彩色化
    if not os.path.exists(fused_dir + 'val_pred_temp'):
        os.mkdir(fused_dir + 'val_pred_temp')
    x_visualize=np.squeeze(x_visualize)
    x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
    # x_visualize = np.transpose(x_visualize,(2,0,1))
    cv2.imwrite('D:\\zhuomian\\ganfusion\\tzt\\tsetval_pred_temp\\' + str(it + 1) + '.jpg', x_visualize)
def main():
    fusion_model_path = './model/Fusion/fusion_model.pth'
    ge_model_path = './model/Fusion/generate/fusion_model.pth'
    fusionmodel = eval('FusionNet3')(output=1)
    # generate = eval('FusionNet2')(output=1)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    total = sum([param.nelement() for param in fusionmodel.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    if args.gpu >= 0:
        fusionmodel.to(device)
        # generate.to(device)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    # generate.load_state_dict(torch.load(ge_model_path))
    print('fusionmodel load done!')
    ir_path = './test_imgs/ir'
    vi_path = './test_imgs/vi'
    test_dataset = Fusion_dataset('val', ir_path=ir_path, vi_path=vi_path)
    # test_dataset = Fusion_dataset('val')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    ww=0
    tt = []
    with torch.no_grad():
        for it, (images_vis, images_ir,name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            # a = images_ir.size()[2]
            # b = images_ir.size()[3]
            # images_ir = transforms.Resize(images_ir.size()[2]/16*16,images_ir.size()[3]/16*16)
            # images_vis = transforms.Resize(images_vis.size()[2]/16*16,images_vis.size()[3]/16*16)

            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)
            images_vis_ycrcb = RGB2YCrCb(images_vis)
            torch.cuda.synchronize()
            start = time.time()
            logits,_,_ = fusionmodel(images_vis_ycrcb, images_ir)
            torch.cuda.synchronize()
            end = time.time()
            print(end - start)
            q=end-start
            tt.append(q)
            ww=ww+q
            char_act(images_ir,it)
            #logits = generate(images_vis, logits)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)


            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )

            fused_image = np.uint8(255.0 * fused_image)

            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))
    print(ww/21)
    arr_std = np.std(tt, ddof=1)
    print(arr_std)

def YCrCb2RGB(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def RGB2YCrCb(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=1)
    args = parser.parse_args()
    n_class = 9
    seg_model_path = './model/Fusion/model_final.pth'
    fusion_model_path = './model/Fusion/fusion_model.pth'
    fused_dir = './Fusion_results'
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    main()
