#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from torch.autograd import Variable
from FusionNet1 import FusionNet1
from FusionNet1 import FusionNet2
from FusionNet1 import FusionNet3
from FusionNet1 import NLayerDiscriminator
from FusionNet1 import NLayerDiscriminator2
from FusionNet1 import out1
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from model_TII import BiSeNet
from cityscapes import CityScapes
from loss import OhemCELoss, Fusionloss
from optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
import warnings
import pytorch_msssim
from torchvision.transforms import Resize

warnings.filterwarnings('ignore')
# 加
import cv2
from net import Fusion_network


def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
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


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
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
def char_act(x_visualize, it, type='tset'):
    fused_dir = os.path.join('./tzt/', type)
    x_visualize = x_visualize.clone().detach().cpu().numpy()[-1, :, :, :]  # 用Numpy处理返回的[1,256,513,513]特征图
    x_visualize = np.expand_dims(x_visualize, 0)
    x_visualize = np.max(x_visualize, axis=1).reshape(480, 640)  # shape为[513,513]，二维
    x_visualize = (((x_visualize - np.min(x_visualize)) / (
            np.max(x_visualize) - np.min(x_visualize))) * 255).astype(np.uint8)  # 归一化并映射到0-255的整数，方便伪彩色化
    if not os.path.exists(fused_dir + 'val_pred_temp'):
        os.mkdir(fused_dir + 'val_pred_temp')
    x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
    cv2.imwrite(fused_dir + 'val_pred_temp/' + str(it + 1) + '.jpg', x_visualize)

def train_seg(i=0, logger=None):
    load_path = './model/Fusion/model_final.pth'
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    os.makedirs(modelpth, mode=0o777, exist_ok=True)
    # if logger == None:
    #     logger = logging.getLogger()
    #     setup_logger(modelpth)

    # dataset
    n_classes = 9
    n_img_per_gpu = 4
    n_workers = 4
    cropsize = [640, 480]
    ds = CityScapes('./MSRS/', cropsize=cropsize, mode='train', Method=Method)
    dl = DataLoader(
        ds,
        batch_size=n_img_per_gpu,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model
    ignore_idx = 255
    net = BiSeNet(n_classes=n_classes)
    if i > 0:
        net.load_state_dict(torch.load(load_path))
    net.cuda()
    net.train()
    print('Load Pre-trained Segmentation Model:{}!'.format(load_path))
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 4
    criteria_p = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    # optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    it_start = i * 20000
    iter_nums = 20000

    optim = Optimizer(
        model=net,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
        it=it_start,
    )

    # train loop
    msg_iter = 10
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(iter_nums):
        try:
            im, lb, _ = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            # sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb, _ = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out, mid = net(im)
        lossp = criteria_p(out, lb)
        loss2 = criteria_16(mid, lb)
        loss = lossp + 0.75 * loss2
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        # print training log message
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)

            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join(
                [
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]
            ).format(
                it=it_start + it + 1, max_it=max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed
    # dump the final model
    save_pth = osp.join(modelpth, 'model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info(
        'Segmentation Model Training done~, The Model is saved to: {}'.format(
            save_pth)
    )
    logger.info('\n')


def train_fusion(num=0, logger=None):  # 这个是生成器
    # num: control the segmodel
    # accumulation_steps = 4
    lr_start = 0.0002
    modelpth = './model'
    Method = 'Fusion'
    Method_ge = 'generate'
    modelpth = os.path.join(modelpth, Method)  # './model/Fusion'
    modelpth_ge = os.path.join(modelpth, Method_ge)
    fusionmodel = eval('FusionNet3')(output=1)
    genear = eval('NLayerDiscriminator')
    genear2 = eval('NLayerDiscriminator2')
    # genear3 = eval('out1')
    fusionmodel.cuda()
    fusionmodel.train()
    # fusionmodel.eval()
    # for p in fusionmodel.parameters():  # 关闭权重和偏差的梯度更新
    #     p.requires_grad = False
    genear = genear().cuda()
    genear.train()
    genear2 = genear2().cuda()
    genear2.train()
    # genear3 = genear3().cuda()
    # genear3.train()
    # fusion_model_path = './model/Fusion/fusion_model.pth'
    # fusionmodel.load_state_dict(torch.load(fusion_model_path))
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, fusionmodel.parameters()), lr=lr_start)  # 优化器，fusionmodel.parameters()可优化的参数
    optimizer_G = torch.optim.Adam(fusionmodel.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(genear.parameters(), lr=lr_start)
    optimizer_D2 = torch.optim.Adam(genear2.parameters(), lr=lr_start)
    # optimizer_D3 = torch.optim.Adam(genear3.parameters(), lr=0.0001)
    if num > 0:
        ge__model_path = './model/Fusion/generate/fusion_model.pth'
        genear.load_state_dict(torch.load(ge__model_path))

        ge__model_path1 = './model/Fusion/generate/fusion_model1.pth'
        genear2.load_state_dict(torch.load(ge__model_path1))

        # ge__model_path2 = './model/Fusion/generate/fusion_model2.pth'
        # genear3.load_state_dict(torch.load(ge__model_path2))

        fusion_model_path = './model/Fusion/fusion_model.pth'
        fusionmodel.load_state_dict(torch.load(fusion_model_path))
    #     segmodel.cuda()
    #     segmodel.eval()
    #     # optimizer_seg = torch.optim.Adam(segmodel.parameters(), lr=lr_start)
    #     segmodel.eval()
    #     for p in segmodel.parameters():  # 关闭权重和偏差的梯度更新
    #         p.requires_grad = False
    #     print('Load Segmentation Model {} Sucessfully~'.format(save_pth))



    # fusion_model_path = './model/Fusion/fusion_model.pth'
    # fusionmodel.load_state_dict(torch.load(fusion_model_path))



    train_dataset = Fusion_dataset('train')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(  # 加载数据集并设置
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=8,  # 多少个worker加载数据
        pin_memory=True,  # 将数据加载进锁业内存
        drop_last=True,  # true，会丢弃图片201/10，丢1张
    )
    train_loader.n_iter = len(train_loader)  # s随机搜索次数
    #
    # if num > 0:
    #     score_thres = 0.7
    #     ignore_idx = 255
    #     n_min = 2 * 640 * 480 // 2
    #     criteria_p = OhemCELoss(
    #         thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    #     criteria_16 = OhemCELoss(
    #         thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    #
    criteria_fusion = Fusionloss()  # 融合损失，对抗损失
    loss_msssim = pytorch_msssim.msssim  # ssim损失

    epoch = 10
    st = glob_st = time.time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
    logger.info('Training Fusion Model start~')
    torch_resize = Resize([256, 256])  # 定义Resize类对象
    for epo in range(0, epoch):
        # print('\n| epo #%s begin...' % epo)
        if num > 0:###############################################################
            lr_start = 0.0002
            lr_decay = 0.75
            lr_this_epo = lr_start * lr_decay ** (num - 1)  # 当前学习率
            for param_group in optimizer_G.param_groups:  # 动态修改学习率
                param_group['lr'] = lr_this_epo
            for param_group in optimizer_D.param_groups:  # 动态修改学习率
                param_group['lr'] = lr_this_epo
            for param_group in optimizer_D2.param_groups:  # 动态修改学习率
                param_group['lr'] = lr_this_epo
            # for param_group in optimizer_D.param_groups:  # 动态修改学习率
            #     param_group['lr'] = lr_this_epo
            # for param_group in optimizer_D2.param_groups:  # 动态修改学习率
            #     param_group['lr'] = lr_this_epo
            # for param_group in optimizer_D3.param_groups:  # 动态修改学习率
            #     param_group['lr'] = lr_this_epo
        # elif num > 0:
        #     lr_start = 0.0001
        #     lr_decay = 0.9
        #     lr_this_epo = lr_start * lr_decay ** (num)  # 当前学习率
        #     for param_group in optimizer_G.param_groups:  # 动态修改学习率
        #         param_group['lr'] = lr_this_epo
        #     # for param_group in optimizer_D.param_groups:  # 动态修改学习率
        #     #     param_group['lr'] = lr_this_epo
        #     # for param_group in optimizer_D2.param_groups:  # 动态修改学习率
        #     #     param_group['lr'] = lr_this_epo
        #     # for param_group in optimizer_D3.param_groups:  # 动态修改学习率
        #     #     param_group['lr'] = lr_this_epo

        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()
            # label = Variable(label).cuda()
            logits_g, weight_loss, _ = fusionmodel(image_vis_ycrcb, image_ir)  # 开始训练模型
            image_ir_r = torch_resize(image_ir)
            image_vis_r = torch_resize(image_vis)
            logits_g_r = torch_resize(logits_g)
            pro_0 = genear(image_vis_r[:, :1])
            pro_2 = genear2(image_ir_r)
            pro_1 = genear(logits_g_r)
            pro_3 = genear2(logits_g_r)



            # pro_0 = genear3(image_ir_r,image_vis_r[:, :1])
            # pro_2 = genear3(logits_g_r, logits_g_r)
            # print(torch.mean(pro_0))
            # print(torch.mean(pro_2))
            # legit = Variable(legit.data.clone(), requires_grad=True)
            # logits_g = genear(image_vis_ycrcb, legit)
            # char_act(x_tzt, it, 'train')
            # fusion_ycrcb = torch.cat(  # 在给定维度上对输入的张量序列seq 进行连接操作，将ycrcb图像的cr和cb进行拼接，只融合了亮度那一个维度
            #     (logits_g, image_vis_ycrcb[:, 1:2, :, :],
            #      image_vis_ycrcb[:, 2:, :, :]),
            #     dim=1,  # 在一维链接，2*3和2*3拼接成2*6
            # )  # fusion_ycrcb  2*3*480*640
            # # print(fusion_ycrcb)
            # fusion_image = YCrCb2RGB(fusion_ycrcb)  # 转换成RGB
            # fusion_image = logits
            # ones = torch.ones_like(fusion_image)  # 根据给定的张量生成相通大小的0或1的张量
            # zeros = torch.zeros_like(fusion_image)
            # fusion_image = torch.where(fusion_image > ones, ones,
            #                            fusion_image)  # 根据条件，返回从x,y中选择元素所组成的张量。如果满足条件，则返回x中元素。若不满足，返回y中元素。
            # fusion_image = torch.where(
            #     fusion_image < zeros, zeros, fusion_image)
            # 到这里，融合的图像已经生成了
            # lb = torch.squeeze(label, 1)  # 将tensor中大小为1的维度删除,这里，将1维里大小为一的维度删除

            # seg loss
            # if num > 0:  # 大于0表示已经生成融合图片了，然后对于每张图片，除了自身的融合损失，还加上了分割损失
            #     out, mid = segmodel(fusion_image)  # 这个网络是语义分割
            #     # print(out.size())
            #     lossp = criteria_p(out, lb)  # 语义损失
            #
            #     loss2 = criteria_16(mid, lb)  # 辅助语义损失
            #     seg_loss = lossp + 0.1 * loss2  # 分割网络的loss
            # # fusion loss
            weight_loss = weight_loss.detach()
            loss_fusion, loss_in, loss_grad, w_in = criteria_fusion(
                image_vis_ycrcb, image_ir, weight_loss, logits_g
            )

            max_image = torch.max(image_ir, image_vis_ycrcb[:, :1, :, :])
            # msssim_loss_temp2 = 1 - loss_msssim(logits_g, image_ir, normalize=True)
            # msssim_loss_temp1 = 1 - loss_msssim(logits_g, image_vis_ycrcb[:, :1], normalize=True)
            mx_ssim = 1 - loss_msssim(logits_g, max_image, normalize=True)

            g_loss = -torch.mean(torch.log( pro_1))*weight_loss[0]-torch.mean(torch.log( pro_3))*weight_loss[1]
            D_loss = torch.mean(-torch.log(pro_0)) - torch.mean(torch.log(1 - pro_1))
            D_loss2 = torch.mean(-torch.log(pro_2)) - torch.mean(torch.log(1 - pro_3))
            # g_loss = 1-torch.mean(torch.log( pro_2))
            # D_loss3 = -torch.mean(torch.log(pro_0) + torch.log(1 - pro_2))
            # if num > 0:
            loss_total = 30*loss_fusion+g_loss

            # loss_total = loss_fusion+g_loss
            # loss_total1 = loss_total / accumulation_steps
            # loss_total1.backward()
            # if ((it + 1) % accumulation_steps) == 0:
            #     # optimizer the net
            #     optimizer.step()  # update parameters of net
            #     optimizer.zero_grad()  # reset gradient
            optimizer_G.zero_grad()  # 清空上一次的梯度
            optimizer_D.zero_grad()  # 清空上一次的梯度
            optimizer_D2.zero_grad()
            # optimizer_D3.zero_grad()


            loss_total.backward(retain_graph=True)
            D_loss.backward(retain_graph=True)
            D_loss2.backward(retain_graph=True)

            # D_loss3.backward()

            optimizer_G.step()
            optimizer_D.step()
            optimizer_D2.step()
            # optimizer_D3.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1#train_loader.n_iter=10
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                if num > 0:
                    loss_seg = 0#seg_loss.item()
                else:
                    loss_seg = 0
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'l_total: {loss_total:.4f}',
                        'l_in: {loss_in:.4f}',
                        'l_grad: {loss_grad:.4f}',
                        'g_loss: {g_loss:.4f}',
                        'l_discrim_v: {l_discrim_v:.4f}',
                        'l_discrim_i: {l_discrim_i:.4f}',
                        'weight_g1: {weight_g1:.4f}',
                        'weight_g2: {weight_g2:.4f}',
                        'En_v: {En_v:.4f}',

                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    g_loss=g_loss.item(),
                    l_discrim_v=D_loss,
                    l_discrim_i=D_loss2,
                    weight_g1=weight_loss[0],
                    weight_g2=weight_loss[1],
                    En_v=w_in,

                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed
        # char_act(x_tzt, epo, 'train')
    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)  # 保存学习到的参数
    fusion_model_file1 = os.path.join(modelpth_ge, 'fusion_model.pth')
    torch.save(genear.state_dict(), fusion_model_file1)  # 保存学习到的参数
    fusion_model_file2 = os.path.join(modelpth_ge, 'fusion_model1.pth')
    torch.save(genear.state_dict(), fusion_model_file2)  # 保存学习到的参数
    # fusion_model_file3 = os.path.join(modelpth_ge, 'fusion_model2.pth')
    # torch.save(genear3.state_dict(), fusion_model_file3)  # 保存学习到的参数

    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')


def run_fusion(type='train'):
    fusion_model_path = './model/Fusion/fusion_model.pth'
    ge__model_path = './model/Fusion/generate/fusion_model.pth'
    fused_dir = os.path.join('./MSRS/Fusion', type, 'MSRS')  # 这里test改了
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = eval('FusionNet3')(output=1)
    # generate = eval('FusionNet2')(output=1)
    # fusionmodel = eval('Fusion_network')#(output=1)
    fusionmodel.eval()
    # generate.eval()
    if args.gpu >= 0:
        fusionmodel.cuda(args.gpu)
        # generate.cuda(args.gpu)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    # generate.load_state_dict(torch.load(ge__model_path))
    print('done!')
    test_dataset = Fusion_dataset(type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        # batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():  # 测试集
        for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):  # name:['00001D.png', '00002D.png']
            # print(name)
            # print(len(name))
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            labels = Variable(labels)
            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
            images_vis_ycrcb = RGB2YCrCb(images_vis)
            logits,_,_ = fusionmodel(images_vis_ycrcb, images_ir)
            # logits = generate(images_vis, logits)
            fusion_image = logits
            # if (it + 1) % 10 == 0:
            #     char_act(x_visualize, it, 'test')
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :,:], images_vis_ycrcb[:, 2:, :, :]),
                dim=1
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)  # uint8专门储存图像的，范围为0-255
            for k in range(len(name)):  # len(name)=2
                image = fused_image[k, :, :, :]
                image = image.squeeze()  # 维度为1的去掉
                image = Image.fromarray(image)  # array转化维image
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SeAFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)  # 16
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=1)  # 8
    args = parser.parse_args()
    # modelpth = './model'
    # Method = 'Fusion'
    # modelpth = os.path.join(modelpth, Method)
    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    for i in range(0, 1):#4
        # train_fusion(i, logger)
        # print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
        run_fusion('test')
        # print("|{0} Fusion Image Sucessfully~!".format(i + 1))
        # train_seg(i, logger)
        # print("|{0} Train Segmentation Model Sucessfully~!".format(i + 1))
    print("training Done!")
    # os.system("shutdown")