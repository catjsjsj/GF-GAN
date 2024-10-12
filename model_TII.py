#!/usr/bin/python
# -*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from resnet import Resnet18
# from modules.bn import InPlaceABNSync as BatchNorm2d






from PIL import Image
import numpy as np
from torch.autograd import Variable
from FusionNet1 import FusionNet1
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from cityscapes import CityScapes
from loss import OhemCELoss, Fusionloss
from optimizer import Optimizer
import torch
from torch.utils.data import DataLoader
import warnings
import pytorch_msssim

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ConvBNSig(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNSig, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid_atten(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class Attentionout(nn.Module):
    def __init__(self, out_chan, *args, **kwargs):
        super(Attentionout, self).__init__()
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1,bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        atten = self.conv_atten(x)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(x, atten)
        x = x+out
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class SAR(nn.Module):
    def __init__(self, in_chan, mid, out_chan, *args, **kwargs):
        super(SAR, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, out_chan, 3, 1, 1)
        self.conv_reduce = ConvBNReLU(in_chan,mid,1,1,0)
        self.conv_atten = nn.Conv2d(2, 1, kernel_size= 3, padding=1,bias=False)
        self.bn_atten = nn.BatchNorm2d(1)
        self.sigmoid_atten = nn.Sigmoid()
    def forward(self, x):
        x_att = self.conv_reduce(x)
        low_attention_mean = torch.mean(x_att,1,True)
        low_attention_max = torch.max(x_att,1,True)[0]
        low_attention = torch.cat([low_attention_mean,low_attention_max],dim=1)
        spatial_attention = self.conv_atten(low_attention)
        spatial_attention = self.bn_atten(spatial_attention)
        spatial_attention = self.sigmoid_atten(spatial_attention)
        x = x*spatial_attention
        x = self.conv1(x)
        #channel attention
 #       low_refine = self.conv_ca_rf(low_refine)
        return x
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class SeparableConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1):
        super(SeparableConvBnRelu, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding, dilation, groups=in_channels,
                               bias=False)
        self.point_wise_cbr = ConvBNReLU(in_channels, out_channels, 1, 1, 0)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.point_wise_cbr(x)
        return x
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
#        self.conv_32 = ConvBNReLU(512, 128, ks=3, stride=1, padding=1)
#        self.conv_16 = ConvBNReLU(256, 128, ks=3, stride=1, padding=1)
#        self.conv_8 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm8 = AttentionRefinementModule(128, 128)
        self.sp16 =  ConvBNReLU(256, 128, ks=1, stride=1, padding=0)
        self.sp8 = ConvBNReLU(256, 128, ks=1, stride=1, padding=0)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
#        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
#        self.conv_context = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.conv_fuse1 = ConvBNSig(128,128,ks=1,stride=1, padding=0)
        self.conv_fuse2 = ConvBNSig(128,128,ks=1,stride=1, padding=0)
        self.conv_fuse = ConvBNReLU(128, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        _, feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

#        avg = F.avg_pool2d(feat32, feat32.size()[2:])
#        avg = self.conv_avg(avg)
#        avg_up = F.interpolate(avg, (H8, W8), mode='nearest')
        feat32_arm = self.arm32(feat32)
        feat32_cat = F.interpolate(feat32_arm, (H8, W8), mode='bilinear')
#        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_arm, (H16, W16), mode='bilinear')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_cat = torch.cat([feat32_up,feat16_arm], dim=1)
        feat16_cat = self.sp16(feat16_cat)
        feat16_cat = F.interpolate(feat16_cat, (H8, W8), mode='bilinear')
        feat16_up = F.interpolate(feat16_arm, (H8, W8), mode='bilinear')
        feat16_up = self.conv_head16(feat16_up)
        
        feat8_arm = self.arm8(feat8)
        feat8_cat = torch.cat([feat16_up,feat8_arm], dim=1)
        feat8_cat = self.sp8(feat8_cat)        
        
        feat16_atten = self.conv_fuse1(feat32_cat)
        feat16_cat = feat16_atten*feat16_cat
        
        feat8_atten = self.conv_fuse2(feat16_cat)
        feat8_out = feat8_cat*feat8_atten
        
        
#        feat8_out = torch.cat([feat8_cat,feat16_cat,feat32_cat], dim=1)
        feat8_out = self.conv_fuse(feat8_out)
        return feat8_out, feat16_arm, feat32_arm # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        ## here self.sp is deleted
#        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(128, 128, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
#        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x) # here return res3b1 feature
#        feat_sp = feat_res8 # use res3b1 feature to replace spatial path feature
#        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_res8)
        feat_out16 = self.conv_out16(feat_cp8)
#        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
#        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    # net = BiSeNet(19)
    # net.cuda()
    # net.eval()
    # in_ten = torch.randn(2, 3, 640, 480).cuda()#16.3.640.480
    # out, out16 = net(in_ten)
    # print(out.shape)
    #
    #
    # net.get_params()
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
    i= 0
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
        print(im.size())
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
            print(msg)
            print(out.shape)
            # plt.show(out)
            loss_avg = []
            st = ed
    # dump the final model
    save_pth = osp.join(modelpth, 'model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    # # logger.info(
    #     'Segmentation Model Training done~, The Model is saved to: {}'.format(
    #         save_pth)
    # )
    # # logger.info('\n')
