#encoding:utf-8
import torch
#encoding:utf-8
from Fusion_Net_test import *
import math

import numpy as np
class EnergyFusion(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(EnergyFusion, self).__init__()
        laplace_filter1 = np.array([[1/4, 2/4, 1/4],
                                    [2/4, 4/4, 2/4],
                                    [1/4, 2/4, 1/4]])
        laplace_filter2 = np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)


        self.convx.weight.data.copy_(torch.from_numpy(laplace_filter1))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(laplace_filter2))

        self.conv_v_weight = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,padding='same',bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.conv_v_weight2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.Sigmoid()
        )
        self.SOB = Sobelxy(channels)
        self.conv1 = ConvBnLeakeyReLu2d(2,2)
        self.conv2 = OneConvLeakeyReLu2d(2,1)
        self.conv3 = ConvBnLeakeyReLu2d(2,1)
        self.BN = nn.BatchNorm2d(channels)
        # self.conv_i_weight = OneConvBnRelu2d(2, 1, 1)
    def forward(self, x):
        xs = self.SOB(x)
        x1 = self.convx(xs+x)
        # y1 = self.convx(y)
        x_w = self.conv_v_weight(x1)
        # y_w = self.conv_v_weight2(y1)
        x2 = torch.mul(x, x_w) + x
        # y2 = torch.mul(y, y_w)+y
        return x2

class Spa(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spa, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)
class Fuse_model(nn.Module):
    def __init__(self, in_channel, mid_channel1, mid_channel2, out_channel):
        super(Fuse_model, self).__init__()
        self.conv1 = ConvBnReLu2d(in_channel, mid_channel1)
        self.conv2 = ConvBnReLu2d(mid_channel1, mid_channel2)
        self.conv3 = ConvBnReLu2d(mid_channel2, out_channel)#ConvBnReLu2d(mid_channel, out_channel)
        self.conv4 = ConvBnTanh2d(out_channel, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x4


class Dusion_model(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Dusion_model, self).__init__()
        # self.conv1 = ConvBnReLu2d(in_channel, in_channel, groups=in_channel)
        self.conv2 = ConvBnLeakeyReLu2d(in_channel, out_channel)
        # self.conv3 = OneConvBnRelu2d(in_channel, out_channel)#, groups=out_channel

    def forward(self, x):
        # x1 = self.conv1(x)
        x2 = self.conv2(x)
        # x3 = self.conv3(x2)
        return x2
class dec_model(nn.Module):
    def __init__(self, in_channel,mid_channel,out_channel,):
        super(dec_model, self).__init__()
        self.conv1 = ConvBnLeakeyReLu2d(mid_channel, out_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, padding='same', bias=False,groups=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.01)
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=5,padding='same'),
        #     nn.BatchNorm2d(out_channel),
        #     nn.LeakyReLU(negative_slope=0.2)
        # )
        self.conv3 = OneConvLeakeyReLu2d(out_channel,out_channel)
        self.conv4 = OneConvLeakeyReLu2d(in_channel,mid_channel)
        self.conv5 = OneConvBnRelu2d(in_channel, out_channel)
        self.sob = Sobelxy(in_channel)

    def forward(self, x):
        # x1 = self.conv4(x)
        # lp = self.sob(x)
        # lpx = self.conv5(lp)
        #
        # x2 = self.conv1(x1)+lpx
        # x3 = self.conv2(x2)
        # x4 = x2+x3
        # x5 = self.conv3(x4)
        x1 = self.conv4(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x5 = self.conv3(x3)
        return x5


class dec_mode1(nn.Module):
    def __init__(self, in_channel,mid_channel,out_channel,):
        super(dec_model, self).__init__()
        self.conv1 = ConvBnLeakeyReLu2d(mid_channel, out_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, padding='same', bias=False,groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.01)
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=5,padding='same'),
        #     nn.BatchNorm2d(out_channel),
        #     nn.LeakyReLU(negative_slope=0.2)
        # )
        self.conv3 = OneConvLeakeyReLu2d(out_channel,out_channel)
        self.conv4 = OneConvLeakeyReLu2d(in_channel,mid_channel)
        self.conv5 = OneConvBnRelu2d(in_channel, out_channel)
        self.sob = Sobelxy(in_channel)

    def forward(self, x):
        x1 = self.conv4(x)
        lp = self.sob(x)
        lpx = self.conv5(lp)

        x2 = self.conv1(x1)+lpx
        x3 = self.conv2(x2)
        x4 = x2+x3
        x5 = self.conv3(x4)
        return x5

class Dense_model(nn.Module):
    def __init__(self, channel,out_channel):
        super(Dense_model, self).__init__()
        self.conv1 = OneConvLeakeyReLu2d(channel, out_channel)
        self.conv2 = ConvBnLeakeyReLu2d(out_channel,out_channel)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x2


class SpatialAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (1, 3, 7), 'kernel size must be 3 or 7'
        if kernel_size == 1:
            padding = 0
        else:
            padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, channel, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.relu = nn.Sigmoid()

    def forward(self, x, y, fg=0):
        if fg == 0:
            x = torch.cat((x, y), 1)
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            # min_out, _ = torch.min(x, dim=1, keepdim=True)
            x = torch.cat((avg_out, max_out), dim=1)
            x = self.conv2(x)
        if fg == 1:
            x = torch.cat([x, y], 1)
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            # min_out, _ = torch.min(x, dim=1, keepdim=True)
            x = torch.cat((max_out, avg_out), 1)
            x = self.conv1(x)
        return self.relu(x)#改成sigmoid


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class Fusion_model(nn.Module):
    def __init__(self, channel):
        super(Fusion_model, self).__init__()
        # self.sob = Sobelxy(channel)
        # self.sob_y = Sobelxy(1)
        # self.aten_i = SpatialAttention(channel, 3)
        # self.aten_vi = SpatialAttention(channel, 1)
        # self.aten_all = SpatialAttention(channel, 1)
        # self.fuse = ConvBnReLu2d(3,1)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=5, padding='same', stride=1),
        #     nn.BatchNorm2d(1),
        #     nn.Sigmoid())
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=5, padding='same', stride=1),
        #     nn.BatchNorm2d(1),
        #     nn.Sigmoid())
        self.sp = Spa(7)

    def forward(self, image_ir, image_vis, ir_features, vis_features):
        # x_vis = image_vis
        # x_inf = image_ir
        # feature_all = self.aten_all(self.conv1(image_ir), self.conv2(image_vis), 1)
        # feature_all_weight = torch.split(feature_all, 1, dim=1)
        #
        # x_inf_s = self.sob_y(x_inf)
        # x_vis_s = self.sob_y(x_vis)
        # feature_s_iv = torch.max(x_vis_s,x_inf_s)
        #
        # ir_features_split = torch.split(ir_features, 1, dim=1)
        # vis_features_split = torch.split(vis_features, 1, dim=1)
        ad = ir_features+vis_features
        feature_all_weight = torch.split(ad, 1, dim=1)
        flag = 0
        # for feature_i, feature_v, feature_all_w in zip(ir_features_split, vis_features_split,
        #                                                         feature_all_weight
        #                                                         ):
        for fe in feature_all_weight:
            # feature_f = self.fuse(torch.cat([feature_all_w,feature_i,feature_v],1))
            # feature_s = self.sob_y(feature_f)
            # feature_s_w = self.aten_vi(feature_s, feature_s_iv)
            # feature_f_ok = torch.add(torch.mul(feature_s_w, feature_s), feature_f)
            feature_f_ok = torch.mul(self.sp(fe),fe)

            if flag == 0:
                n_f_all = feature_f_ok
                flag = 1
                continue
            if flag:
                n_f_all = torch.cat((feature_f_ok, n_f_all), dim=1)

        return n_f_all


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class TG(nn.Module):
    def __init__(self):
        super(TG, self).__init__()
        self.vis1 = ConvBnPRelu2d(32, 32)
        self.vis2 = ConvBnPRelu2d(32, 64)
        # self.max1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.inf1 = ConvBnPRelu2d(32, 32)
        self.inf2 = ConvBnPRelu2d(32, 64)
        # self.max2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.eng1 = EnergyFusion(16)
        self.eng2 = EnergyFusion(16)
        self.eng3 = EnergyFusion(128)

        self.de1 = ConvBnPRelu2d(128,128)
        self.de2 = ConvBnPRelu2d(128, 64)
        self.de3 = ConvBnPRelu2d(128, 64)
        self.de4 = ConvBnPRelu2d(64, 32)
        self.de5 = ConvBnPRelu2d(32, 16)
        self.de6 = ConvBnPRelu2d(16, 1)

    def forward(self, vis, inf):
        en1 = self.eng1(inf)
        en2 = self.eng2(vis)
        vis1 = self.vis1(torch.cat((en1, vis), 1))
        inf1 = self.inf1(torch.cat((en2, inf), 1))
        vis2 = self.vis2(vis1)
        inf2 = self.inf2(inf1)

        dec1 = self.de1(torch.cat((vis2, inf2), 1))
        en3 = self.eng3(dec1)
        dec2 = self.de2(en3 + dec1)
        vi = torch.cat((vis1, inf1), 1)
        dec3 = self.de3(torch.cat((vi, dec2), 1))
        dec4 = self.de4(dec3)
        dec5 = self.de5(dec4)
        out = self.de6(dec5)

        return out

class FusionNet1(nn.Module):
    def __init__(self):
        super(FusionNet1, self).__init__()
        self.vis_encoder1 = ConvBnPRelu2d(in_channels=1, out_channels=16, kernel_size=3)
        self.vis_encoder2 = ConvBnPRelu2d(in_channels=16, out_channels=16, kernel_size=3)
        self.vis_encoder3 = ConvBnPRelu2d(in_channels=32, out_channels=16, kernel_size=3)

    def forward(self, image_vis):
        x1 = self.vis_encoder1(image_vis)
        x2 = self.vis_encoder2(x1)
        x3 = torch.cat((x1, x2), 1)
        x4 = self.vis_encoder3(x3)

        return torch.cat((x3, x4), 1)


class FusionNet2(nn.Module):
    def __init__(self):
        super(FusionNet2, self).__init__()
        self.inf_encoder1 = ConvBnPRelu2d(in_channels=1, out_channels=16, kernel_size=3)
        self.inf_encoder2 = ConvBnPRelu2d(in_channels=16, out_channels=16, kernel_size=3)
        self.inf_encoder3 = ConvBnPRelu2d(in_channels=32, out_channels=16, kernel_size=3)

    def forward(self, image_ir):
        x1 = self.inf_encoder1(image_ir)
        x2 = self.inf_encoder2(x1)
        x3 = torch.cat((x1, x2), 1)
        x4 = self.inf_encoder3(x3)

        return torch.cat((x3, x4), 1)


class FusionNet3(nn.Module):
    def __init__(self, output):
        super(FusionNet3, self).__init__()



        self.v_en = FusionNet1()
        self.i_en = FusionNet2()
        self.sbl = Sobelxy(channels=48)

        self.fc = nn.Sequential(
            nn.Linear(96, 128),

            nn.ReLU(),
            nn.Linear(128, 128),

            nn.ReLU(),
            nn.Linear(128, 96),
            nn.Sigmoid()
        )


        self.de1 = ConvBnPRelu2d(96, 64)
        self.de2 = ConvBnPRelu2d(64, 32)
        self.de3 = ConvBnPRelu2d(32, 16)
        self.de4 = ConvBnTanh2d(16, 1)
        # freeze(self.v_en)
        # freeze(self.i_en)

        # self.tg = TG()

    def forward(self, image_vis, image_ir):

        ven_3 = self.v_en(image_vis[:,:1,:,:])
        ien_3 = self.i_en(image_ir)
        ir = self.sbl(ien_3)
        vis = self.sbl(ven_3)
        gra_ir = torch.norm(ir, dim=(2,3))/(ir.size(2)*ir.size(3))/4
        grair_sum = torch.sum(gra_ir, dim=1).unsqueeze(1)

        gra_vis = torch.norm(vis, dim=(2,3))/(vis.size(2)*vis.size(3))/4
        gravis_sum = torch.sum(gra_vis, dim=1).unsqueeze(1)
        x=torch.cat((gravis_sum,grair_sum),1)/0.002
        weight_l = F.softmax(x,dim=1)
        weight_loss = torch.sum(weight_l, dim=0)/4

        all = torch.cat((gra_vis, gra_ir), 1)
        weight_all = self.fc(all).view(1, 96, 1, 1)
        vi_ir = torch.mul(weight_all, torch.cat((ven_3, ien_3), 1))
        # vi_ir = self.eng(torch.cat((ven_3, ien_3), 1))

        # rh1 = self.rh1(vi_ir)#96-44
        # d1 = torch.cat((vi_ir, rh1), 1)#140
        # rh2 = self.rh2(d1)#140-44
        # d2 = torch.cat((d1, rh2), 1)#184
        # rh3 = self.rh3(d2)#184-44
        # d3 = torch.cat((d2, rh3), 1)#238
        # rh4 = self.rh4(d3)#238-44
        # d4 = torch.cat((d3, rh4), 1)#282

        out1 = self.de1(vi_ir)
        out2 = self.de2(out1)
        out3 = self.de3(out2)
        out4 = self.de4(out3)

        return out4, weight_loss, vi_ir

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func != nn.BatchNorm2d
        # else:
        use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=2, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        self.BN = nn.BatchNorm2d(1)
        self.sig = nn.Sigmoid()
    def forward(self, input):
        """Standard forward."""
        # print(input.shape)
        x=self.model(input)
        # x = self.BN(x)
        x=self.sig(x)
        return torch.mean(x, dim=(2,3))

class NLayerDiscriminator2(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator2, self).__init__()
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func != nn.BatchNorm2d
        # else:
        use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=2, padding=padw)]  # output 1 channel prediction map
        self.BN = nn.BatchNorm2d(1)
        self.model = nn.Sequential(*sequence)
        self.sig = nn.Sigmoid()
    def forward(self, input):
        """Standard forward."""
        # print(input.shape)
        x=self.model(input)
        # x = self.BN(x)
        x=self.sig(x)
        return torch.mean(x, dim=(2,3))

class out1(nn.Module):
    def __init__(self):
        super(out1, self).__init__()
        self.inf = NLayerDiscriminator()
        self.vis = NLayerDiscriminator2()
        self.conv1 = ConvBnLeakeyReLu2d(2,16)
        self.conv2 = ConvBnLeakeyReLu2d(16, 8)
        self.conv3 = ConvBnLeakeyReLu2d(8, 1)
        self.fc = nn.Sequential(
            nn.Linear(1131, 1024, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=False),
            nn.Sigmoid()
        )
        self.en = EnergyFusion(1)

    def forward(self, image_ir, image_vis):


        ir = self.inf(self.en(image_ir)-image_ir)
        vis = self.vis(self.en(image_vis)-image_vis)
        all = self.conv1(torch.cat((ir, vis), 1))
        all = self.conv2(all)
        all = self.conv3(all)
        all = torch.squeeze(all)
        image = all.view(all.size(0), -1)
        out = self.fc(image)

        return out