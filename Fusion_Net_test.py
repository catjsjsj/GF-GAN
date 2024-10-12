#encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OneConvBnRelu2d(nn.Module):
    ''' 1*1 '''

    def __init__(self, in_channels, out_channels,groups=1):
        super(OneConvBnRelu2d,self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False,groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.Conv(x)
class OneConvLeakeyReLu2d(nn.Module):
    ''' 1*1 '''

    def __init__(self, in_channels, out_channels,groups=1):
        super(OneConvLeakeyReLu2d,self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False,groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.Conv(x)

class ConvBnPRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', stride=1, dilation=1, groups=1):
        super(ConvBnPRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.Prl = nn.PReLU()

    def forward(self, x):
        return self.Prl(self.conv(x))
        # return F.relu(self.conv(x), inplace=True)

class ConvBn2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', stride=1, dilation=1, groups=1):
        super(ConvBnPRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels),


    def forward(self, x):
        return self.bn(self.conv(x))
        # return F.relu(self.conv(x), inplace=True)
class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels,momentum=0.9)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


class OneConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(OneConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))  # 向下取整
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


class ConvBnReLu2d(nn.Module):
    def __init__(self, in_channels, out_channels,groups=1):
        super(ConvBnReLu2d,self).__init__()
        self.ones_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,groups=groups),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(negative_slope=0.2)
            nn.ReLU()
        )

    def forward(self, x):
        return self.ones_conv(x)

class ConvBnLeakeyReLu2d(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(ConvBnLeakeyReLu2d,self).__init__()
        self.ones_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=group),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.ones_conv(x)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBn,self).__init__()
        self.ones_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
        )

    def forward(self, x):
        return self.ones_conv(x)


class OneConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OneConvBn,self).__init__()
        self.ones_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
        )

    def forward(self, x):
        return self.ones_conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv,self).__init__()
        self.sconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            # nn.LeakyReLU(negative_slope=0.2)
            nn.ReLU()
        )

    def forward(self, x):
        return self.sconv(x)


class Laplace(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Laplace, self).__init__()
        # laplace_filter1 = np.array([[0, -1, 0],
        #                             [-1, 4, -1],
        #                             [0, -1, 0]])
        laplace_filter2 = np.array([[-1, -1, -1],
                                    [-1, 8, -1],
                                    [-1, -1, -1]])
        # self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
        #                        dilation=dilation, groups=channels, bias=False)
        # self.convx.weight.data.copy_(torch.from_numpy(laplace_filter1))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(laplace_filter2))

    def forward(self, x):
        # laplacex = self.convx(x)
        laplacey = self.convy(x)
        # x = torch.abs(laplacex) + torch.abs(laplacey)
        return laplacey


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class SEAttention(nn.Module):

    def __init__(self, channel, reduction=16):  # channel为输入通道数，reduction压缩大小
        super(SEAttention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y = torch.tensor([item.cpu().detach().numpy() for item in y])
        y = self.fc(y).view(b, c, 1, 1)
        return torch.mul(x, y)


''' 浅层特征提取'''


# class Conv16(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Conv16, self).__init__()
#         self.conv1 = ConvBnReLu2d(in_channels, out_channels)
#         self.conv2 = ConvBnReLu2d(in_channels, out_channels)
#
#     def forward(self, x, y):
#         x = self.conv1(x)
#         y = self.conv2(y)
#
#         return x, y


class ExConv_vis(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExConv_vis, self).__init__()
        self.conv1_vis = ConvBnReLu2d(in_channels, out_channels)
        self.conv2_vis = ConvBnReLu2d(out_channels, 2 * out_channels)
        self.conv3_vis = OneConvBn(2 * out_channels, out_channels)
        self.conv4_vis = OneConvBnRelu2d(out_channels, out_channels)
        self.laplace_vis = Sobelxy(out_channels)

        # self.conv1_inf = ConvBnReLu2d(in_channels, out_channels)
        # self.conv2_inf = ConvBnReLu2d(out_channels, 2 * out_channels)
        # self.conv3_inf = ConvBnReLu2d(2 * out_channels, out_channels)
        # self.conv4_inf = OneConvBnRelu2d(out_channels, out_channels)
        # self.laplace_inf = Sobelxy(out_channels)
        # self.one1 = OneConvBnRelu2d(out_channels, 2 * out_channels)
        # self.one2 = OneConvBnRelu2d(out_channels, 2 * out_channels)

    def forward(self, x):
        x1 = self.conv1_vis(x)

        x2 = self.conv2_vis(x1)
        x3 = self.conv3_vis(x2)
        x4 = self.laplace_vis(x1)
        x4 = self.conv4_vis(x4)
        # x7,_ = torch.max(x4,dim=1)
        # x7 = torch.unsqueeze(x7, 1)
        # x_visualize = F.interpolate(x_visualize, size=(640, 480), mode='bilinear',
        #                             align_corners=False)  # 这种做法费时间 shape为[1,256,513,513]
        # y2 = self.conv2_inf(y1)
        # y3 = self.conv3_inf(y2)
        # y4 = self.laplace_inf(y1)
        # y4 = self.conv4_inf(y4)

        x5 = torch.add(x3, x4)
        x5 = torch.add(x5, x1)

        # y6 = torch.add(y4, y3)
        # x_visualize = y6
        return F.relu(x5)


''' 浅层特征融合,生成图片'''


class ExConv_rh(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ExConv_rh, self).__init__()
        self.conv1 = ConvBnReLu2d(in_channels, mid_channels)
        self.conv2 = ConvBnReLu2d(mid_channels, mid_channels)
        self.conv3 = ConvBnTanh2d(mid_channels, out_channels)
        self.conv4 = OneConvBnRelu2d(in_channels, mid_channels)
        self.laplace = Sobelxy(in_channels)

    def forward(self, x):
        # xy = torch.cat((x, y), 1)
        xy1 = self.conv1(x)
        xy2 = self.conv2(xy1)
        laplace = self.conv4(self.laplace(x))
        xy3 = torch.add(laplace, xy2)
        xy4 = self.conv3(xy3)

        return xy4


''' 浅层特征融合'''


class FuseConv_rh(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FuseConv_rh, self).__init__()
        self.conv1 = ConvBnReLu2d(in_channels, out_channels)
        self.conv2 = ConvBnReLu2d(out_channels, out_channels)

    def forward(self, x, y):
        xy = torch.cat((x, y), 1)
        xy1 = self.conv1(xy)
        xy2 = self.conv2(xy1)

        return xy2


''' 深层特征融合BLOCK '''


class deepConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(deepConv, self).__init__()
        self.conv1 = ConvBnReLu2d(mid_channels, mid_channels)
        self.conv2 = ConvBnReLu2d(mid_channels, mid_channels)
        self.conv3 = OneConvBnRelu2d(in_channels, mid_channels)
        self.conv4 = OneConvBnRelu2d(mid_channels, out_channels)
        self.conv5 = OneConvBnRelu2d(mid_channels, out_channels)

    def forward(self, x, y):
        diffY = torch.tensor([x.size()[2] - y.size()[2]])
        diffX = torch.tensor([x.size()[3] - y.size()[3]])

        y = F.pad(y, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        xy = torch.cat((x, y), 1)
        xy1 = self.conv3(xy)
        xy2 = self.conv1(xy1)
        xy3 = self.conv2(xy2)
        xy4 = self.conv4(xy3)

        y = self.conv5(y)
        xy4 = torch.add(y, xy4)

        return xy4


class en_deFuse(nn.Module):
    def __init__(self, in_channels):
        super(en_deFuse, self).__init__()
        self.conv1 = ConvBnReLu2d(in_channels, in_channels)
        self.conv2 = ConvBn(in_channels, in_channels)
        self.conv3 = ConvBnReLu2d(2 * in_channels, in_channels)

    def forward(self, x, y):
        # xy = torch.add(x, y)
        xy = self.conv3(torch.cat((x, y), 1))
        xy1 = self.conv1(xy)
        xy2 = self.conv2(xy1)
        xy3 = torch.add(F.relu(xy), xy2)

        return F.relu(xy3)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down,self).__init__()
        self.down_pool1 = nn.Sequential(
            OneConvBnRelu2d(in_channels, in_channels // 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            ConvBnReLu2d(in_channels // 2, out_channels),
            ConvBnReLu2d(out_channels, out_channels)
        )

    def forward(self, x):
        x1 = self.down_pool1(x)
        return x1


"""ConvDown"""


class Down_vis(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_vis,self).__init__()
        self.F_conv1 = nn.Sequential(
            ConvBnReLu2d(in_channels, in_channels),
            DownConv(in_channels, in_channels),
            OneConvBnRelu2d(in_channels, out_channels)
        )
        self.F_conv2 = nn.Sequential(
            Sobelxy(in_channels),
            DownConv(in_channels, in_channels),
            OneConvBnRelu2d(in_channels, out_channels)
        )

    def forward(self, x):
        x1 = self.F_conv1(x)
        x2 = self.F_conv2(x)
        x3 = torch.add(x1, x2)
        # x3 = F.leaky_relu(x3, negative_slope=0.2)
        x3 = F.relu(x3)
        return x3


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = OneConvBnRelu2d(in_channels, out_channels)
            self.conv1 = ConvBnReLu2d(out_channels, out_channels)
            self.conv2 = ConvBnReLu2d(out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = ConvBnReLu2d(in_channels, in_channels)
            self.conv1 = OneConvBnRelu2d(in_channels, out_channels)
            self.conv2 = OneConvBnRelu2d(in_channels, out_channels)

    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = self.conv1(x2)
        x5 = self.up(x3)
        x6 = self.conv2(x5)
        return x6

        # x1 = self.conv2(x1)
        # x2 = self.up(x1)
        # diffY = torch.tensor([y1.size()[2] - x2.size()[2]])
        # diffX = torch.tensor([y1.size()[3] - x2.size()[3]])
        #
        # x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
        #                     diffY // 2, diffY - diffY // 2])
        # x2 = torch.cat((x2, y1), 1)
        # x3 = self.conv(x2)
        # x4 = self.conv1(x3)

        # return x4


class fusion_m(nn.Module):
    def __init__(self, in_channels, out_channels):  # 256  128
        super(fusion_m, self).__init__()
        self.fusion_1 = nn.Sequential(
            ConvBnReLu2d(in_channels, out_channels),
            ConvBnReLu2d(out_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9)
        )
        self.fusion_2 = nn.Sequential(
            Sobelxy(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9)
        )

    def forward(self, x):
        x1 = self.fusion_1(x)
        x2 = self.fusion_2(x)
        # return F.leaky_relu(torch.add(x1, x2), negative_slope=0.2, inplace=True)
        return F.relu(torch.add(x1, x2), inplace=True)


class fusion(nn.Module):

    def __init__(self, in_channels, out_channels):  # 256  128
        super(fusion, self).__init__()
        self.add_fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            # nn.LeakyReLU()
            nn.ReLU()
        )

        self.ch_fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            # nn.LeakyReLU()
            nn.ReLU()
        )
        self.SEAttention_add = SEAttention(out_channels)
        self.SEAttention_cat = SEAttention(in_channels)
        self.Out_fusion = fusion_m(in_channels, out_channels)

    def forward(self, x1, x2):
        diffY = torch.tensor([x1.size()[2] - x2.size()[2]])
        diffX = torch.tensor([x1.size()[3] - x2.size()[3]])

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x3 = torch.add(x1, x2)
        x3 = self.add_fusion(x3)
        x4 = torch.cat((x1, x2), dim=1)
        x4 = self.ch_fusion(x4)
        # x5 = F.leaky_relu(torch.cat((x3, x4), dim=1), negative_slope=0.18)
        x5 = F.relu(torch.cat((x3, x4), dim=1))

        x5 = self.Out_fusion(x5)

        return x5


class Denseblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Denseblock, self).__init__()
        mid_channles = int((in_channels + out_channels) / 2)
        self.conv1 = ConvLayer(in_channels, mid_channles, 3, 1)
        self.conv2 = ConvLayer(mid_channles, out_channels, 1, 1)
        self.conv3 = ConvLayer(out_channels * 2, out_channels * 2, 3, 1)
        self.conv4 = ConvLayer(out_channels * 2, out_channels, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.up(x1)
        x1 = self.conv2(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = self.conv3(torch.cat((x1, x2), dim=1))
        x = self.conv4(x)

        return x
