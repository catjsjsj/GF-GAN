#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_msssim
import numpy as np
import cv2





class OhemCELoss(nn.Module):#其中 thresh 表示的是，损失函数大于多少的时候，会被用来做反向传播。n_min 表示的是，在一个 batch 中，最少需要考虑多少个样本。
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')#设置 reduction 为 none，保留每个元素的损失，返回的维度为 N\times H\times W。

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)#将预测的损失拉平为一个长向量，每个元素为一个 pixel 的损失。
        loss, _ = torch.sort(loss, descending=True)#将预测的损失拉平为一个长向量，每个元素为一个 pixel 的损失。
        if loss[self.n_min] > self.thresh:#最少考虑 n_min 个损失最大的 pixel，如果前 n_min 个损失中最小的那个的损失仍然大于设定的阈值，那么取实际所有大于该阈值的元素计算损失：loss=loss[loss>thresh]。
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)#这些 hard example 的损失的均值作为最终损失


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(nn.FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class NormalLoss(nn.Module):
    def __init__(self,ignore_lb=255, *args, **kwargs):
        super( NormalLoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels)
        return torch.mean(loss)


def calc_array(img):
    # img = cv2.imread('20201210_3.bmp',0)
    # img = np.zeros([16,16]).astype(np.uint8)

    #

    hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])  # [0,256]的范围是0~255.返回值是每个灰度值出现的次数

    # plt.subplot(111)
    # plt.plot(hist_cv)
    # plt.show()
    lst = []
    P = hist_cv / (len(img[1]) * len(img[0]))  # 概率
    for p in P:
        if p!=0:
            a = p * np.log2(1 / p)
            lst.append(a)
    E = np.sum(lst)


    return E # 熵

def zhuanhuan(img):
    y_t = (((img - np.min(img)) / (
            np.max(img) - np.min(img))) * 255).astype(np.uint8)
    return y_t

def fenli(img):
    img = img.squeeze()
    img1, img2, img3, img4 = torch.split(img, split_size_or_sections=1, dim=0)

    img1 = img1.squeeze().cpu().numpy()
    img2 = img2.squeeze().cpu().numpy()
    img3 = img3.squeeze().cpu().numpy()
    img4 = img4.squeeze().cpu().numpy()
    img1 = zhuanhuan(img1)
    img2 = zhuanhuan(img2)
    img3 = zhuanhuan(img3)
    img4 = zhuanhuan(img4)
    E1 = calc_array(img1)
    E2 =calc_array(img2)
    E3 =calc_array(img3)
    E4 =calc_array(img4)

    return (E1+E2+E3+E4)/4



class Fusionloss(nn.Module):

    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        self.L2 = nn.MSELoss(reduction='none',reduce=True, size_average=True)

    def forward(self,image_vis,image_ir,weight,generate_img):
        # image_y = image_vis[:, :1, :, :]
        # loss_in = F.l1_loss(image_y, generate_img)
        # ir_grad = self.sobelconv(image_y)
        # generate_img_grad = self.sobelconv(generate_img)
        # loss_grad = F.l1_loss(ir_grad, generate_img_grad)
        # loss_tal = loss_grad+loss_in
        #
        # return  loss_tal, loss_in,loss_grad
        # loss_msssim = pytorch_msssim.msssim
        # return loss_tal, loss_in, loss_grad
        image_y=image_vis[:,:1,:,:]

        image_y_en = torch.unsqueeze(torch.Tensor(np.array(fenli(image_y))).cuda(), dim=0)
        image_ir_en = torch.unsqueeze(torch.Tensor(np.array(fenli(image_ir))).cuda(), dim=0)
        gl = torch.cat((image_y_en, image_ir_en), 0)

        x_in_max=torch.max(image_y,image_ir)

        loss_in1=self.L2(image_y,generate_img)  # 与可见光图像的像素损失
        loss_in2 = self.L2(image_ir,generate_img )  # 与红外图像的像素损失
        # loss_in=self.L2(generate_img,x_in_max)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        # y_n = torch.norm(image_y,p=1,  dim=(2, 3))/(image_y.size(2)*image_y.size(3))
        # i_n = torch.norm(image_ir,p=1,  dim=(2, 3))/(image_ir.size(2)*image_ir.size(3))
        # grair_sum = torch.sum(i_n, dim=1).unsqueeze(1)
        # gravis_sum = torch.sum(y_n, dim=1).unsqueeze(1)
        # x = torch.cat((gravis_sum, grair_sum), 1)/0.4
        weight_in = F.softmax(gl, dim=0)
        # weight_in = torch.sum(weight_l, dim=0) / 4
        # weight_in = weight_in.detach()
        loss_in = (weight_in[0])*loss_in1+(weight_in[1])*loss_in2#权重计算EN，之前是0.1和0.15
        #
        generate_img_grad=self.sobelconv(generate_img)

        x_grad_joint= torch.max(y_grad,ir_grad)
        loss_grad=self.L2(x_grad_joint,generate_img_grad)
        # loss_grad2 = F.l1_loss(ir_grad, generate_img_grad)
        # loss_grad = weight[0]*loss_grad1+weight[1]*loss_grad2

        # # s_max = torch.max(y_grad,ir_grad)
        # # loss_s = pytorch_msssim.msssim(s_max, generate_img_grad)

        loss_total=loss_in+loss_grad#原本系数是5


        # ssim_vis = SSIM(generate_img,image_y)
        # ssim_ir = SSIM(generate_img,image_ir)
        # ssim = torch.max(ssim_ir, ssim_vis)
        return loss_total,loss_in,loss_grad, weight_in[0]
        # loss_in = F.l1_loss(image_ir, generate_img)
        # ir_grad = self.sobelconv(image_ir)
        # generate_img_grad=self.sobelconv(generate_img)
        # loss_grad = F.l1_loss(ir_grad, generate_img_grad)
        # loss_total = loss_in + loss_grad
        # return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)#增加维度
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()#将参数转化为可学习的，false不可训练

        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class Laplace(nn.Module):
    def __init__(self):
        super(Laplace, self).__init__()
        # laplace_filter1 = np.array([[0, -1, 0],
        #                             [-1, 4, -1],
        #                             [0, -1, 0]])
        laplace_filter2 = [[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]]
        kernely = torch.FloatTensor(laplace_filter2).unsqueeze(0).unsqueeze(0)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        # self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
        #                        dilation=dilation, groups=channels, bias=False)
        # self.convx.weight.data.copy_(torch.from_numpy(laplace_filter1))
        # self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
        #                        dilation=dilation, groups=channels, bias=False)
        # self.convy.weight.data.copy_(torch.from_numpy(laplace_filter2))

    def forward(self, x):
        # laplacex = self.convx(x)
        laplacey = F.conv2d(x, self.weighty, padding=1)
        # x = torch.abs(laplacex) + torch.abs(laplacey)
        return laplacey

if __name__ == '__main__':
    pass

