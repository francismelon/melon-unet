# -*- encoding: utf-8 -*-
'''
Filename         :unet.py
Description      :unet
Time             :2023/11/03 15:40:54
Author           :Brianmmmmm
Version          :1.0
'''

import torch
import torch.nn as nn
from .vgg import VGG16
from .resnet import ResNet50
from .unet_base import UnetBase
from global_define import TYPE_BASE, TYPE_VGG, TYPE_RESNET

class UnetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, input1, input2):
        #------------------------------------------------#
        #	cat里的第二个参数为1，代表在通道数上进行叠加
        #   (N, C, H, W) C为第二维度，即是1
        #------------------------------------------------#
        output      = torch.cat([input1, self.up(input2)], 1)
        output      = self.conv1(output)
        output      = self.relu(output)
        output      = self.conv2(output)
        output      = self.relu(output)
        return output

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = TYPE_VGG):
        super(Unet, self).__init__()
        #------------------------------------------------#
        #	共用变量初始化
        #------------------------------------------------#
        in_filters          = []
        out_filters         = [64, 128, 256, 512]
        up_conv             = None

        #------------------------------------------------#
        #	参数设置
        #------------------------------------------------#
        if backbone == TYPE_VGG:
            self.vgg        = VGG16(pretrained=pretrained)
            #-----------------------------------------------------------------#
            #   输出通道数解释如下
            #	up4         C4 + C5     = 512 + 512 = 1024  -> 512(调整后的通道数)
            #	up3         C3 + up4    = 256 + 512 = 768   -> 256(调整后的通道数)
            #	up2         C2 + up3    = 128 + 256 = 384   -> 128(调整后的通道数)
            #	up1         C1 + up2    =  64 + 128 = 192   ->  64(调整后的通道数)
            #-----------------------------------------------------------------#
            in_filters      = [192, 384, 768, 1024]
        elif backbone == TYPE_RESNET:
            self.resnet     = ResNet50(pretrained=pretrained)
            #-----------------------------------------------------------------#
            #   输出通道数解释如下，可以看resnet类定义部分
            #	up4         C4 + C5     = 1024 + 2048 = 3072    -> 512(调整后的通道数)
            #	up3         C3 + up4    =  512 +  512 = 1024    -> 256(调整后的通道数)
            #	up2         C2 + up3    =  256 +  256 =  512    -> 128(调整后的通道数)
            #	up1         C1 + up2    =   64 +  128 =  192    ->  64(调整后的通道数)
            #-----------------------------------------------------------------#
            in_filters      = [192, 512, 1024, 3072]
        elif backbone == TYPE_BASE:
            self.unetbase   = UnetBase()
            in_filters      = [64 + 128, 128 + 256, 256 + 512, 512 + 512]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))

        #------------------------------------------------#
        #	堆叠层初始化
        #------------------------------------------------#
        self.concat4        = UnetUp(in_filters[3], out_filters[3])
        self.concat3        = UnetUp(in_filters[2], out_filters[2])
        self.concat2        = UnetUp(in_filters[1], out_filters[1])
        self.concat1        = UnetUp(in_filters[0], out_filters[0])

        #------------------------------------------------#
        #	resnet需要加多一层上采样
        #------------------------------------------------#
        if backbone == TYPE_RESNET:
            up_conv         = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU()
            )

        #------------------------------------------------#
        #	分类器定义，主干网络、类型及上采样定义
        #------------------------------------------------#
        self.classifier     = nn.Conv2d(in_channels=out_filters[0], out_channels=num_classes, kernel_size=1)
        self.backbone       = backbone
        self.up_conv        = up_conv

    def forward(self, inputs):
        if self.backbone == TYPE_VGG:
            [first, second, third, fourth, fifth] = self.vgg.forward(inputs)
        elif self.backbone == TYPE_RESNET:
            [first, second, third, fourth, fifth] = self.resnet.forward(inputs)
        elif self.backbone == TYPE_BASE:
            [first, second, third, fourth, fifth] = self.unetbase.forward(inputs)
        else:
            return None

        up4                 = self.concat4(fourth,  fifth)
        up3                 = self.concat3(third,   up4)
        up2                 = self.concat2(second,  up3)
        up1                 = self.concat1(first,   up2)
        up1                 = up1 if self.up_conv is None else self.up_conv(up1)

        classifier          = self.classifier(up1)
        return classifier

    def freeze_backbone(self, isUnFreeze: bool):
        """
        Description
        -----------
        - 设置是否冻结网络, 冻结会修改requires_grad参数
        
        Arguments
        ---------
        - isUnFreeze: True为解冻, False为冻结
        """
    
        for param in self.backbone_net.parameters():
            param.requires_grad = isUnFreeze

    