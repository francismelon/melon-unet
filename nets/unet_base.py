# -*- encoding: utf-8 -*-
'''
Filename         :unet_base.py
Description      :无修改版unet_base
Time             :2023/11/15 13:15:54
Author           :Brianmmmmm
Version          :1.0
'''

import torch.nn as nn

class BackBone(nn.Module):
    def __init__(self, features) -> None:
        super(BackBone, self).__init__()
        #------------------------------------------------#
        #	第一层不需要最大池化
        #------------------------------------------------#
        self.features           = features
        self._init_weights()

    def forward(self, x):
        #-----------------------------------------------------------------#
        #   总共34层
        #   第一层: 卷积 + BN层 + relu + 卷积 + BN层 + relu
        #   第二层: 池化 + 卷积 + BN层 + relu + 卷积 + BN层 + relu
        #   第三层: 池化 + 卷积 + BN层 + relu + 卷积 + BN层 + relu
        #   第四层: 池化 + 卷积 + BN层 + relu + 卷积 + BN层 + relu
        #   第五层: 池化 + 卷积 + BN层 + relu + 卷积 + BN层 + relu
        #   并且后面unet结构会用到，需要以列表的形式把张量返回出去
        #-----------------------------------------------------------------#
        #-----------------------------------------------------------------#
        #   每层输出大小解释, 以输入shape为[512, 512]为例
        #   第一层      512, 512, 3     -> 512, 512, 64
        #	第二层      512, 512, 64    -> 256, 256, 64     -> 256, 256, 128 
        #   第三层      256, 256, 128   -> 128, 128, 128    -> 128, 128, 256 
        #   第四层      128, 128, 256   ->  64,  64, 256    ->  64,  64, 512
        #   第五层       64,  64, 512   ->  32,  32, 512    ->  32,  32, 512
        #-----------------------------------------------------------------#
        first   = self.features[  : 6](x)
        second  = self.features[ 6:13](first)
        third   = self.features[13:20](second)
        fourth  = self.features[20:27](third)
        fifth   = self.features[27:-1](fourth)
        return [first, second, third, fourth, fifth]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm = False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]

def UnetBase(in_channels = 3, **kwarg):
    #------------------------------------------------#
    #	没有voc的预训练模型，只能从头开始训练
    #------------------------------------------------#
    layers  = make_layers(cfgs, batch_norm=True, in_channels=in_channels)
    model   = BackBone(layers, **kwarg)
    return model
