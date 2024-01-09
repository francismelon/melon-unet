# -*- encoding: utf-8 -*-
'''
Filename         :vgg.py
Description      :backbone, 主干特征提取网络
Time             :2023/11/03 14:51:40
Author           :Brianmmmmm
Version          :1.0
'''

import torch.nn as nn
from torch.hub import load_state_dict_from_url

class VGG(nn.Module):
    def __init__(self, features, num_classes = 1000):
        super(VGG, self).__init__()
        #-----------------------------------------------------#
        #	看网络组成，前面的层自定义，后面池化层和分类器全连接层
        #-----------------------------------------------------#
        self.features           = features
        self.avgpool            = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier         = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self._init_weights()

    def forward(self, x):
        #-----------------------------------------------------------------#
        #   需要注意的是结构不止16层那么少，因为没算relu和池化层，总共	
        #   第一层: 卷积 + relu + 卷积 + relu
        #   第二层: 池化 + 卷积 + relu + 卷积 + relu
        #   第三层: 池化 + 卷积 + relu + 卷积 + relu + 卷积 + relu
        #   第四层: 池化 + 卷积 + relu + 卷积 + relu + 卷积 + relu
        #   第五层: 池化 + 卷积 + relu + 卷积 + relu + 卷积 + relu + 池化
        #   并且后面unet结构会用到，需要以列表的形式把张量返回出去
        #-----------------------------------------------------------------#
        #-----------------------------------------------------------------#
        #   每层输出大小解释，以输入shape为[512, 512]为例
        #   第一层      512,512,3       -> 512,512,64
        #	第二层      512,512,64      -> 256,256,128 
        #   第三层      256,256,128     -> 128,128,256 
        #   第四层      128,128,256     ->  64, 64,512 
        #   第五层       64, 64,512     ->  32, 32,512 
        #-----------------------------------------------------------------#
        first   = self.features[  : 4](x)
        second  = self.features[ 4: 9](first)
        third   = self.features[ 9:16](second)
        fourth  = self.features[16:23](third)
        fifth   = self.features[23:-1](fourth)
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
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

def VGG16(pretrained, in_channels = 3, **kwargs):
    layers  = make_layers(cfgs['D'], batch_norm=False, in_channels=in_channels)
    model   = VGG(layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)

    del model.avgpool
    del model.classifier
    return model