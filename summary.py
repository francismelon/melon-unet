# -*- encoding: utf-8 -*-
'''
Filename         :summary.py
Description      :查看网络结构
Time             :2023/11/14 14:05:16
Author           :Brianmmmmm
Version          :1.0
'''

import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.unet import Unet
from global_define import *

if __name__ == "__main__":
    input_shape     = DEFINE_INPUT_SHAPE
    num_classes     = DEFINE_NUM_CLASSES
    backbone        = DEFINE_UNET_BACKBONE
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(num_classes = num_classes, backbone = backbone).to(device)

    print('*'*30)
    for i in model._modules.items():
        print(i)
    print('*'*30)

    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))