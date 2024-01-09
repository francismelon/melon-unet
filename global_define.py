# -*- encoding: utf-8 -*-
'''
Filename         :global_define.py
Description      :全局配置
Time             :2023/11/16 10:47:10
Author           :Brianmmmmm
Version          :1.0
'''

#-----------------------------------------------------------------------------------------#
#	全局变量定义
#   TYPE_VGG        vgg16
#   TYPE_RESNET     resnet50 
#   TYPE_BASE       base 
#-----------------------------------------------------------------------------------------#
BackboneType                = ('base', 'vgg', 'resnet50')
TYPE_BASE                   = BackboneType[0]
TYPE_VGG                    = BackboneType[1]
TYPE_RESNET                 = BackboneType[2]

#-----------------------------------------------------------------------------------------#
#	配置定义
#-----------------------------------------------------------------------------------------#
DEFINE_NAME_CLASSES         = ["_background_","cat","dog"]
DEFINE_UNET_BACKBONE        = TYPE_BASE
DEFINE_INPUT_SHAPE          = [512, 512]
DEFINE_NUM_CLASSES          = 1 + 1