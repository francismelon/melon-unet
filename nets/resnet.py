# -*- encoding: utf-8 -*-
'''
Filename         :resnet.py
Description      :backbone, 主干特征提取网络
Time             :2023/11/08 11:53:04
Author           :Brianmmmmm
Version          :1.0
'''

import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

#-----------------------------------------------------------------------------------------#
#	全局变量定义
#   RESNET_BLOCK_TYPE_BASICBLOCK        BasicBlock
#   RESNET_BLOCK_TYPE_BOTTLENECK        Bottleneck 
#-----------------------------------------------------------------------------------------#
ResNetBasicBlockType                = (0, 1)
RESNET_BLOCK_TYPE_BASICBLOCK        = ResNetBasicBlockType[0]
RESNET_BLOCK_TYPE_BOTTLENECK        = ResNetBasicBlockType[1]

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:  
    """
    Description
    -----------
    - 返回一个对应尺寸的3x3卷积大小的卷积层
    
    Arguments
    ---------
    - in_planes:    输入通道大小
    - out_planes:   输出通道大小
    - stride:       步长, 默认为1
    - groups:       分组, 默认为1
    - dilation:     填充和卷积核元素间距, 默认为1
    
    Returns
    -------
    - nn.Conv2d:    3x3核大小的卷积层
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    Description
    -----------
    - 返回一个对应尺寸的1x1卷积大小的卷积层
    
    Arguments
    ---------
    - in_planes:    输入通道大小
    - out_planes:   输出通道大小
    - stride:       步长, 默认为1
    
    Returns
    -------
    - nn.Conv2d:    1x1核大小的卷积层
    """
        
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    #-----------------------------------------------------------------------------------------#
    #	resnet里面的子模块结构，18和34用的是BasicBlock，50用的是Bottleneck
    #-----------------------------------------------------------------------------------------#
    expansion = 4
    def __init__(self, in_size, out_size, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        #------------------------------------------------#
        #	成员变量自定义
        #------------------------------------------------#
        if norm_layer is None:
            norm_layer  = nn.BatchNorm2d

        in_width          = int(out_size * (base_width / 64.)) * groups
        out_width         = out_size * self.expansion        
        
        #-----------------------------------------------------------------#
        #	每个Bottleneck包含
        #   1x1卷积层       用来下降通道数
        #   归一层          防止过拟合
        #   3x3卷积层       提取特征
        #   归一层          防止过拟合
        #   1x1卷积层       用来上升通道数
        #   归一层          防止过拟合
        #   激活函数        输出时用到
        #   下采样层        某些时候需要
        #-----------------------------------------------------------------#
        self.conv1      = conv1x1(in_size, in_width)
        self.bn1        = norm_layer(in_width)
        self.conv2      = conv3x3(in_width, in_width, stride, groups, dilation)
        self.bn2        = norm_layer(in_width)
        self.conv3      = conv1x1(in_width, out_width)
        self.bn3        = norm_layer(out_width)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        #------------------------------------------------#
        #	记录原始输入，后面拿来做矩阵加法
        #------------------------------------------------#
        origin_x        = x

        #------------------------------------------------#
        #	卷积完成之后，进行叠加
        #------------------------------------------------#
        out             = self.conv1(x)
        out             = self.bn1(out)
        out             = self.relu(out)
        out             = self.conv2(out)
        out             = self.bn2(out)
        out             = self.relu(out)
        out             = self.conv3(out)
        out             = self.bn3(out)
        out             += self.downsample(x) if self.downsample is not None else origin_x
        out             = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        #------------------------------------------------#
        #	成员变量自定义
        #------------------------------------------------#
        self.in_size    = 64

        #-----------------------------------------------------------------#
        #	和vgg类似，resnet可以分成五层然后进行堆叠
        #   具体可以看论文和知乎上面的resnet网络结构介绍，这里以resnet50为例
        #   计算公式：  [(W - F + 2*P) / S] + 1
        #   第一层：    卷积 + 归一 + 激活函数
        #   第2~5层：   都是Bottleneck的组合，具体看需要几个Bottleneck
        #-----------------------------------------------------------------#
        #-----------------------------------------------------------------#
        #	以输入shape为[600, 600]的为例
        #   第一层      600,600,3   -> 300,300,64
        #   池化层      300,300,64  -> 150,150,64
        #-----------------------------------------------------------------#
        self.conv1      = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1        = nn.BatchNorm2d(64)
        self.relu       = nn.ReLU(inplace=True)
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        #-----------------------------------------------------------------#
        #   expansion   = 4, 除了第二层, 其他层步长为2
        #	第二层      150,150,64      -> 150,150,256      (64  * 4 / 1)
        #   第三层      150,150,256     -> 75 ,75 ,512      (256 * 4 / 2)
        #   第四层      75 ,75 ,512     -> 38 ,38 ,1024     (512 * 4 / 2)
        #   第五层      38 ,38 ,1024    -> 19 ,19 ,2048     (1024* 4 / 2)
        #   池化层和全连接层，因为这个resnet是unet的子模块，所以后面会删掉
        #-----------------------------------------------------------------#
        self.layer2     = self._make_layer(block, 64,  layers[0])
        self.layer3     = self._make_layer(block, 128, layers[1], stride=2)
        self.layer4     = self._make_layer(block, 256, layers[2], stride=2)
        self.layer5     = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool    = nn.AvgPool2d(7)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
        #-----------------------------------------------------------------#
        #	初始层的权重
        #-----------------------------------------------------------------#
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #------------------------------------------------#
                #	H * W * C
                #------------------------------------------------#
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channel, blocks, stride=1, block_type=RESNET_BLOCK_TYPE_BOTTLENECK):
        dowmsample      = None
        #------------------------------------------------#
        #	下面这个判断是专门给BOTTLENECK用的
        #------------------------------------------------#
        # if stride != 1 or self.in_size != channel * block.expansion:
        if block_type == RESNET_BLOCK_TYPE_BOTTLENECK:
            dowmsample  = nn.Sequential(
                nn.Conv2d(self.in_size, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        #------------------------------------------------#
        #	子模块层内容定义
        #------------------------------------------------#
        layers          = []
        layers.append(block(self.in_size, channel, stride, dowmsample))
        self.in_size    = channel * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_size, channel))

        return nn.Sequential(*layers) 

    def forward(self, x):
        #-----------------------------------------------------------------#
        #	下面是没有unet用来堆叠才这样写，也就是不删除最后的池化层和分类层
        #-----------------------------------------------------------------#
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        #-----------------------------------------------------------------#
        #	下面是需要进行unet的连接
        #-----------------------------------------------------------------#
        x           = self.conv1(x)
        x           = self.bn1(x)
        first       = self.relu(x)

        x           = self.maxpool(first)
        second      = self.layer2(x)
        third       = self.layer3(second)
        fourth      = self.layer4(third)
        fifth       = self.layer5(fourth)
        return [first, second, third, fourth, fifth]
        
def ResNet50(pretrained=False, **kwargs):
    #------------------------------------------------#
    #	每种resnet层数、子模块不同
    #   resnet18    [2, 2, 2, 2],   BasicBlock
    #   resnet34    [3, 4, 6, 3],   BasicBlock
    #   resnet50    [3, 4, 6, 3],   Bottleneck
    #   resnet101   [3, 4, 23, 3],  Bottleneck
    #   resnet152   [3, 8, 36, 3],  Bottleneck
    #------------------------------------------------#
    model           = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
                                                 model_dir='model_data'), strict=False)
    
    #------------------------------------------------#
    #	跟unet连接，池化层和分类层不需要
    #------------------------------------------------#
    del model.avgpool
    del model.classifier
    return model

