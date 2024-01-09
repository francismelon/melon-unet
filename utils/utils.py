# -*- encoding: utf-8 -*-
'''
Filename         :utils.py
Description      :工具
Time             :2023/11/09 16:57:29
Author           :Brianmmmmm
Version          :1.0
'''

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def image_to_net_numpy(rgb_image, net_shape):
    """
    Description
    -----------
    将rgb图片转成numpy数据, 并返回新的长度和高度
    
    Arguments
    ---------
    - rgb_image: rbg图片
    - net_shape: 推理网络的图片长宽
    
    Returns
    -------
    - numpy数组
    - 新宽度
    - 新高度
    """
    
    #------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    #------------------------------------------------#
    image_data, nw, nh  = resize_image(rgb_image, (net_shape[1], net_shape[0]))
    #------------------------------------------------#
    #	增加一个维度，并调整(H, W, C)->(C, H, W)
    #------------------------------------------------#
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    
    return image_data, nw, nh

def get_inference_pr(net, image_data, is_cuda, net_shape, nw, nh, original_w = None, original_h = None, is_resize = True):
    """
    Description
    -----------
    如果需要复原图片, 会获取推理结果
    
    Arguments
    ---------
    - net: 网络
    - image_data: 需要推理的numpy数组
    - is_cuda: 是否使用cuda
    - net_shape: 推理网络的图片长宽
    - nw: 新宽度
    - nh: 新高度
    - original_w: 复原宽度
    - original_h: 复原高度
    - is_resize: 是否进行复原
    
    Returns
    -------
    - 复原的numpy数组
    """
    
    with torch.no_grad():
        images      = torch.from_numpy(image_data)
        if is_cuda:
            images  = images.cuda()

        #------------------------------------------------#
        #	因为只有一张图片，推理后取出第一张
        #------------------------------------------------#
        pr          = net(images)[0]
        #------------------------------------------------#
        #	取出每一个像素点的种类,并调整(C, H, W)->(H, W, C)
        #------------------------------------------------#
        pr          = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
        #------------------------------------------------#
        #	截取灰条
        #------------------------------------------------#
        pr          = pr[int((net_shape[0] - nh) // 2) : int((net_shape[0] - nh) // 2 + nh), \
                         int((net_shape[1] - nw) // 2) : int((net_shape[1] - nw) // 2 + nw)]
        
        if is_resize:         
            #------------------------------------------------#
            #	图片的resize
            #------------------------------------------------#
            pr          = cv2.resize(pr, (original_w, original_h), interpolation = cv2.INTER_LINEAR)
            #------------------------------------------------#
            #	通过像素点种类，计算模型输出的概率分布中最大值的索引
            #------------------------------------------------#
            pr = pr.argmax(axis=-1)
            
            return pr

def cvtColor(image):
    """
    Description
    -----------
    - 将图像转换成RGB图像, 防止灰度图在预测时报错
    - 代码仅仅支持RGB图像的预测, 所有其它类型的图像都会转化成RGB
    
    Arguments
    ---------
    - image: 图片
    
    Returns
    -------
    - RGB图片
    """    
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def resize_image(image, size):
    """
    Description
    -----------
    - 重新调整图像大小, 并且返回新图像、长、宽
    
    Arguments
    ---------
    - image: 旧图像
    - size: 输入网络的图片大小, 不是图片原来的大小
    
    Returns
    -------
    - 新图片
    - 长
    - 宽
    """    
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    """
    Description
    -----------
    - 将字典里的键值分别打印出来
    """    
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'vgg'       : 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50'  : 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)