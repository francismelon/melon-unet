# -*- encoding: utf-8 -*-
'''
Filename         :dataloader.py
Description      :数据集加载器
Time             :2023/11/10 10:23:09
Author           :Brianmmmmm
Version          :1.0
'''

import os 
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input, resize_image

class UnetDataset(Dataset):
    def __init__(self, annotation_lines, shape, num_classes, isTrainDataset, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines           = annotation_lines
        self.shape                      = shape
        self.num_classes                = num_classes
        self.isTrainDataset             = isTrainDataset
        self.dataset_path               = dataset_path
        self.length                     = len(annotation_lines)
        self.parent_path_jpg            = "VOC2007/JPEGImages"
        self.parent_path_png            = "VOC2007/SegmentationClass"

    #------------------------------------------------#
    #	需要重写两个方法
    #------------------------------------------------#    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        cur_line    = self.annotation_lines[index]
        cur_name    = cur_line.split()[0] 
        jpg         = Image.open(os.path.join(self.dataset_path, self.parent_path_jpg, cur_name + ".jpg"))
        png         = Image.open(os.path.join(self.dataset_path, self.parent_path_png, cur_name + ".png"))

        #------------------------------------------------#
        #	数据增强
        #------------------------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.shape, isRandom=self.isTrainDataset)

        #-----------------------------------------------------------------#
        #	(h, w, c) -> (c, h, w), opencv的图片保存是(h, w, c)的格式
        #-----------------------------------------------------------------#
        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png         = np.array(png)
        #-----------------------------------------------------------------#
        #	防止数组中出现无效的分类标签， 大于num_classes的全部变成最大标签
        #-----------------------------------------------------------------#
        png[png >= self.num_classes] = self.num_classes
        #------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略
        #------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.shape[0]), int(self.shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    #------------------------------------------------#
    #	下面是类自己用到的方法
    #------------------------------------------------#
    def rand(self, a=0, b=1):
        """
        Description
        -----------
        - 获取一个随机数
        
        Arguments
        ---------
        - a: 自定义
        - b: 自定义
        
        Returns
        -------
        - 浮点数
        """        
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, shape, isRandom = True):
        """
        Description
        -----------
        - 处理原图和标注图片, 数据增强
        
        Arguments
        ---------
        - image: 原图
        - label: 标注图
        - shape: 网络图片大小
        - isRandom: False不做增强处理, 调整大小和背景后原图返回图片
        
        Returns
        -------
        - image_data: 处理后的原图
        - label_data: 处理后的标注图
        """
        image           = cvtColor(image)
        label           = Image.fromarray(np.array(label))
        iw, ih          = image.size
        h, w            = shape
        jitter          = 0.3
        hue             = 0.1
        sat             = 0.7
        val             = 0.3

        if not isRandom:
            #------------------------------------------------#
            #	分别处理原图和标注图
            #------------------------------------------------#
            new_image, nw, nh   = resize_image(image, shape)

            label               = label.resize((nw, nh), Image.NEAREST)
            new_label           = Image.new('L', shape, (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------------#
        #	对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------------#
        new_ar          = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale           = self.rand(0.25, 2)
        if new_ar < 1:
            nh          = int(scale*h)
            nw          = int(nh*new_ar)
        else:
            nw          = int(scale*w)
            nh          = int(nw/new_ar)
        image           = image.resize((nw,nh), Image.BICUBIC)
        label           = label.resize((nw,nh), Image.NEAREST)

        #------------------------------------------------#
        #	翻转图像
        #------------------------------------------------#
        flip            = self.rand()<.5
        if flip: 
            image       = image.transpose(Image.FLIP_LEFT_RIGHT)
            label       = label.transpose(Image.FLIP_LEFT_RIGHT)

        #------------------------------------------------#
        #	将图像多余的部分加上灰条
        #------------------------------------------------#
        dx              = int(self.rand(0, w-nw))
        dy              = int(self.rand(0, h-nh))
        new_image       = Image.new('RGB', (w,h), (128,128,128))
        new_label       = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image           = new_image
        label           = new_label

        image_data      = np.array(image, np.uint8)
        #------------------------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #------------------------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #------------------------------------------------#
        #	将图像转到HSV上
        #------------------------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #------------------------------------------------#
        #	应用变换
        #------------------------------------------------#
        x               = np.arange(0, 256, dtype=r.dtype)
        lut_hue         = ((x * r[0]) % 180).astype(dtype)
        lut_sat         = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val         = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data      = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data      = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label

# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    #------------------------------------------------#
    #	将每个batch合成一个张量
    #------------------------------------------------#
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels