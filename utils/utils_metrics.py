# -*- encoding: utf-8 -*-
'''
Filename         :utils_metrics.py
Description      :指标计算方法
Time             :2023/11/10 13:27:52
Author           :Brianmmmmm
Version          :1.0
'''

from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .utils import show_config

def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5, isthrehold = False):
    #------------------------------------------------#
    #	变量定义
    #------------------------------------------------#
    beta                = 1
    smooth              = 1e-5
    threhold            = 0.5
    n, c, h, w          = inputs.size()
    nt, ht, wt, ct      = target.size()

    #------------------------------------------------#
    #	预测值图片和目标图片的长宽对不上，调整预测值图片
    #------------------------------------------------#
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    #------------------------------------------------#
    #	softmax里面写的-1表示只计算最后一个维度
    #------------------------------------------------#
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    #------------------------------------------------#
    #	计算dice系数
    #------------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float() if isthrehold else temp_inputs
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):
    print('Num classes', num_classes)
    #------------------------------------------------#
    #	创建一个全是0的矩阵，是一个混淆矩阵
    #------------------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    #------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    #------------------------------------------------#
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]

    #------------------------------------------------#
    #	读取每一个（图片-标签）对
    #------------------------------------------------#
    for index in range(len(gt_imgs)):
        #------------------------------------------------#
        #	分别读取切割结果和标签并转化成numpy数组
        #------------------------------------------------#
        pred    = np.array(Image.open(pred_imgs[index]))
        label   = np.array(Image.open(gt_imgs[index]))

        #------------------------------------------------#
        #	如果图像分割结果与标签的大小不一样，这张图片就不计算
        #------------------------------------------------#
        if len(label.flatten()) != len(pred.flatten()):  
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                  len(label.flatten()), len(pred.flatten()), gt_imgs[index], pred_imgs[index])
            )
            continue

        #------------------------------------------------#
        #	对一张图片计算21×21的hist矩阵，并累加
        #   10张输出一下目前以计算的图片中所有类别平局mIOU值 
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if name_classes is not None and index > 0 and index % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    index,
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )
        
    #------------------------------------------------#
    #	计算所有验证集图片的逐类别mIoU值
    #------------------------------------------------#
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    #------------------------------------------------#
    #	逐类别输出一下mIoU值
    #------------------------------------------------#
    if name_classes is not None:
        for ind_class in range(num_classes):
            curRecall       = round(PA_Recall[ind_class] * 100, 2)
            curPrecision    = round(Precision[ind_class] * 100, 2)
            show_dict       = {
                'class'     : name_classes[ind_class],
                'Iou'       : str(round(IoUs[ind_class] * 100, 2)),
                'Recall'    : str(curRecall),
                'Precision' : str(curPrecision),
                'Dice'      : str(round((2 * curPrecision * curRecall) / (curPrecision + curRecall), 2))
            }
            show_config(**show_dict)
            # print('===>' + name_classes[ind_class] + ':[Iou-' + str(round(IoUs[ind_class] * 100, 2)) \
            #     + '; Recall (equal to the PA)-' + str(curRecall)+ '; Precision-' + str(curPrecision) \
            #     + '; Dice-' + str(round((2 * curPrecision * curRecall) / (curPrecision + curRecall), 2)))
    
    #-----------------------------------------------------------------#
    #	在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))  
    return np.array(hist, int), IoUs, PA_Recall, Precision


