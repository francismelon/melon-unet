# -*- encoding: utf-8 -*-
'''
Filename         :utils_loss.py
Description      :定义loss类
Time             :2023/11/09 13:18:50
Author           :Brianmmmmm
Version          :1.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_metrics import f_score



class LossFunction:
    def __init__(self):
        pass
    """
    Description
    -----------
    - 提供三种计算损失的方法, 分别是交叉熵、交叉熵改进版(正负均衡)、Dice计算
    """
    def CE_Loss(self, predicts, target, cls_weights, num_classes=21):
        n, c, h, w      = predicts.size()
        nt, ht, wt      = target.size()
        #------------------------------------------------#
        #	预测值图片和目标图片的长宽对不上，调整预测值图片
        #------------------------------------------------#
        if h != ht and w != wt:
            predicts    = F.interpolate(predicts, size=(ht, wt), mode='bilinear', align_corners=True)

        #-----------------------------------------------------------------#
        #	连续transpose将(n, c, h, w) -> (n, h, w, c)
        #   contiguous将张量转变为连续的内存布局
        #   view中-1为自动计算，在不创建新张量的情况下输出(n*h*w, c)大小的张量 
        #-----------------------------------------------------------------#
        tmp_predict     = predicts.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        tmp_target      = target.view(-1)

        ce_loss = nn.CrossEntropyLoss(weight=cls_weights, 
                                      ignore_index=num_classes)(tmp_predict, tmp_target)
        return ce_loss

    def Focal_Loss(self, predicts, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
        n, c, h, w      = predicts.size()
        nt, ht, wt      = target.size()
        #------------------------------------------------#
        #	预测值图片和目标图片的长宽对不上，调整预测值图片
        #------------------------------------------------#
        if h != ht and w != wt:
            predicts    = F.interpolate(predicts, size=(ht, wt), mode='bilinear', align_corners=True)

        #-----------------------------------------------------------------#
        #	连续transpose将(n, c, h, w) -> (n, h, w, c)
        #   contiguous将张量转变为连续的内存布局
        #   view中-1为自动计算，在不创建新张量的情况下输出(n*h*w, c)大小的张量 
        #-----------------------------------------------------------------#
        tmp_predict     = predicts.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        tmp_target      = target.view(-1)

        logpt           = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes,
                                               reduction='none')(tmp_predict, tmp_target)
        pt              = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss            = -((1 - pt) ** gamma) * logpt
        focal_loss      = loss.mean()
        return focal_loss

    def Dice_loss(self, predicts, target):
        #------------------------------------------------#
        #	dice_loss 计算就是 1 - f_score
        #------------------------------------------------#
        dice_loss = 1 - f_score(predicts, target)
        return dice_loss