# -*- encoding: utf-8 -*-
'''
Filename         :callbacks.py
Description      :回调函数
Time             :2023/11/10 13:19:13
Author           :Brianmmmmm
Version          :1.0
'''

import os
import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, get_inference_pr, image_to_net_numpy
from .utils_metrics import compute_mIoU

class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir            = log_dir
        self.val_loss_flag      = val_loss_flag

        self.losses             = []
        if self.val_loss_flag:
            self.val_loss       = []
        
        os.makedirs(self.log_dir)
        self.writer             = SummaryWriter(self.log_dir)
        try:
            dummy_input         = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss = None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
            
        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
            
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
        miou_out_path=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious              = [0]
        self.epoches            = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")
    
    def get_miou_png(self, image):
        #------------------------------------------------#
        #	转成RGB图片
        #------------------------------------------------#
        image       = cvtColor(image)
        #------------------------------------------------#
        #	获取原图长宽，后面恢复用到
        #------------------------------------------------#
        orinigal_h  = np.array(image).shape[0]
        orinigal_w  = np.array(image).shape[1]
        #------------------------------------------------#
        #	转成numpy数组，当前数组为四维
        #------------------------------------------------#
        image_data, nw, nh  = image_to_net_numpy(image, self.input_shape)
        #------------------------------------------------#
        #	进行推理，获取推理结果
        #------------------------------------------------#
        pr                  = get_inference_pr(self.net, image_data, self.cuda, self.input_shape,
                                               nw, nh, orinigal_w, orinigal_h, is_resize=True)
        
        image = Image.fromarray(np.uint8(pr))
        return image

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            gt_dir      = os.path.join(self.dataset_path, "VOC2007/SegmentationClass/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)

            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   创建文件夹
                #-------------------------------#
                childList   = str(image_id).split('/')
                target_dir  = os.path.join(pred_dir, childList[0], childList[1], childList[2])
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#
                image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                image       = Image.open(image_path)
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                image       = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate miou.")
            _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")

            print("Get miou done.")
            shutil.rmtree(self.miou_out_path)