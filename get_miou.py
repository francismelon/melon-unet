# -*- encoding: utf-8 -*-
'''
Filename         :get_miou.py
Description      :评估模型
Time             :2023/11/14 13:39:42
Author           :Brianmmmmm
Version          :1.0
'''

import os
from PIL import Image
from tqdm import tqdm

from unet_eval import UnetEval
from utils.utils_metrics import compute_mIoU
from utils.utils_plot import show_results
from global_define import DEFINE_NUM_CLASSES, DEFINE_NAME_CLASSES

#-----------------------------------------------------------------------------------------#
# 进行指标评估需要注意以下几点：
# 1、该文件生成的图为灰度图，因为值比较小，按照JPG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
# 2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
# 3、仅有按照VOC格式数据训练的模型可以利用这个文件进行miou的计算。
#-----------------------------------------------------------------------------------------#

if __name__ == "__main__":
    #-----------------------------------------------------------------------------------------#
    #	评估模式定义
    #-----------------------------------------------------------------------------------------#
    MiouMode                = [0, 1, 2]
    MIOU_MODE_ALL           = MiouMode[0]
    MIOU_MODE_RESULT        = MiouMode[1]
    MIOU_MODE_MIOU          = MiouMode[2]
    #-----------------------------------------------------------------------------------------#
    #   MIOU_MODE_ALL       计算整个miou流程，包括预测结果和miou
    #   MIOU_MODE_RESULT    仅获得预测结果
    #   MIOU_MODE_MIOU      仅计算miou
    #-----------------------------------------------------------------------------------------#
    miou_mode               = MIOU_MODE_ALL
    #-----------------------------------------------------------------------------------------#
    #	num_classes         分类个数 + 1
    #-----------------------------------------------------------------------------------------#
    num_classes             = DEFINE_NUM_CLASSES
    #-----------------------------------------------------------------------------------------#
    #	区分的种类，和json_to_dataset里面的一样
    #-----------------------------------------------------------------------------------------#
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["_background_","cat","dog"]
    # name_classes    = ["_background_", "fat"]
    name_classes            = DEFINE_NAME_CLASSES
    #-----------------------------------------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-----------------------------------------------------------------------------------------#
    VOCdevkit_path          = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == MIOU_MODE_ALL or miou_mode == MIOU_MODE_RESULT:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        unet = UnetEval() 
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            childList   = str(image_id).split('/')
            target_dir  = os.path.join(pred_dir, childList[0], childList[1], childList[2])
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == MIOU_MODE_ALL or miou_mode == MIOU_MODE_MIOU:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)