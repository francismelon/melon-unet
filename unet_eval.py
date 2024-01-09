# -*- encoding: utf-8 -*-
'''
Filename         :unet_eval.py
Description      :模型评估
Time             :2023/11/11 10:02:49
Author           :Brianmmmmm
Version          :1.0
'''

import colorsys
import copy
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from nets.unet import Unet
from utils.utils import cvtColor, show_config, get_inference_pr, image_to_net_numpy
from global_define import *


def get_mix_image(colors, mix_type, pr, orininal_h, orininal_w, old_img, lbl_image=None):
    if mix_type == 0:
        #------------------------------------------------#
        #	注释的代码效果一样, 不过要指定像素种类
        #------------------------------------------------#
        # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        # for c in range(self.num_classes):
        #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
        #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
        #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        #------------------------------------------------#
        #   将新图片转换成Image的形式
        #------------------------------------------------#
        image   = Image.fromarray(np.uint8(seg_img))
        #------------------------------------------------#
        #   将新图与原图及进行混合
        #------------------------------------------------#
        image   = Image.blend(old_img, image, 0.7)
        #------------------------------------------------#
        #	显示预测和实际的重合部分，lbl_image是实际图片
        #------------------------------------------------#
        if lbl_image is not None:
            # arr = np.zeros(seg_img.shape[:2], dtype=np.uint8)
            lbl_image   = cvtColor(lbl_image)
            arr         = np.array(lbl_image, dtype=np.uint8)
            for curH in range(arr.shape[0]):
                for curW in range(arr.shape[1]):
                    if (arr[curH][curW] == colors[1])[0] == True:
                    # if (arr[curH][curW] == colors[1]) == True:
                        arr[curH][curW] = (0, 128, 0)   # 设置为绿色
            lbl_image = Image.fromarray(arr)
            image       = Image.blend(lbl_image, image, 0.7)

    elif mix_type == 1:
        # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        # for c in range(self.num_classes):
        #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
        #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
        #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        #------------------------------------------------#
        #   将新图片转换成Image的形式
        #------------------------------------------------#
        image   = Image.fromarray(np.uint8(seg_img))

    elif mix_type == 2:
        seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
        #------------------------------------------------#
        #   将新图片转换成Image的形式
        #------------------------------------------------#
        image = Image.fromarray(np.uint8(seg_img))
    
    return image

def count_image_pixel(pr, num_classes, original_h, original_w, name_classes):
    classes_nums        = np.zeros([num_classes])
    total_points_num    = original_h * original_w
    print('-' * 63)
    print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
    print('-' * 63)
    for i in range(num_classes):
        num     = np.sum(pr == i)
        ratio   = num / total_points_num * 100
        if num > 0:
            print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
            print('-' * 63)
        classes_nums[i] = num
    print("classes_nums:", classes_nums)

def get_colors_tuples(num_classes):
    if num_classes <= 21:
        colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                   (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                   (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                   (128, 64, 12)]
    else:
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    
    return colors

class UnetEval(object):
    #------------------------------------------------#
    #	继承object可以使用__dict__进行参数配置
    #------------------------------------------------#
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        #-------------------------------------------------------------------#
        # "model_path"    : 'model_data/unet_vgg_voc.pth',
        "model_path"    : 'logs/best_epoch_weights.pth',
        #--------------------------------#
        #   所需要区分的类的个数+1
        #--------------------------------#
        "num_classes"   : DEFINE_NUM_CLASSES,
        #--------------------------------#
        #   所使用的的主干网络：vgg、resnet50、base  
        #--------------------------------#
        "backbone"      : DEFINE_UNET_BACKBONE,
        #--------------------------------#
        #   输入图片的大小
        #--------------------------------#
        "input_shape"   : DEFINE_INPUT_SHAPE,
        #-------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        #-------------------------------------------------#
        "mix_type"      : 0,
        #--------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #--------------------------------#
        "cuda"          : True,
    }

    def __init__(self, **kwargs):
        #-----------------------------------------------------------------#
        #	将字典里的参数注册到类里面，这样写方便，但是没有代码提示
        #-----------------------------------------------------------------#
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        #------------------------------------------------#
        #	画框设置不同的颜色
        #------------------------------------------------#
        self.colors = get_colors_tuples(self.num_classes)
        #------------------------------------------------#
        #	生成模型，打印字典参数
        #------------------------------------------------#
        self.generate()
        show_config(**self._defaults)

    def generate(self, onnx=False):
        #------------------------------------------------#
        #	创造模型，并且进行基础设置
        #------------------------------------------------#
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net        = Unet(num_classes=self.num_classes, backbone=self.backbone)
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net        = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx and self.cuda:
            self.net    = nn.DataParallel(self.net)
            self.net    = self.net.cuda()

    def detect_image(self, image, count=False, name_classes=None, lbl_image=None):
        #------------------------------------------------#
        #	转成RGB图片
        #------------------------------------------------#
        image               = cvtColor(image)
        #------------------------------------------------#
        #	对输入图像进行一个备份，后面用于绘图
        #------------------------------------------------#
        old_image           = copy.deepcopy(image)
        original_h          = np.array(image).shape[0]
        original_w          = np.array(image).shape[1]
        #------------------------------------------------#
        #	转成numpy数组，当前数组为四维
        #------------------------------------------------#
        image_data, nw, nh  = image_to_net_numpy(image, self.input_shape)
        #------------------------------------------------#
        #	进行推理，获取推理结果
        #------------------------------------------------#
        pr                  = get_inference_pr(self.net, image_data, self.cuda, self.input_shape,
                                               nw, nh, original_w, original_h)

        #-----------------------------------------------------------------#
        #	目标像素计数
        #-----------------------------------------------------------------#
        if count:
            count_image_pixel(pr, self.num_classes, original_h, original_w, name_classes)

        result_image = get_mix_image(self.colors, self.mix_type, pr, original_h, original_w, old_image, lbl_image)
        return result_image

    def get_FPS(self, image, test_interval):
        #------------------------------------------------#
        #	转成RGB图片
        #------------------------------------------------#
        image               = cvtColor(image)
        #------------------------------------------------#
        #	转成numpy数组，当前数组为四维
        #------------------------------------------------#
        image_data, nw, nh  = image_to_net_numpy(image, self.input_shape)

        #-----------------------------------------------------------------#
        #	会发现这里写了两次，因为第一次会很慢，后面那次才是真正的算时间
        #	进行推理，获取推理结果
        #------------------------------------------------#
        pr                  = get_inference_pr(self.net, image_data, self.cuda, self.input_shape,
                                               nw, nh, is_resize=False)

        t1 = time.time()
        for _ in range(test_interval):
            get_inference_pr(self.net, image_data, self.cuda, self.input_shape,
                             nw, nh, is_resize=False)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_miou_png(self, image):
        #------------------------------------------------#
        #	转成RGB图片
        #------------------------------------------------#
        image               = cvtColor(image)
        #------------------------------------------------#
        #	获取原图长宽，后面恢复用到
        #------------------------------------------------#
        original_h          = np.array(image).shape[0]
        original_w          = np.array(image).shape[1]
        #------------------------------------------------#
        #	转成numpy数组，当前数组为四维
        #------------------------------------------------#
        image_data, nw, nh  = image_to_net_numpy(image, self.input_shape)

        #------------------------------------------------#
        #	进行推理，获取推理结果
        #------------------------------------------------#
        pr                  = get_inference_pr(self.net, image_data, self.cuda, self.input_shape,
                                               nw, nh, original_w, original_h)
    
        image = Image.fromarray(np.uint8(pr))
        return image

    def convert_to_onnx(self, simplify, export_model_path):
        import onnx
        self.generate(onnx=True)

        #-----------------------------------------------------------------#
        #	input_layer_names和output_layer_names会在写C++动态库中用到
        #-----------------------------------------------------------------#
        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]

        #------------------------------------------------#
        #	开始导出onnx模型
        #------------------------------------------------#
        print(f'Starting export with onnx {onnx.__version__}.')
        try:
            torch.onnx.export(self.net,
                            im,
                            f               = export_model_path,
                            verbose         = False,
                            opset_version   = 12,
                            training        = torch.onnx.TrainingMode.EVAL,
                            do_constant_folding = True,
                            input_names     = input_layer_names,
                            output_names    = output_layer_names,
                            dynamic_axes    = None)
        except:
            raise Exception("Export onnx model falid!")

        #------------------------------------------------#
        #	导出模型检查
        #------------------------------------------------#
        model_onnx          = onnx.load(export_model_path)
        onnx.checker.check_model(model_onnx)

        #------------------------------------------------#
        #	优化模型推理速度
        #------------------------------------------------#
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, export_model_path)

        print('Onnx model save as {}'.format(export_model_path))

class UnetONNXEval(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   onnx_path指向model_data文件夹下的onnx权值文件
        #-------------------------------------------------------------------#
        "onnx_path"    : 'model_data/models.onnx',
        #--------------------------------#
        #   所需要区分的类的个数+1
        #--------------------------------#
        "num_classes"   : DEFINE_NUM_CLASSES,
        #--------------------------------#
        #   所使用的的主干网络：vgg、resnet50   
        #--------------------------------#
        "backbone"      : DEFINE_UNET_BACKBONE,
        #--------------------------------#
        #   输入图片的大小
        #--------------------------------#
        "input_shape"   : DEFINE_INPUT_SHAPE,
        #-------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        #-------------------------------------------------#
        "mix_type"      : 0,
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #-----------------------------------------------------------------#
    #	初始化ONNX
    #-----------------------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        import onnxruntime
        self.onnx_session   = onnxruntime.InferenceSession(self.onnx_path)
        #------------------------------------------------#
        #	获取输出输出节点名字
        #------------------------------------------------#
        self.input_name     = self.get_input_name()
        self.output_name    = self.get_output_name()
        #------------------------------------------------#
        #	画框设置不同的颜色
        #------------------------------------------------#
        self.colors = get_colors_tuples(self.num_classes)
        show_config(**self._defaults)

    def get_input_name(self):
        #------------------------------------------------#
        #	获得所有的输入node
        #------------------------------------------------#
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_output_name(self):
        #------------------------------------------------#
        #	获得所有的输出node
        #------------------------------------------------#
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self, image_tensor):
        #------------------------------------------------#
        #	利用input_name获得输入的tensor
        #------------------------------------------------#
        input_feed={}
        for name in self.input_name:
            input_feed[name] = image_tensor
        return input_feed

    def detect_image(self, image, count=False, name_classes=None):
        #------------------------------------------------#
        #	转成RGB图片
        #------------------------------------------------#
        image               = cvtColor(image)
        #------------------------------------------------#
        #	对输入图像进行一个备份，后面用于绘图
        #------------------------------------------------#
        old_image           = copy.deepcopy(image)
        original_h          = np.array(image).shape[0]
        original_w          = np.array(image).shape[1]
        #------------------------------------------------#
        #	转成numpy数组，当前数组为四维
        #------------------------------------------------#
        image_data, nw, nh  = image_to_net_numpy(image, self.input_shape)
        #------------------------------------------------#
        #	进行推理，获取推理结果
        #------------------------------------------------#
        input_feed          = self.get_input_feed(image_data)
        pr                  = self.onnx_session.run(output_names=self.output_name, input_feed=input_feed)[0][0]

        def softmax(x, axis):
            x -= np.max(x, axis=axis, keepdims=True)
            f_x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
            return f_x
        print(np.shape(pr))
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = softmax(np.transpose(pr, (1, 2, 0)), -1)
        #--------------------------------------#
        #   将灰条部分截取掉
        #--------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        #   进行图片的resize
        #---------------------------------------------------#
        pr = cv2.resize(pr, (original_w, original_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)

        #-----------------------------------------------------------------#
        #	目标像素计数
        #-----------------------------------------------------------------#
        if count:
            count_image_pixel(pr, self.num_classes, original_h, original_w, name_classes)

        result_image = get_mix_image(self.colors, self.mix_type, pr, original_h, original_w, old_image)
        return result_image