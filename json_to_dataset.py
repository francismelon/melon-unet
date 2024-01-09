# -*- encoding: utf-8 -*-
'''
Filename         :json_to_dataset.py
Description      :标注数据转换成标签图片
Time             :2023/11/14 14:01:26
Author           :Brianmmmmm
Version          :1.0
'''

import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils
from global_define import DEFINE_NAME_CLASSES

#-----------------------------------------------------------------------------------------#
# 制作自己的语义分割数据集需要注意以下几点：
# 1、我使用的labelme版本是3.16.7，建议使用该版本的labelme，有些版本的labelme会发生错误，
#    具体错误为：Too many dimensions: 3 > 2
#    安装方式为命令行pip install labelme==3.16.7
# 2、此处生成的标签图是8位彩色图，与视频中看起来的数据集格式不太一样。
#    虽然看起来是彩图，但事实上只有8位，此时每个像素点的值就是这个像素点所属的种类。
#    所以其实和视频中VOC数据集的格式一样。因此这样制作出来的数据集是可以正常使用的。也是正常的。
#-----------------------------------------------------------------------------------------#

if __name__ == '__main__':
    jpgs_path   = "datasets/JPEGImages"
    pngs_path   = "datasets/SegmentationClass"
    # classes     = ["_background_","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # classes     = ["_background_","cat","dog"]
    classes     = DEFINE_NAME_CLASSES
    
    
    jsonNum = 0
    dirCount = os.listdir("./datasets/before/") 
    for i in range(0, len(dirCount)):
        dirPath = os.path.join("./datasets/before", dirCount[i])
        imageCount = os.listdir(dirPath)
        for j in range(0, len(imageCount)):
            imagePath = os.path.join(dirPath, imageCount[j])
            photoCount = os.listdir(imagePath)
            for k in range(0, len(photoCount)):
                path = os.path.join("./datasets/before", dirCount[i], imageCount[j], photoCount[k])

                if os.path.isfile(path) and path.endswith('json'):
                    data = json.load(open(path))
                
                    # 图片数据写入对象
                    if data['imageData']:
                        imageData = data['imageData']
                    else:
                        imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                        with open(imagePath, 'rb') as f:
                            imageData = f.read()
                            imageData = base64.b64encode(imageData).decode('utf-8')

                    # 对象转成数组
                    img = utils.img_b64_to_arr(imageData)
                    label_name_to_value = {'_background_': 0}
                    for shape in data['shapes']:
                        label_name = shape['label']
                        if label_name in label_name_to_value:
                            label_value = label_name_to_value[label_name]
                        else:
                            label_value = len(label_name_to_value)
                            label_name_to_value[label_name] = label_value
                    
                    # label_values must be dense
                    label_values, label_names = [], []
                    for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                        label_values.append(lv)
                        label_names.append(ln)
                    assert label_values == list(range(len(label_values)))
                    
                    lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
                    
                    jpgDir = osp.join(jpgs_path, dirCount[i], imageCount[j])
                    jpgPath = osp.join(jpgDir, photoCount[k].split(".")[0]+'.jpg')
                    if not osp.exists(jpgDir):
                        os.makedirs(jpgDir)
                        
                    PIL.Image.fromarray(img).save(jpgPath)

                    new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
                    for name in label_names:
                        index_json = label_names.index(name)
                        index_all = classes.index(name)
                        new = new + index_all*(np.array(lbl) == index_json)

                    pngDir = osp.join(pngs_path, dirCount[i], imageCount[j])
                    pngPath = osp.join(pngDir, photoCount[k].split(".")[0]+'.png')
                    if not osp.exists(pngDir):
                        os.makedirs(pngDir)

                    utils.lblsave(pngPath, new)
                    print('Saved ' + photoCount[k].split(".")[0] + '.jpg and ' + photoCount[k].split(".")[0] + '.png')
                    jsonNum += 1
    print(f"json 数量：{jsonNum}")