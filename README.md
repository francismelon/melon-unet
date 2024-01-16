# melon-unet

**melon-unet 是一个深度学习训练框架，你可以通过它来训练自己的模型，在这里你可以根据需要选择不同的主干网络来训练模型，现在暂时只有Unet主干、ResNet50和VGG16.**

## 依赖环境
**环境上的选择大同小异了，这里是作者在训练时使用的环境，如果版本差异不大，基本上都能用**
| python包 | 版本 |
| --- | --- |
scipy|1.10.1
numpy|1.24.4
matplotlib|3.7.2
opencv_python|4.1.2.30
torch|2.0.1+cu117
torchvision|0.15.2+cu117
tqdm|4.65.0
Pillow|9.5.0
labelme|3.16.7

## 预训练集
**作者使用的是网上有的预训练模型，只是数据集上有所不同，训练效果来说还算不错**
| 预训练模型 | 下载地址 | 密码 |
| --- | --- | --- |
vgg16 | https://pan.quark.cn/s/15134e3ce16d | rb53
resnet50 | https://pan.quark.cn/s/5d25ab97addb | 54bB

## 训练步骤
### 一、获取训练集
```
1) 使用labelme标注自己的数据集，并按照labelme的格式进行保存
2) 以`dir1/dir2/jpg`和`dir1/dir2/json`的格式保存数据集
3) 修改global_define.py中DEFINE_NAME_CLASSES的列表，请以用labelme标注的类别来区分
4) 运行json_to_dataset.py文件，运行完成后，可以在datasets文件夹下看到新生成的两个JPEGImages和SegmentationClass文件夹，这个py文件的代码按需修改即可
5) 将第四步生成的两个JPEGImages和SegmentationClass文件夹放到VOCdevkit/VOC2007这个文件夹下面
6) 运行voc_annotation.py文件，运行完成后，会在ImageSets/Segmentation下看到生成四个txt文件夹，train.txt、val.txt、trainval.txt、test.txt分别对应训练集、验证集、训练集+验证集、测试集的图片索引
```
### 二、修改训练参数
```
具体参数修改参考unet_train.py里面的提示，然后运行这个文件开始训练
```
### 三、等待训练完成
```
训练完成后，会在logs文件夹下生成一个best_epoch_weights.pth文件和其他后缀为pth的文件，best_epoch_weights这个文件基本就是多个epoch训练中最好的模型
```
### 四、开始验证
```
修改unet_eval.py文件参数（具体看里面注释内容），这个文件会加载训练好的模型。然后修改predict.py文件参数（具体看里面注释内容），选择要测试的模式，运行predict.py文件就能得到结果
```

## 其他
### 一、查看网络结构
```
运行summary.py文件，可以查看网络结构
```

### 二、评估
```
1) 修改global_define.py里面的DEFINE_NUM_CLASSES为预测的类的数量加1。  
2) 修改global_define.py里面的DEFINE_NAME_CLASSES为需要区分的类别。
3) 运行get_miou.py文件即可，等待评估完成。
```

### 三、引用
https://github.com/bubbliiiing/unet-pytorch