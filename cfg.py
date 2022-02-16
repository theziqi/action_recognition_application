# -*- coding:utf-8 -*-
# @time :2022/1/25
# @IDE : pycharm
# @author : theziqi
# @github : https://github.com/theziqi
# @Email : qigo@hotmail.com


'''
定义web服务所需的一些配置参数
'''

CameraId = 0

# 输入图像的大小
InputSize = (224, 224)

# 输入图像的通道数
Channel = 3

# 数据类型
ImageType = 'float32'

# 模型运行设备
Device = 'cpu'

# redis运行的图像队列
ImageQueue = 'image_queue'

# redis运行的结果键值
ResultValue = 'result_value'

ServeSleep = 0.5

ClientSleep = 0.5

ModelPath = './model/slowfast_k400.onnx'

LabelPath = './model/label_map_k400.txt'

Threshold = 0.2

AverageSize = 3

SampleLength = 16

InferenceFPS = 4

DrawingFPS = 20
