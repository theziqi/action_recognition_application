# -*- coding:utf-8 -*-
# @time :2022/1/25
# @IDE : pycharm
# @author : theziqi
# @github : https://github.com/theziqi
# @Email : qigo@hotmail.com


import numpy as np
from PIL import Image

from flask import Flask
from cfg import *
import redis
from io import BytesIO
import base64
import cv2
import sys

from torchvision import transforms


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def image_transform(inputsize):
    return transforms.Compose([
        Resize((int(inputsize[0] * (256 / 224)), int(inputsize[1] * (256 / 224)))),
        transforms.CenterCrop(inputsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def decode_predictions(predition):
    return predition


# 构建redis数据库
db = redis.StrictRedis(host="127.0.0.1", port=6379, db=0, decode_responses=True)


def base64_encode_image(frame):
    # 面向opencv视频帧
    img = Image.fromarray(frame)  # 将每一帧转为Image
    output_buffer = BytesIO()  # 创建一个BytesIO
    img.save(output_buffer, format='JPEG')  # 写入output_buffer
    byte_data = output_buffer.getvalue()  # 在内存中读取
    base64_data = base64.b64encode(byte_data)  # 转为BASE64
    return base64_data


def base64_decode_image(img, dtype, shape):
    # base64 decode should meet the padding rules
    # if len(img) % 3 == 1:
    #     img += "=="
    # elif len(img) % 3 == 2:
    #     img += "="
    img_b64decode = base64.b64decode(img)  # base64解码
    img_array = np.fromstring(img_b64decode, np.uint8)  # 转换np序列
    # img_array = np.frombuffer(base64.decodebytes(img), dtype=dtype)
    img_n = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)  # 转换Opencv格式
    img_n = cv2.resize(img_n, shape)
    img_n = np.array(img_n[:, :, ::-1], dtype=dtype)
    # # 查看python版本,如果是python3版本进行转换
    # if sys.version_info.major == 3:
    #     img = bytes(img, encoding="utf-8")
    # img = np.frombuffer(base64.decodebytes(img), dtype=dtype)
    # img = img.reshape(shape)
    return img_n


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    image = image.resize(target)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    # return the processed image
    return image
