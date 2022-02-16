# -*- coding:utf-8 -*-
# @time :2022/1/25
# @IDE : pycharm
# @author : theziqi
# @github : https://github.com/theziqi
# @Email : qigo@hotmail.com

from PIL import Image

import io
from threading import Thread
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import time

import cfg
from redis_db import classify_process
import json
import uuid

from cfg import *
from utils import db, image_transform, base64_encode_image

# 初始化实例
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

client_connecting = False
pc = None
pr = None


@socketio.on('connect')
def connect():
    print('Client connected.')
    global client_connecting, pc
    client_connecting = True
    socketio.emit('ready_for_frame', 'yes!')
    if not pc:
        pc = socketio.start_background_task(classify_process, cfg.ModelPath, cfg.LabelPath)


@socketio.on('disconnect')
def disconnect():
    print('Client disconnected.')
    global client_connecting
    client_connecting = False


@socketio.on('frame')
def handle_frame(img):
    print('Receive a frame!')
    if img:
        db.lpush(ImageQueue, img)
        db.ltrim(ImageQueue, 0, SampleLength - 1)


@socketio.on('ready_for_result')
def ready_for_result(msg):
    print('ready for result: ', msg)
    global pr
    pr = socketio.start_background_task(send_results)


def send_results():
    # 运行服务
    while True:
        # 获取输出结果
        # output = db.lrange(ResultQueue, 0, AverageSize - 1)
        output = db.get(ResultValue)
        if output is not None:
            # r = [json.loads(o) for o in output]
            r = json.loads(output)
            socketio.emit('result', r)
            print('Send result!')
        if not client_connecting:
            break
        time.sleep(ServeSleep)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
