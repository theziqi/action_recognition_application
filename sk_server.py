# -*- coding:utf-8 -*-
# @time :2022/1/25
# @IDE : pycharm
# @author : theziqi
# @github : https://github.com/theziqi
# @Email : qigo@hotmail.com

import socket
import cv2
import base64
import numpy as np
import json
from threading import Thread

from redis_db import classify_process
from utils import db, base64_encode_image
from cfg import *

HOST = 'localhost'
PORT = 5000
ADDRESS = (HOST, PORT)


def send_results():
    while True:
        print("等待连接……")
        client_socket, client_address = tcpServer.accept()
        print("连接成功！")
        try:
            while True:
                # 接收标志数据
                data = client_socket.recv(1024)
                if "fg" == data.decode():
                    print("收到flag，开始接收图片")
                    data = client_socket.recv(1024)
                    if data:
                        # 通知客户端“已收到标志数据，可以发送图像数据”
                        client_socket.send(b"is")
                        # 处理标志数据
                        flag = data.decode().split(",")
                        # 图像字节流数据的总长度
                        total = int(flag[0])
                        # 接收到的数据计数
                        cnt = 0
                        # 存放接收到的数据
                        img_bytes = b""

                        while cnt < total:
                            # 当接收到的数据少于数据总长度时，则循环接收图像数据，直到接收完毕
                            data = client_socket.recv(256000)
                            img_bytes += data
                            cnt += len(data)
                            # 已接收到的字节大小
                            print("receive:" + str(cnt) + "/" + flag[0])
                        # 通知客户端“已经接收完毕，可以开始下一帧图像的传输”
                        client_socket.send(b"ie")
                        # 将图片转换成base64
                        img = np.asarray(bytearray(img_bytes), dtype="uint8")
                        img_np = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        image = cv2.imencode('.jpg', img_np)[1]
                        image_code = str(base64.b64encode(image))[2:-1]
                        # 存入redis
                        db.lpush(ImageQueue, image_code)
                        db.ltrim(ImageQueue, 0, SampleLength - 1)

                        # 解析接收到的字节流数据，并显示图像
                        # img = np.asarray(bytearray(img_bytes), dtype="uint8")
                        # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        # cv2.imshow("img", img)
                        # cv2.waitKey(1)
                    else:
                        print("已断开！")
                        break

                # 发送算法处理结果
                client_socket.send(b'rs')
                output = db.get(ResultValue)
                if output is not None:
                    client_socket.send(output.encode('utf8'))
                    print('Send result!')
                # client_socket.send(b're')
        finally:
            client_socket.close()


if __name__ == '__main__':
    global tcpServer
    # 创建一个套接字
    tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定本地ip
    tcpServer.bind(ADDRESS)
    # 开始监听
    tcpServer.listen(5)
    # 创建tcp连接进程和算法处理进程
    try:
        pw = Thread(target=send_results, args=(), daemon=True)
        pr = Thread(target=classify_process, args=(ModelPath, LabelPath), daemon=True)
        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass
