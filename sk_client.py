# -*- coding:utf-8 -*-
# @time :2022/1/25
# @IDE : pycharm
# @author : theziqi
# @github : https://github.com/theziqi
# @Email : qigo@hotmail.com

import cv2
import time
import socket
import json
from collections import deque
from threading import Thread

import cfg

# 服务端ip地址
HOST = '127.0.0.1'
# 服务端端口号
PORT = 5000
ADDRESS = (HOST, PORT)

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def create_client():
    while True:
        # 计时
        start = time.perf_counter()
        # 读取图像
        # ref, cv_image = camera.read()
        # 压缩图像
        img_encode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 99])[1]
        # 转换为字节流
        bytedata = img_encode.tobytes()
        # 标志数据，包括待发送的字节流长度等数据，用‘,’隔开
        flag_data = (str(len(bytedata))).encode() + ",".encode() + " ".encode()
        tcpClient.send(b'fg')
        print('发送flag')
        tcpClient.send(flag_data)

        # 接收服务端开始传输图片的应答
        data = tcpClient.recv(1024)
        if "is" == data.decode():
            # 服务端已经收到标志数据，开始发送图像字节流数据
            tcpClient.send(bytedata)
        # 接收服务端传输图片完毕的应答
        data = tcpClient.recv(1024)
        if "ie" == data.decode():
            # 计算发送完成的延时
            print("延时：" + str(int((time.perf_counter() - start) * 1000)) + "ms")
        # 接收服务端开始传输算法结果的应答
        data = tcpClient.recv(1024)
        if "rs" == data.decode():
            result = tcpClient.recv(1024).decode()
            print(result)
            result_queue.append(json.loads(result))
            # print(result)
        # 接收服务端传输算法结果完毕的应答
        # data = tcpClient.recv(1024)
        # if "re" == data.decode():
        #     print("接收结果！")


def show_results():
    print('Press "Esc", "q" or "Q" to exit')

    text_info = {}
    cur_time = time.time()
    while True:
        msg = 'Waiting for action ...'
        global frame
        _, frame = camera.read()

        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result
                score = float(score)
                if score < cfg.Threshold:
                    break
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score, 2))
                text_info[location] = text
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        elif len(text_info) != 0:
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

        cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        if cfg.DrawingFPS > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / cfg.DrawingFPS - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    global tcpClient, camera, frame, result_queue
    # 创建一个套接字
    tcpClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接远程ip
    tcpClient.connect(ADDRESS)

    camera = cv2.VideoCapture(cfg.CameraId)

    try:
        # frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)
        _, frame = camera.read()
        pw = Thread(target=show_results, args=(), daemon=True)
        pr = Thread(target=create_client, args=(), daemon=True)
        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass
