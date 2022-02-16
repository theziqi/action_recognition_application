# -*- coding:utf-8 -*-
# @time :2022/1/25
# @IDE : pycharm
# @author : theziqi
# @github : https://github.com/theziqi
# @Email : qigo@hotmail.com

from threading import Thread
from collections import deque
from operator import itemgetter
import time
import cv2

import cfg
from cfg import *
from utils import base64_encode_image

import socketio

# 请求的URL
REST_API_URL = "http://127.0.0.1:5000"

# 并发数
NUM_REQUESTS = 500
# 请求间隔
SLEEP_COUNT = 0.05

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def create_client():
    sio = socketio.Client(logger=True, engineio_logger=True)

    @sio.event
    def connect():
        print("I'm connected!")
        sio.emit('ready_for_result', 'yes!')

    @sio.event
    def ready_for_frame(msg):
        print('ready for frame: ', msg)
        while True:
            frame_base64 = base64_encode_image(frame)
            sio.emit('frame', frame_base64)
            time.sleep(ClientSleep)
            print('Send a frame!')

    @sio.event
    def connect_error():
        print("The connection failed!")

    @sio.event
    def disconnect():
        print("I'm disconnected!")

    @sio.event
    def result(data):
        print('Receive results!')
        result_queue.append(data)

    sio.connect(REST_API_URL)
    sio.wait()


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
    global camera, frame, result_queue

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
