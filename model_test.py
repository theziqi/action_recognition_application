# -*- coding:utf-8 -*-
# @time :2022/1/25
# @IDE : pycharm
# @author : theziqi
# @github : https://github.com/theziqi
# @Email : qigo@hotmail.com

import time
from collections import deque
from operator import itemgetter
from threading import Thread

import cv2
import numpy as np
import onnxruntime as rt

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

checkpoint = './model/slowfast_k400.onnx'
labelPath = './model/label_map_k400.txt'
device = 'cuda:0'
cameraId = 0
threshold = 0.2
average_size = 3
sample_length = 16
inference_fps = 4
drawing_fps = 20


def show_results():
    print('Press "Esc", "q" or "Q" to exit')

    text_info = {}
    cur_time = time.time()
    while True:
        msg = 'Waiting for action ...'
        _, frame = camera.read()
        frame_resized = cv2.resize(frame, (224, 224))
        frame_queue.append(np.array(frame_resized[:, :, ::-1]))

        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result
                if score < threshold:
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

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()


def inference():
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    while True:
        cur_windows = []
        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue, dtype=np.float32))
        # HWC->CHW
        cur_windows = np.transpose(cur_windows, (3, 0, 1, 2))
        # Normalization
        norm_arg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375]
        )
        cur_windows[0, :, :, :] = (cur_windows[0, :, :, :] - norm_arg['mean'][0]) / norm_arg['std'][0]
        cur_windows[1, :, :, :] = (cur_windows[1, :, :, :] - norm_arg['mean'][1]) / norm_arg['std'][1]
        cur_windows[2, :, :, :] = (cur_windows[2, :, :, :] - norm_arg['mean'][2]) / norm_arg['std'][2]
        # CHW->BNCHW, Batch=1, Num_clips=1
        cur_windows = cur_windows[np.newaxis, np.newaxis, :, :, :, :]
        # print(cur_windows)

        input_name = sess.get_inputs()[0].name
        scores = sess.run([], {input_name: cur_windows})[0][0]
        # print(scores)

        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()

        if inference_fps > 0:
            # add a limiter for actual inference fps <= inference_fps
            sleep_time = 1 / inference_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()

    camera.release()
    cv2.destroyAllWindows()


def main():
    global camera, sess, data, label, frame_queue, result_queue

    sess = rt.InferenceSession(checkpoint)
    camera = cv2.VideoCapture(cameraId)
    data = dict(img_shape=None, modality='RGB', label=-1)

    with open(labelPath, 'r') as f:
        label = [line.strip() for line in f]

    try:
        frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)
        pw = Thread(target=show_results, args=(), daemon=True)
        pr = Thread(target=inference, args=(), daemon=True)
        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
