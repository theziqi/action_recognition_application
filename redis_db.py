# -*- coding:utf-8 -*-
# @time :2022/1/25
# @IDE : pycharm
# @author : theziqi
# @github : https://github.com/theziqi
# @Email : qigo@hotmail.com

import torch
import numpy as np
import time
import onnxruntime as rt

import cfg
import json
from utils import db, base64_decode_image, decode_predictions
from collections import deque
from operator import itemgetter


def classify_process(modelPath, labelPath):
    sess = rt.InferenceSession(modelPath)
    with open(labelPath, 'r') as f:
        label = [line.strip() for line in f]

    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    while True:
        cur_windows = []
        while len(cur_windows) == 0:
            if db.llen(cfg.ImageQueue) == cfg.SampleLength:
                frame_list = db.lrange(cfg.ImageQueue, 0, cfg.SampleLength - 1)
                frame_list_t = [base64_decode_image(f, np.float32, cfg.InputSize) for f in frame_list]
                cur_windows = frame_list_t
        # HWC->CHW
        cur_windows = np.transpose(np.array(cur_windows), (3, 0, 1, 2))
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

        if len(score_cache) == cfg.AverageSize:
            scores_avg = scores_sum / cfg.AverageSize
            num_selected_labels = min(len(label), 5)

            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]
            results = np.array(results).tolist()
            # print(results)
            # db.rpush(cfg.ResultQueue, json.dumps(results))
            # db.ltrim(cfg.ResultQueue, 0, cfg.AverageSize - 1)
            db.set(cfg.ResultValue, json.dumps(results))
            # result_queue.append(results)
            scores_sum -= score_cache.popleft()

        if cfg.InferenceFPS > 0:
            # add a limiter for actual inference fps <= inference_fps
            sleep_time = 1 / cfg.InferenceFPS - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()


if __name__ == '__main__':
    classify_process(cfg.ModelPath, cfg.LabelPath)
