# coding=utf-8
# ----tuzixini----
# WIN10 Python3.6.6
# tools/utils.py
'''
存放一些通用的工具
'''
import os
import cv2
import pdb
import time
import copy
import torch
import shutil
import random
import collections
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.clean()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clean(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


def copy_cur_env(work_dir, dst_dir, exception='exp'):
    # 复制本次运行的工作环境,排除部分文件夹
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for filename in os.listdir(work_dir):
        file = os.path.join(work_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        if os.path.isdir(file) and exception not in filename:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file, dst_file)


class ExBuffer(object):
    def __init__(self, capacity, flashRat=1):
        self.cap = capacity
        self.replaceInd = 0
        self.replaceMax = int(capacity*flashRat)
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, exp):
        self.buffer.append(exp)
        self.replaceInd += 1

    @property
    def isFull(self):
        if len(self.buffer) < self.cap:
            return False
        else:
            return True

    @property
    def ready2train(self):
        if self.isFull:
            if self.replaceInd < self.replaceMax:
                return False
            else:
                self.replaceInd = 0
                return True
        else:
            return False
    

    def samlpe(self, BS):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)

    def clean(self):
        self.buffer.clear()
        self.replaceInd = 0


def visOneLane(img, meanImg, gt, initX,xpoints):
    # xpoints{'5':[2,3,4],'4':[1,1,1]}
    margin = 6
    temp = np.ones((margin*2+100, margin*2+100, 3)) * 255
    img = img + meanImg
    temp[margin:margin+100, margin:margin+100,:] = img
    initY = np.array([11, 31, 51, 71, 91])
    initY = initY + margin
    initX = np.array(initX)
    initX = initX + margin
    gt = gt+margin
    for i in xpoints.keys():
        xpoints[i] = np.array(xpoints[i]) + margin
    finalImg = []
    # 画gt
    for i in range(5):
        if gt[i]>0:
            x = gt[i]
            y = initY[i]
            pt = (int(x),int(y))
            cv2.circle(temp, pt, 8, (0, 0, 255), 2)

    # 五行
    for k in np.arange(5, 0, -1):
        # 画一行
        oneLine = []
        ttemp = copy.deepcopy(temp)
        # 画当前行的初始图(除了当前点之外的所有点)
        for i in np.arange(5, k, -1):
            x = xpoints[str(i-1)][-1]
            y = initY[i - 1]
            pt = (int(x),int(y))
            cv2.line(ttemp, pt, pt, (255, 0, 0), 4)
        for i in np.arange(k - 1, 0, -1):
            # pdb.set_trace()
            x = xpoints[str(i - 1)][0]
            y = initY[i - 1]
            pt = (int(x),int(y))
            cv2.line(ttemp, pt, pt, (255, 0, 0), 4)
        y = initY[k-1]
        for i in range(len(xpoints[str(k-1)])):
            tttemp = copy.deepcopy(ttemp)
            x = xpoints[str(k-1)][i]
            pt = (int(x),int(y))
            cv2.line(tttemp, pt, pt, (0, 255, 0), 4)
            # pdb.set_trace()
            oneLine.append(tttemp)
        finalImg.append(oneLine)
    return finalImg

def catFinalImg(finalImg):
    maxlen = 0
    for line in finalImg:
        if len(line) > maxlen:
            maxlen = len(line)
    x = finalImg[0][0].shape[0]
    w = x*maxlen
    h = x*5
    img = np.ones((h, w, 3)) * 255
    for i in range(len(finalImg)):
        for j in range(len(finalImg[i])):
            img[x * i:x * (i + 1), x * j:x * (j + 1),:] = finalImg[i][j]
    return img
