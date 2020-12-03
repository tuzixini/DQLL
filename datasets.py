# coding=utf-8
# ----tuzixini----
# MACOS Python3.6.6
'''
载入 self_lane数据集
'''
import pdb
import collections
from torch.utils import data
from scipy import io as sio
from torch.utils.data import DataLoader
import os.path as osp
import json
import numpy as np
from PIL import Image


def getData(cfg):
    if cfg.DATA.NAME =='SelfLane':
        trainset = SelfLane(cfg.DATA.TRAIN_LIST)
        valset = SelfLane(cfg.DATA.VAL_LIST)
        trainloader = DataLoader(trainset, 
                                num_workers=cfg.DATA.NUM_WORKS,
                                batch_size=cfg.DATA.TRAIN_IMGBS, 
                                shuffle=cfg.DATA.IMGSHUFFLE)
        valloader = DataLoader(valset, 
                            num_workers=cfg.DATA.NUM_WORKS,
                            batch_size=cfg.DATA.VAL_IMGBS, 
                            shuffle=cfg.DATA.IMGSHUFFLE)
        meanImg = sio.loadmat(cfg.DATA.MEAN_IMG_PATH)
        meanImg = meanImg['meanImg']
        return meanImg,trainloader, valloader
    if cfg.DATA.NAME == 'TuSimpleLane':
        trainset = TuSimpleLane(cfg.DATA.ROOT,cfg.DATA.TRAIN_LIST,isTrain=True)
        valset =TuSimpleLane(cfg.DATA.ROOT,cfg.DATA.VAL_LIST,isTrain=False)
        trainloader =DataLoader(trainset,batch_size=cfg.DATA.TRAIN_IMGBS,shuffle=cfg.DATA.IMGSHUFFLE,num_workers=cfg.DATA.NUM_WORKS)
        valloader =DataLoader(valset,batch_size=cfg.DATA.VAL_IMGBS,shuffle=cfg.DATA.IMGSHUFFLE,num_workers=cfg.DATA.NUM_WORKS)
        meanImg =np.load(cfg.DATA.MEAN_IMG_PATH)
        return meanImg,trainloader,valloader

class TuSimpleLane(data.Dataset):
    def __init__(self, dataroot, ListPath, isTrain=True,im_tf=None, gt_tf=None):
        if isTrain:
            self.root = osp.join(dataroot, 'train')
        else:
            self.root = osp.join(dataroot, 'test')
        self.root = osp.join(self.root, 'DRL', 'resize')
        with open(ListPath, 'r') as f:
            self.pathList= json.load(f)
        self.im_tf = im_tf
        self.gt_tf = gt_tf

    def __getitem__(self, index):
        # img
        temp = osp.join(self.root, self.pathList[index] + '.png')
        img = np.array(Image.open(temp))
        temp = osp.join(self.root, self.pathList[index] + '.json')
        with open(temp, 'r') as f:
            data = json.load(f)
        img = np.array(img)
        img = img.astype(np.float32)
        cla = np.array(data['class'])
        gt = np.array(data['gt'])
        return cla, img, gt

    def __len__(self):
        return len(self.pathList)


class SelfLane(data.Dataset):
    def __init__(self, pathList, im_tf=None, gt_tf=None):
        self.pathList = pathList
        self.im_tf = im_tf
        self.gt_tf = gt_tf

    def __getitem__(self, index):
        temp = sio.loadmat(self.pathList[index])
        img = temp['img']
        img = np.array(Image.fromarray(img).resize((100,100)))
        img =img.astype(np.float32)
        # fea = temp['fea']
        cl = np.array(int(temp['class_name'][0]))
        gt = np.array(temp['mark'][0])
        return cl, img, gt

    def __len__(self):
        return len(self.pathList)


class bufferLoader(data.Dataset):
    def __init__(self, buffer, tf=None):
        self.buffer = buffer
        self.tf = tf

    def __getitem__(self, index):
        fea, state, Q = self.buffer[index]
        fea = np.array(fea).astype(np.float32)
        state = np.array(state).astype(np.float32)
        Q = np.array(Q).astype(np.float32)
        if self.tf is not None:
            fea = self.tf(fea)
        return fea, state, Q

    def __len__(self):
        return len(self.buffer)
