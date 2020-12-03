# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: DRL_Lane Pytorch 实现
# config.py
import os
import os.path as osp
import time
from easydict import EasyDict as edict

# init
__C = edict()
cfg = __C
__C.EXP = edict()
__C.DATA = edict()
__C.TRAIN = edict()
__C.TEST =edict()

# train*************************train
__C.TRAIN.LR = 1e-4
__C.TRAIN.WEIGHT_DECAY = 5e-4
__C.TRAIN.MAX_EPOCH = 100
# 每个buffer 训练的epoch数量
__C.TRAIN.INER_EPOCH = 10
# GPU 设置
__C.TRAIN.USE_GPU = True
if __C.TRAIN.USE_GPU:
    __C.TRAIN.GPU_ID = [2, 3]
# 断点续训
__C.TRAIN.RESUME = False
__C.TRAIN.RESUME_PATH = '20-04-18-10-30_TuSimpleLane/EP_9_HitRat0.60316.pth'

# test
__C.TEST.BS = 1

# data*************************data
__C.DATA.NAME = 'TuSimpleLane'  # SelfLane/TuSimpleLane
if __C.DATA.NAME == 'TuSimpleLane':
    __C.DATAROOT = r'/opt/disk/zzy/datasets/TuSimpleLane'
    __C.DATA.TRAIN_LIST = osp.join(__C.DATAROOT,'train_DRL_list.json')
    __C.DATA.VAL_LIST = osp.join(__C.DATAROOT,'test_DRL_list.json')
    __C.DATA.ROOT = osp.join(__C.DATAROOT,'MyTuSimpleLane')
    # meanImagePath
    __C.DATA.MEAN_IMG_PATH = osp.join(__C.DATAROOT,r'meanImgTemp.npy')
if __C.DATA.NAME == 'SelfLane':
    __C.DATA.TRAIN_LIST =''#  TODO:
    __C.DATA.VAL_LIST = ''  #  TODO:
    # meanImagePath
    __C.DATA.MEAN_IMG_PATH =r''#TODO:
# buffer 的dataloader的设置
__C.DATA.NUM_WORKS = 8
__C.DATA.BS = 2048
__C.DATA.SHUFFLE = True
# img dataloder的设置
__C.DATA.TRAIN_IMGBS = 100  #  TODO:
__C.DATA.VAL_IMGBS =1#  TODO:
__C.DATA.IMGSHUFFLE = True


# DQL*************************DQL
# 最大步数
__C.MAX_STEP = 10
# 距离阈值
__C.DST_THR = 5
# action 数量 确定为4
__C.ACT_NUM = 4
# History数量
__C.HIS_NUM = 8
# epsilon
__C.EPSILON = 1
# gamma
__C.GAMMA = 0.90
# landmark 数量
__C.LANDMARK_NUM = 5
# reward
__C.reward_terminal_action = 3
__C.reward_movement_action = 1
__C.reward_invalid_movement_action = -5
__C.reward_remove_action = 1
# buffer capacity
__C.BUFFER_CAP = 20480*5


# exp*************************exp
__C.SEED = 666
__C.EXP.ROOT = 'exp'
now = time.strftime("%y-%m-%d-%H-%M", time.localtime())
__C.EXP.NAME = now+'_'+__C.DATA.NAME
__C.EXP.PATH = os.path.join(__C.EXP.ROOT,__C.EXP.NAME)