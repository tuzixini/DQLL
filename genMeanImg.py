# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 计算 裁剪 resize 之后的TuSimple数据的meanImage
import os.path as osp
import json
from PIL import Image
import numpy as np
from tqdm import tqdm


def genMeanImg(jsonListPath, root):
    with open(jsonListPath, 'r') as f:
        jsonList = json.load(f)
    meanImg = np.array(Image.open(osp.join(root,jsonList[0]+'.png')))
    for name in tqdm(jsonList):
        temp = osp.join(root, name + '.png')
        img = np.array(Image.open(temp))
        meanImg = (meanImg + img) / 2
    return meanImg

DATAROOT = r"/opt/disk/zzy/dataset/TuSimple"

jsonListPath = osp.join(DATAROOT,'train_DRL_list.json')
root = osp.join(DATAROOT,'MyTuSimpleLane/train/DRL/resize/')
meanImg = genMeanImg(jsonListPath, root)
print(meanImg.shape)
print(meanImg)
savePath = osp.join(DATAROOT,'meanImgTemp.npy')
np.save(savePath,meanImg)
