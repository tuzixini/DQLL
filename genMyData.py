# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: 处理TuSimpleLane 数据集
# 生成需要的数据
# genMyData.py
import json
import os
import os.path as osp
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import shutil
from tqdm import tqdm

faileList = []

def doit(sroot, jlistpath, droot,namelist=[],DRL_list=[],DRLcount=0):
    os.makedirs(droot, exist_ok=True)
    os.makedirs(osp.join(droot, 'img'), exist_ok=True)
    os.makedirs(osp.join(droot, 'mask'), exist_ok=True)
    os.makedirs(osp.join(droot, 'mask_color'), exist_ok=True)
    os.makedirs(osp.join(droot, 'bbox'), exist_ok=True)
    os.makedirs(osp.join(droot, 'DRL'), exist_ok=True)
    os.makedirs(osp.join(droot, 'DRL','ori'), exist_ok=True)
    os.makedirs(osp.join(droot, 'DRL', 'resize'), exist_ok=True)
    jlist =[]
    with open(jlistpath, 'r') as f:
        for line in f.readlines():
            jlist.append(json.loads(line))
    for ins in tqdm(jlist):
        faileIns = dict()
        imgpath = ins['raw_file']
        temp = imgpath.split('/')
        newname = temp[1] + temp[2] + temp[3][:-4]
        namelist.append(newname)
        spath = osp.join(sroot, imgpath)
        # 计算
        try:
            temp = getInfo(spath, ins)
        except:
            faileIns['sroot'] = sroot
            faileIns['jlistpath'] = jlistpath
            faileIns['ins'] = ins
            faileList.append(faileIns)
            continue
        mask, mask_color, bbox, box, box_mask, gt, rebox, rebox_mask, regt = temp
        if len(box) > 5:
            faileIns['sroot'] = sroot
            faileIns['jlistpath'] = jlistpath
            faileIns['ins'] = ins
            faileIns['Reason']="cot>5"
            faileList.append(faileIns)
            continue
        #  copyimg
        dpath = osp.join(droot, 'img', newname + '.jpg')
        shutil.copy(spath, dpath)
        # mask
        dpath = osp.join(droot, 'mask', newname + '.png')
        mask = Image.fromarray(mask.astype('uint8'))
        mask.save(dpath)
        # mask_color
        dpath = osp.join(droot, 'mask_color', newname + '.png')
        mask_color = Image.fromarray(mask_color.astype('uint8'))
        mask_color.save(dpath)
        # bbox
        dpath = osp.join(droot, 'bbox', newname + '.json')
        with open(dpath,'w') as f:
            json.dump(bbox,f)
        # box 裁剪出来的图片
        for i in range(len(box)):
            # 获取裁剪出来图片的名称
            DRLname = newname + '_' + str(DRLcount)
            DRL_list.append(DRLname)
            # box
            temp = box[i]
            temp = Image.fromarray(temp.astype('uint8'))
            dpath = osp.join(droot,'DRL','ori',DRLname+'.png')
            temp.save(dpath)
            # boxmask
            temp = box_mask[i]
            temp = Image.fromarray(temp.astype('uint8'))
            dpath = osp.join(droot, 'DRL', 'ori', DRLname+'_mask.png')
            temp.save(dpath)
            # boxmask_color
            boxmask = box_mask[i]
            temp = np.zeros(boxmask.shape)
            temp[boxmask == 1] = 255
            temp = Image.fromarray(temp.astype('uint8'))
            dpath = osp.join(droot, 'DRL', 'ori', DRLname + '_mask_color.png')
            temp.save(dpath)
            # gt
            dpath = osp.join(droot,'DRL','ori',DRLname+'.json')
            with open(dpath, 'w') as f:
                json.dump(gt[i], f)
            # rebox
            temp = rebox[i]
            temp = Image.fromarray(temp.astype('uint8'))
            dpath = osp.join(droot, 'DRL', 'resize', DRLname+'.png')
            temp.save(dpath)
            # reboxmask
            temp = rebox_mask[i]
            temp = Image.fromarray(temp.astype('uint8'))
            dpath = osp.join(droot, 'DRL', 'resize', DRLname+'_mask.png')
            temp.save(dpath)
            # reboxmask_color
            boxmask = rebox_mask[i]
            temp = np.zeros(boxmask.shape)
            temp[boxmask == 1] = 255
            temp = Image.fromarray(temp.astype('uint8'))
            dpath = osp.join(droot, 'DRL', 'resize',DRLname + '_mask_color.png')
            temp.save(dpath)
            # regt
            dpath = osp.join(droot, 'DRL', 'resize', DRLname+'.json')
            with open(dpath, 'w') as f:
                json.dump(regt[i], f)
            DRLcount += 1
    return namelist,DRL_list,DRLcount


def getInfo(ipath,data):
    img = Image.open(ipath)
    img = np.array(img)
    img_t = np.zeros(img.shape)
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask_color = np.zeros((img.shape[0], img.shape[1]))
    gt_lanes_vis = [[(x, y)for(x, y)in zip(lane, data['h_samples'])if x >= 0]
                    for lane in data['lanes']]

    for lane in gt_lanes_vis:
        cv2.polylines(img_t, np.int32(
            [lane]), isClosed=False, color=(0, 255, 0), thickness=5)
    mask_color[img_t[:, :, 1] == 255] = 255
    mask[img_t[:, :, 1] == 255] = 1
    # 计算bbox
    temp = Image.fromarray(mask.astype('uint8'))
    temp = np.array(temp)
    bbox = []
    box = []
    box_mask = []
    gt = []
    rebox = []
    rebox_mask = []
    regt = []
    temp, cons, hier = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for con in cons:
        x, y, w, h = cv2.boundingRect(con)
        # 截取原图
        box.append(img[y:y+h, x:x+w, :].astype('uint8'))
        # 截取mask
        temp = np.zeros(img.shape)
        cv2.drawContours(temp, [con], 0, (0, 0, 255), -1)
        ttemp = np.zeros((img.shape[0], img.shape[1]))
        ttemp[temp[:, :, 2] == 255] = 1
        box_mask.append(ttemp[y:y+h, x:x+w].astype('uint8'))
        # 计算bbox [{'class':cl,'points',[x1,y1,x2,y2]},{}]
        temp = ttemp[y:y+h, x:x+w]
        [vx, vy, xx, yy] = cv2.fitLine(con, cv2.DIST_L2, 0, 0.01, 0.01)
        slope = -float(vy)/float(vx)
        if slope <= 0:
            # 左上 到右下
            cl = 0
        else:
            # 左下到右上
            cl = 1
        ttemp = dict()
        ttemp['points'] = [int(x), int(y), int(x+w), int(y+h)]
        ttemp['class'] = cl
        bbox.append(ttemp)
        # 计算gt[{'class':cl,'gt':[x1,x2,x3,x4,x5]},{}]
        ttemp = dict()
        ttemp['class'] = cl
        initY = []
        for i in range(5):
            initY.append(int((i+1)*(h/6)))
        initX = []
        for y in initY:
            xx = temp[y, :]
            xx = np.where(xx == 1)
            x = int((np.max(xx)+np.min(xx))/2)
            initX.append(x)
        ttemp['gt'] = initX
        gt.append(ttemp)
    # 生成resize的DRL材料
    for i in range(len(box)):
        temp = box[i].copy()
        temp = cv2.resize(temp, (100, 100))
        rebox.append(temp)

        temp = box_mask[i].copy()
        temp = cv2.resize(temp, (100, 100))
        rebox_mask.append(temp)
        # pdb.set_trace()

        ttemp = dict()
        ttemp['class'] = gt[i]['class']
        initY = [11, 31, 51, 71, 91]
        initX = []
        for y in initY:
            xx = temp[y, :]
            xx = np.where(xx == 1)
            x = int((np.max(xx)+np.min(xx))/2)
            initX.append(x)
        ttemp['gt'] = initX
        regt.append(ttemp)
    result = [mask, mask_color, bbox, box, box_mask, gt, rebox, rebox_mask, regt]
    return result

DATAROOT = r"/opt/disk/zzy/dataset/TuSimple"

sroot = 'train_set'
jlistpath = r'train_set/label_data_0531.json'
droot = 'MyTuSimpleLane/train'
sroot = osp.join(DATAROOT, sroot)
jlistpath = osp.join(DATAROOT, jlistpath)
droot = osp.join(DATAROOT,droot)
namelist, DRL_list,DRLcount = doit(sroot, jlistpath, droot, namelist=[], DRL_list=[], DRLcount=0)
print(len(faileList))

sroot = 'train_set'
jlistpath = r'train_set/label_data_0313.json'
droot = 'MyTuSimpleLane/train'
sroot = osp.join(DATAROOT, sroot)
jlistpath = osp.join(DATAROOT, jlistpath)
droot = osp.join(DATAROOT,droot)
namelist, DRL_list, DRLcount = doit(sroot, jlistpath, droot, namelist=namelist, DRL_list=DRL_list, DRLcount=DRLcount)
print(len(faileList))

sroot = 'train_set'
jlistpath = r'train_set/label_data_0601.json'
droot = 'MyTuSimpleLane/train'
sroot = osp.join(DATAROOT, sroot)
jlistpath = osp.join(DATAROOT, jlistpath)
droot = osp.join(DATAROOT,droot)
namelist, DRL_list, DRLcount = doit(sroot, jlistpath, droot, namelist=namelist, DRL_list=DRL_list, DRLcount=DRLcount)
print(len(faileList))
with open(osp.join(DATAROOT,'train_img_list.json'), 'w') as f:
    json.dump(namelist, f)

with open(osp.join(DATAROOT,'train_DRL_list.json'), 'w') as f:
    json.dump(DRL_list,f)


sroot = 'test_set'
jlistpath = r'test_label.json'
droot = 'MyTuSimpleLane/test'
sroot = osp.join(DATAROOT, sroot)
jlistpath = osp.join(DATAROOT, jlistpath)
droot = osp.join(DATAROOT,droot)
namelist, DRL_list, DRLcount = doit(sroot, jlistpath, droot, namelist=[], DRL_list=[], DRLcount=0)
print(len(faileList))
with open(osp.join(DATAROOT,'test_img_list.json'), 'w') as f:
    json.dump(namelist, f)

with open(osp.join(DATAROOT,'test_DRL_list.json'), 'w') as f:
    json.dump(DRL_list,f)


with open(osp.join(DATAROOT,'failList.json'), 'w') as f:
    json.dump(faileList, f)
