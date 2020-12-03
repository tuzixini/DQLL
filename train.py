# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: DRL_Lane Pytorch 实现
# train.py
import os
import pdb
import torch
import scipy
import random
import collections
import numpy as np
import os.path as osp
from tqdm import tqdm
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from PIL import Image

from config import cfg
import utils
import datasets
import model
import reward
from utils import Timer

class trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(self.cfg.EXP.PATH, exist_ok=True)
        os.makedirs(self.cfg.EXP.PATH+'/valimg', exist_ok=True)
        # logger
        self.writer = SummaryWriter(self.cfg.EXP.PATH)
        # 计时器
        self.t = {'iter': Timer(), 'train': Timer(), 'val': Timer()}
        # 保存实验环境 # TODO: 启用
        temp = os.path.join(self.cfg.EXP.PATH, 'code')
        utils.copy_cur_env('./', temp, exception='exp')
        # 读取数据集
        self.meanImg, self.trainloader, self.valloader = datasets.getData(self.cfg)
        # 定义网络
        self.net = model.getModel(cfg)
        # 损失函数
        self.criterion = torch.nn.MSELoss()
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        # 初始化一些变量
        self.beginEpoch = 1
        self.batch = 1
        self.bestacc = 0
        # 载入预训练模型
        if self.cfg.TRAIN.RESUME:
            print('Loading Model..........')
            saved_state = torch.load(self.cfg.TRAIN.RESUME_PATH)
            self.net.load_state_dict(saved_state['weights'])
            self.beginEpoch = saved_state['epoch']
            self.batch = saved_state['batch']
            self.bestacc = saved_state['bestacc']
        # GPU设定
        self.gpu = torch.cuda.is_available() and self.cfg.TRAIN.USE_GPU
        self.device = 'cuda' if self.gpu else 'cpu'
        if self.gpu:
            torch.cuda.set_device(self.cfg.TRAIN.GPU_ID[0])
            self.criterion.cuda()
            if len(self.cfg.TRAIN.GPU_ID) > 1:
                self.net = torch.nn.DataParallel(
                    self.net, device_ids=self.cfg.TRAIN.GPU_ID)
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()
            self.criterion.cpu()

    def train(self):
        for self.epoch in range(self.beginEpoch, self.cfg.TRAIN.MAX_EPOCH):
            # 训练一个Epoch
            self.t['train'].tic()
            self.trainEpoch()
            temp = self.t['train'].toc(average=False)
            print('Train time of Epoch {} is : {:.2f}s'.format(self.epoch, temp))
            # 在验证集上测试
            self.t['val'].tic()
            acc = self.val()
            temp = self.t['val'].toc(average=False)
            print('Val time of/after Epoch {} is : {:.2f}s'.format(self.epoch, temp))
            print('Acc for Epoch {} is : {:.4f}'.format(self.epoch, acc))
            print('BestAcc is:{:.4f}'.format(self.bestacc))
            self.writer.add_scalar('ValHitRate_PerEpoch', acc, self.epoch)
            # 保存模型
            if acc > self.bestacc:
                self.bestacc = acc
                temp = "HitRat{:.5f}".format(acc)
                self.save(temp)


    def trainEpoch(self):
        self.buffer = utils.ExBuffer(self.cfg.BUFFER_CAP)
        self.buffer.clean()
        print('Build Buffer.........')
        for batch_index, (clas, imgs, gts) in tqdm(enumerate(self.trainloader)):
            clas = clas.numpy()
            imgs = imgs.numpy()
            gts = gts.numpy()
            for j in range(len(imgs)):
                self.img = imgs[j]-self.meanImg
                self.cl = clas[j]
                self.gt = gts[j]
                if self.cl == 1:
                    self.initMarkX = [91.0, 71.0, 51.0, 31.0, 11.0]
                else:
                    self.initMarkX = [11.0, 31.0, 51.0, 71.0, 91.0]
                self.updateBuffer()
                if self.trainFlag:
                    self.trainBuffer()
                    print('Build Buffer.........')
        self.trainBuffer()

    def trainBuffer(self):
        print('Training..........')
        self.net.train()
        tf = transforms.ToTensor()
        dataset = datasets.bufferLoader(self.buffer.buffer,tf=tf)
        loader = DataLoader(dataset, num_workers=self.cfg.DATA.NUM_WORKS, batch_size=self.cfg.DATA.BS, shuffle=self.cfg.DATA.SHUFFLE)
        for epoch in tqdm(range(self.cfg.TRAIN.INER_EPOCH)):
            for fea, state, Q in loader:
                fea, state, Q = fea.to(self.device), state.to(self.device), Q.to(self.device)
                self.optimizer.zero_grad()
                output = self.net(fea, state)
                loss = self.criterion(output, Q)
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('trian_loss', loss.item(), self.batch)
                self.batch += 1

    def val(self):
        self.net.eval()
        hit_cnt = 0
        detect_hit_cnt = 0
        test_cnt = 0
        sup_cnt = 0
        steps_cnt = 0
        for valIndex, (cl, img, gt) in tqdm(enumerate(self.valloader)):
            img = np.squeeze(img.numpy())
            cl = np.squeeze(cl.numpy())
            gt = np.squeeze(gt.numpy())
            img = img - self.meanImg
            if cl == 1:
                initMarkX = [91.0, 71.0, 51.0, 31.0, 11.0]
            else:
                initMarkX = [11.0, 31.0, 51.0, 71.0, 91.0]
            # 循环处理五个landmark point
            xpoints = dict()
            for k in np.arange(self.cfg.LANDMARK_NUM, 0, -1):  # 5.4.3.2.1
                cur_x = []
                step = 0
                allActList = np.zeros(self.cfg.MAX_STEP)
                status = 1
                if gt[k - 1] == -1:
                    gt[k - 1] = -20
                gt_point = gt[k - 1]
                fea_t = np.array(img[(k - 1) * 20:k * 20,:,:])
                fea_t = np.transpose(fea_t, (2, 0, 1))
                fea_t = fea_t.astype(np.float32)
                fea_t = fea_t.reshape((1,fea_t.shape[0],fea_t.shape[1],fea_t.shape[2]))
                fea_t = torch.from_numpy(fea_t).cuda()
                cur_point = initMarkX[k - 1]
                cur_x.append(cur_point)
                if self.cfg.HIS_NUM == 0:
                    hist_vec = []
                else:
                    hist_vec = np.zeros([self.cfg.ACT_NUM * self.cfg.HIS_NUM])
                state = reward.get_state(cur_point, hist_vec)
                while (status == 1) & (step < self.cfg.MAX_STEP):
                    step += 1
                    state= state.astype(np.float32).reshape((1,-1))
                    state = torch.from_numpy(state).cuda()
                    qval = np.squeeze(self.net(fea_t, state).detach().cpu().numpy())
                    action = (np.argmax(qval)) + 1
                    allActList[step - 1] = action
                    if action != 4:
                        if action == 1:
                            cur_point = -20
                        elif action == 2:
                            cur_point -= 5
                        elif action == 3:
                            cur_point += 5
                        cur_x.append(cur_point)
                    else:
                        status = 0
                    if self.cfg.HIS_NUM != 0:
                        hist_vec = reward.update_history_vector(
                            hist_vec, action)
                    state = reward.get_state(cur_point, hist_vec)
                steps_cnt += step
                finalPoint = cur_point
                finalDist = abs(finalPoint - gt_point)
                det_dst = abs(initMarkX[k-1]-gt_point)
                if det_dst < self.cfg.DST_THR:
                    detect_hit_cnt += 1
                test_cnt += 1
                if finalDist <= self.cfg.DST_THR:
                    hit_cnt += 1
                if finalDist <= det_dst:
                    sup_cnt += 1
                xpoints[str(k-1)] = cur_x
        finImg = utils.visOneLane(img, self.meanImg, gt, initMarkX, xpoints)
        finImg = utils.catFinalImg(finImg)
        tempPath = osp.join(self.cfg.EXP.PATH,'valimg','val'+str(self.epoch)+'.png')
        Image.fromarray(finImg.astype('uint8')).save(tempPath)
        finImg=np.transpose(finImg, (2,0,1))
        self.writer.add_image('Val_Vis',finImg,self.epoch)
        self.writer.add_scalar('Val_RL_HR', float(hit_cnt) / test_cnt, self.epoch)
        self.writer.add_scalar('Val_Hit_Cnt',hit_cnt,self.epoch)
        self.writer.add_scalar('Val_Det_HR', float(detect_hit_cnt) / test_cnt, self.epoch)
        self.writer.add_scalar('Val_Det_Hit_Cnt',detect_hit_cnt,self.epoch)
        self.writer.add_scalar('Val_RLsupDet_HR', float(sup_cnt)/test_cnt, self.epoch)
        self.writer.add_scalar('Val_Average_Step', float(steps_cnt) / ((valIndex + 1) * 5), self.epoch)
        return float(hit_cnt) / test_cnt

    def updateBuffer(self):
        self.trainFlag = False
        buf = collections.namedtuple('buf', field_names=['fea', 'state', 'Q'])
        # generateExpReplay
        for k in np.arange(cfg.LANDMARK_NUM, 0, -1):  # [5,4,3,2,1]
            if self.gt[k - 1] == -1:
                self.gt[k - 1] = -20
            gt_point = self.gt[k - 1]
            # generate actions
            # status indicates whether the agent is still alive and has not triggered the terminal action
            status = 1
            step = 0
            cur_point = self.initMarkX[k - 1]
            landmark_fea = np.array(self.img[(k - 1) * 20:k * 20, :, :])
            landmark_fea_trans = np.reshape(landmark_fea, (1, 20, 100, 3))
            if self.cfg.HIS_NUM == 0:
                hist_vec = []
            else:
                hist_vec = np.zeros([self.cfg.HIS_NUM*self.cfg.ACT_NUM])
            state = reward.get_state(cur_point, hist_vec)
            cur_dst = reward.get_dst(gt_point, cur_point)
            last_point = cur_point
            last_dst = cur_dst
            while (status == 1) & (step < self.cfg.MAX_STEP):
                rew = []
                qval = np.array(self.predict(landmark_fea, state))
                step += 1
                # 挑选action 计算reward
                # we force terminal action in case actual IoU is higher than 0.5, to train faster the agent
                if cur_dst < self.cfg.DST_THR:
                    action = 4
                # epsilon-greedy policy
                elif random.random() < self.cfg.EPSILON:
                    action = np.random.randint(1, 5)
                else:
                    action = (np.argmax(qval)) + 1
                # terminal action
                if action == 4:
                    rew = reward.get_reward_trigger(cur_dst)
                # move action,performe the crop of the corresponding subregion
                elif action == 1:
                    cur_point = -20
                    cur_dst = reward.get_dst(gt_point, cur_point)
                    rew = reward.getRewRm(cur_dst)
                    last_dst = cur_dst
                    last_point = cur_point
                elif action == 2:  # to left
                    cur_point = cur_point - 5
                    cur_dst = reward.get_dst(gt_point, cur_point)
                    rew = reward.getRewMov0427(cur_point, last_point, gt_point)
                    last_dst = cur_dst
                    last_point = cur_point
                elif action == 3:  # to right
                    cur_point = cur_point + 5
                    cur_dst = reward.get_dst(gt_point, cur_point)
                    rew = reward.getRewMov0427(cur_point, last_point, gt_point)
                    last_dst = cur_dst
                    last_point = cur_point
                if self.cfg.HIS_NUM != 0:
                    hist_vec = reward.update_history_vector(hist_vec, action)
                new_state = reward.get_state(cur_point, hist_vec)
                # 计算 用来训练的Q值
                if action == 4:
                    temp = rew
                else:
                    temp = np.array(self.predict(landmark_fea, new_state))
                    temp = np.argmax(temp)
                    temp = rew + self.cfg.GAMMA * temp
                qval[action-1] = temp
                # 将数据存入buffer
                if self.buffer.ready2train:
                    self.trainFlag = True
                    break
                else:
                    temp = buf(landmark_fea, state, qval)
                    self.buffer.append(temp)
                if action == 4:
                    status = 0
                state = new_state

    def predict(self, fea, sta):
        self.net.eval()
        fea = np.transpose(fea, (2, 0, 1))
        fea = fea.astype(np.float32)
        fea = fea.reshape((1,fea.shape[0],fea.shape[1],fea.shape[2]))
        fea = torch.from_numpy(fea).cuda()
        sta = sta.astype(np.float32)
        sta = sta.reshape((1,-1))
        sta = torch.from_numpy(sta).cuda()

        x = self.net(fea, sta)
        return np.squeeze(x.data.detach().cpu().numpy())

    def save(self,temp):
        temp = 'EP_'+str(self.epoch)+'_'+temp+'.pth'
        path = osp.join(self.cfg.EXP.PATH, temp)
        if (not self.cfg.TRAIN.USE_GPU) or (len(self.cfg.TRAIN.GPU_ID) == 1):
            to_saved_weight = self.net.state_dict()
        else:
            to_saved_weight = self.net.module.state_dict()
        toSave = {
            'weights': to_saved_weight,
            'epoch': self.epoch,
            'batch': self.batch,
            'bestacc': self.bestacc
        }
        torch.save(toSave, path)
        print('Model Saved!')


if __name__ == "__main__":
    utils.setup_seed(cfg.SEED)
    MyTrainer = trainer(cfg)
    MyTrainer.train()
