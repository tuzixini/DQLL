import numpy as np
import cv2
import pdb
from config import cfg


dst_threshold = cfg.DST_THR
reward_terminal_action = cfg.reward_terminal_action
reward_movement_action = cfg.reward_movement_action
reward_invalid_movement_action = cfg.reward_invalid_movement_action
reward_remove_action = cfg.reward_remove_action
# Different actions that the agent can do
num_of_actions = cfg.ACT_NUM
# Actions captures in the history vector
num_of_history = cfg.HIS_NUM


def get_dst(gt_point, cur_point):
    dst = abs(gt_point-cur_point)
    return dst


def get_reward_trigger(cur_dst):
    if cur_dst < dst_threshold:
        reward = reward_terminal_action
    else:
        reward = - reward_terminal_action
    return reward


def get_reward_movement(cur_point, last_point, gt_point):
    if gt_point == -100:
        if cur_point == -100:
            reward = reward_movement_action
        else:
            reward = - reward_movement_action
    else:
        cur_dst = get_dst(gt_point, cur_point)
        last_dst = get_dst(gt_point, last_point)
        if cur_dst < last_dst:
            reward = reward_movement_action
        else:
            reward = - reward_movement_action
    return reward


def getRewMov0427(cur_point, last_point, gt_point):
    if gt_point == -20:  # should be removed, but the action is 2 or 3
        reward = - reward_remove_action
    else:  # should be moved, the action is 2 or 3
        cur_dst = get_dst(gt_point, cur_point)
        last_dst = get_dst(gt_point, last_point)
        if cur_dst < last_dst:
            reward = reward_movement_action
        else:
            if cur_point < 0 or cur_point >= 100:  # moved out of the image, the reward is change to -5
                reward = reward_invalid_movement_action
            else:
                reward = - reward_movement_action
    return reward


def getRewRm(cur_dst):
    if cur_dst == 0:
        reward = reward_remove_action
    else:
        reward = - reward_remove_action
    return reward


def update_history_vector(history_vector, action):
    action_vector = np.zeros(num_of_actions)
    action_vector[action-1] = 1
    # number of real history in the current history vector
    num_real_cur_history = np.size(np.nonzero(history_vector))
    updated_history_vector = np.zeros(num_of_actions*num_of_history)
    if num_real_cur_history < num_of_history:
        aux2 = 0
        for l in range(num_of_actions*num_real_cur_history, num_of_actions*num_real_cur_history+num_of_actions):
            history_vector[l] = action_vector[aux2]
            aux2 += 1
        return history_vector
    else:
        for j in range(0, num_of_actions*(num_of_history-1)):
            updated_history_vector[j] = history_vector[j+num_of_actions]
        aux = 0
        for k in range(num_of_actions*(num_of_history-1), num_of_actions*num_of_history):
            updated_history_vector[k] = action_vector[aux]
            aux += 1
        return updated_history_vector


def get_state(cur_point, hist_vec):
    history_vector = np.reshape(hist_vec, (num_of_actions*num_of_history, 1))
    state = np.vstack((cur_point, history_vector))
    state = np.squeeze(state)
    return state
