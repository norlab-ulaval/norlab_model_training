# coding=utf-8
import pandas as pd
import numpy as np
import torch
from util.util_func import *

# load one dataset
ls_CCW = pd.read_csv("data/low_speed_CCW_data_raw.csv")
ls_CCW = ls_CCW.to_numpy() # [ros_time, joy_switch, icp_index, cmd_left_vel, meas_left_vel,
                            # cmd_right_vel, meas_right_vel, icp_pos_x, icp_pos_y,
#                             icp_quat_x, icp_quat_y, icp_quat_z, icp_quat_w]

euler_angles = np.zeros((ls_CCW.shape[0], 3))

len_data = 4 #length of data vectors to feed to network
data_x = np.zeros((len_data, 8)) # input is [roll, vx, vy, yaw_rate, u_left_t0, u_right_t0, u_left_t1, u_right_t1]
data_y = np.zeros((len_data, 3)) # output is [x, y, yaw]
_id = 0

for i in range(1, ls_CCW.shape[0]):
    euler_angles[i, :] = quaternion_to_euler(ls_CCW[i, 13], ls_CCW[i, 10], ls_CCW[i, 11], ls_CCW[i, 12])[2]
    dt = ls_CCW[i, 1] - ls_CCW[i-1, 1]
    if ls_CCW[i, 3] != _id:
        data_x[_id, 0] = euler_angles[i, 0]
        data_x[_id, 1] = (ls_CCW[i, 8] - ls_CCW[i - 1, 8])/dt # vx # need to do the inverse transform from map to body
        data_x[_id, 2] = (ls_CCW[i, 9] - ls_CCW[i - 1, 9])/dt # vy
        data_x[_id, 3] = (euler_angles[i, 2] - euler_angles[i - 1, 2])/dt # yaw rate
        data_x[_id, 4] = ls_CCW[i - 1, 4] # u_left_t0
        data_x[_id, 5] = ls_CCW[i - 1, 6] # u_right_t0
        data_x[_id, 6] = ls_CCW[i, 4] # u_left_t1
        data_x[_id, 7] = ls_CCW[i, 6] # u_right_t1

        data_y[_id, 0] = ls_CCW[i, 8]
        data_y[_id, 1] = ls_CCW[i, 9]
        data_y[_id, 2] = euler_angles[i, 2]
        _id = int(ls_CCW[i, 3])
    if _id >= len_data:
        break

X = torch.tensor(data_x)
y = torch.tensor(data_y)