import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from util import *
from models.neural_network import *
from models.diff_drive import *
import models.neural_network

# init ideal diff-drive model
k = np.array([0.3, 1.296])
diff_drive = Diff_drive(k)

# init trained neural network
motion_network = MotionNetwork()
w1 = np.load('../params/original_files/dynamics_W1.npy')
motion_network.fc1.weight = torch.nn.Parameter(torch.from_numpy(w1))
b1 = np.load('../params/original_files/dynamics_b1.npy')
motion_network.fc1.bias = torch.nn.Parameter(torch.from_numpy(b1))

w2 = np.load('../params/original_files/dynamics_W2.npy')
motion_network.fc2.weight = torch.nn.Parameter(torch.from_numpy(w2))
b2 = np.load('../params/original_files/dynamics_b2.npy')
motion_network.fc2.bias = torch.nn.Parameter(torch.from_numpy(b2))

w3 = np.load('../params/original_files/dynamics_W3.npy')
motion_network.fc3.weight = torch.nn.Parameter(torch.from_numpy(w3))
b3 = np.load('../params/original_files/dynamics_b3.npy')
motion_network.fc3.bias = torch.nn.Parameter(torch.from_numpy(b3))

# w1 = np.load('../params/norlab_autorally_nn/dynamics_W1.npy')
# motion_network.fc1.weight = torch.nn.Parameter(torch.from_numpy(w1))
# b1 = np.load('../params/norlab_autorally_nn/dynamics_b1.npy')
# motion_network.fc1.bias = torch.nn.Parameter(torch.from_numpy(b1))
#
# w2 = np.load('../params/norlab_autorally_nn/dynamics_W2.npy')
# motion_network.fc2.weight = torch.nn.Parameter(torch.from_numpy(w2))
# b2 = np.load('../params/norlab_autorally_nn/dynamics_b2.npy')
# motion_network.fc2.bias = torch.nn.Parameter(torch.from_numpy(b2))
#
# w3 = np.load('../params/norlab_autorally_nn/dynamics_W3.npy')
# motion_network.fc3.weight = torch.nn.Parameter(torch.from_numpy(w3))
# b3 = np.load('../params/norlab_autorally_nn/dynamics_b3.npy')
# motion_network.fc3.bias = torch.nn.Parameter(torch.from_numpy(b3))

# load dataset

data = pd.read_csv('../data/validation_data_raw.csv')
data = data.to_numpy()
data = np.delete(data, 0, 0)
data[:, 1] = data[:, 1] / 10**9
data[:, 1] = data[:, 1] - data[0, 1]

time = data[:, 1]

icp_idx = data[:, 3]
icp_idx = icp_idx.astype('int32')

icp_poses = np.zeros((icp_idx[-1], 6))
icp_poses[0, 5] = time[0]
icp_poses[0, :2] = data[0, 11:13]
icp_poses[0, 2:5] = quaternion_to_euler(data[0, 16], data[0, 13],
                                        data[0, 14], data[0, 15])

for i in range(1, data.shape[0]-1):
    if icp_idx[i] != icp_idx[i-1]:
        icp_poses[icp_idx[i-1], :2] = data[i, 11:13]
        icp_poses[icp_idx[i-1], 2:5] = quaternion_to_euler(data[i, 16], data[i, 13],
                                                data[i, 14], data[i, 15])
        icp_poses[icp_idx[i-1], 5] = time[i]

icp_poses[-1, 5] = time[-1]
icp_poses[icp_idx[-3], :2] = data[-1, 11:13]
icp_poses[icp_idx[-3], 2:5] = quaternion_to_euler(data[-1, 16], data[-1, 13],
                                        data[-1, 14], data[-1, 15])

cmds = np.zeros((data.shape[0], 3))
cmds[:, 0] = data[:, 4]
cmds[:, 2] = data[:, 5]

imu_yaw_rate = data[:, 10]

# compute icp-estimated velocities

icp_vels = np.zeros((icp_idx[-1], 2))
for i in range(1, icp_idx[-1]):
    dt = icp_poses[i, 5] - icp_poses[i - 1, 5]
    vx_map = (icp_poses[i, 0] - icp_poses[i - 1, 0]) / dt
    vy_map = (icp_poses[i, 1] - icp_poses[i - 1, 1]) / dt
    v_map = np.array([vx_map, vy_map, 0])
    body_to_map = rigid_tranformation(np.array([0, 0, icp_poses[i - 1, 4]]))
    map_to_body = np.linalg.inv(body_to_map)
    v_body = np.matmul(map_to_body, v_map)
    icp_vels[i, 0] = v_body[0]
    icp_vels[i, 1] = v_body[1]

# compute prediction error over entire trajectory
icp_kin_pose = np.zeros(3)
icp_kin_pose[:2] = icp_poses[0, :2]
icp_kin_pose[2] = icp_poses[0, 4]

cmd_kin_pose = np.zeros(3)
cmd_kin_pose[:2] = icp_poses[0, :2]
cmd_kin_pose[2] = icp_poses[0, 4]
cmd_body_to_world = rigid_tranformation(np.array([0, 0, icp_poses[i, 4]]))
cmd_err = np.zeros(icp_idx[-1])

nn_kin_pose = np.zeros(3)
nn_kin_pose[:2] = icp_poses[0, :2]
nn_kin_pose[2] = icp_poses[0, 4]
nn_dyn_pose = np.array([icp_poses[0, 2], icp_vels[0, 0], icp_vels[0, 1], imu_yaw_rate[0]])
nn_body_to_world = rigid_tranformation(np.array([0, 0, icp_poses[i, 4]]))
nn_err = np.zeros(icp_idx[-1])

for i in range(data.shape[0] - 1):
    dt = time[i+1] - time[i]

    cmd_kin_pose = cmd_kin_pose + cmd_body_to_world @ cmds[i] * dt
    cmd_kin_pose[2] = wrap2pi(cmd_kin_pose[2])

    X = np.hstack((nn_dyn_pose, cmds[i, 0]))
    X = np.hstack((X, cmds[i, 2]))
    nn_kin_pose, nn_dyn_pose = motion_network.predict(torch.from_numpy(X),
                                                nn_kin_pose, dt)
    nn_kin_pose[2] = wrap2pi(nn_kin_pose[2])

    if icp_idx[i+1] != icp_idx[i]:
        icp_kin_pose[:2] = icp_poses[icp_idx[i+1]-1, :2]
        icp_kin_pose[2] = icp_poses[icp_idx[i+1]-1, 4]
        cmd_err[icp_idx[i]] = comp_disp(cmd_kin_pose, icp_kin_pose)
        nn_err[icp_idx[i]] = comp_disp(nn_kin_pose, icp_kin_pose)

        cmd_kin_pose[:2] = icp_poses[icp_idx[i+1]-1, :2]
        cmd_kin_pose[2] = icp_poses[icp_idx[i+1]-1, 4]
        nn_kin_pose[:2] = icp_poses[icp_idx[i+1]-1, :2]
        nn_kin_pose[2] = icp_poses[icp_idx[i+1]-1, 4]
        nn_dyn_pose[0] = icp_poses[icp_idx[i+1]-1, 2]
        nn_dyn_pose[1] = icp_vels[icp_idx[i+1]-1, 0]
        nn_dyn_pose[2] = icp_vels[icp_idx[i+1]-1, 1]
        nn_dyn_pose[3] = imu_yaw_rate[i+1]

plt.scatter(range(icp_idx[-1]), cmd_err, s=5, label='Ideal diff-drive')
plt.ylim(0, 2)
plt.scatter(range(icp_idx[-1]), nn_err, s=5, label='Autorally NN')
plt.ylabel('Model prediction error')
plt.xlabel('Icp index')
plt.legend()
plt.title('Model prediction error on validation dataset (doughnuts)')
plt.savefig('../figs/model_prediction_error.png')
plt.show()