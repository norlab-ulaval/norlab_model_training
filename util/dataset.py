# coding=utf-8
import pandas as pd
import numpy as np
import torch
from util.util_func import *
from torch.utils.data import Dataset, DataLoader

class MotionDataset(Dataset):
    """Motion Dataset."""

    def __init__(self, csv_file):
        """
        :param csv_file:
        """
        self.motion_data_df = pd.read_csv(csv_file)
        self.motion_data_np = self.motion_data_df.to_numpy()

        len_data = self.motion_data_np.shape[0]
        icp_hits = int(self.motion_data_np[-1, 3])
        icp_poses = np.zeros((icp_hits, 6))
        icp_poses[0, :2] = self.motion_data_np[0, 10:12]
        icp_poses[0, 2:5] = quaternion_to_euler(self.motion_data_np[0, 15], self.motion_data_np[0, 12],
                                            self.motion_data_np[0, 13], self.motion_data_np[0, 14])[2]
        icp_poses[0, 5] = self.motion_data_np[0, 0]
        imu_ori = np.zeros(3)

        self.data_x = np.zeros((len_data, 6))  # input is [roll, vx, vy, yaw_rate, cmd_lin, cmd_ang]
        self.data_y = np.zeros((len_data, 4))  # output is [roll, vx, vy, yaw_rate at t2]
        _id = int(self.motion_data_np[0, 3]) # first icp index

        for i in range(1, self.motion_data_np.shape[0]):
            dt = self.motion_data_np[i, 1] - self.motion_data_np[i - 1, 1]
            imu_ori[:] = quaternion_to_euler(self.motion_data_np[i, 8], self.motion_data_np[i, 5],
                                             self.motion_data_np[i, 6], self.motion_data_np[i, 7])

            self.data_x[i, 0] = imu_ori[0] # Roll angle (from IMU)
            self.data_x[i, 3] = self.motion_data_np[i, 9] # Yaw rate (from IMU)

            if self.motion_data_np[i, 3] != _id:
                icp_poses[_id, :2] = self.motion_data_np[i, 10:12]
                icp_poses[_id, 2:5] = quaternion_to_euler(self.motion_data_np[i, 15], self.motion_data_np[i, 10],
                                                         self.motion_data_np[i, 11], self.motion_data_np[i, 12])[2]
                icp_poses[_id, 5] = self.motion_data_np[i, 0]

        dt = 0
        for i in range(1, self.motion_data_np.shape[0] - 1):
            # dt = dt + self.motion_data_np[i, 1] - self.motion_data_np[i - 1, 1]
            # if self.motion_data_np[i, 3] == _id: #if no new icp hit
            #     dt = icp_poses[_id+1, 0] - icp_poses[id - 1, 0]
            #     vx_map = (icp_poses[_id + 1, 0] - icp_poses[_id, 0]) / dt
            #     vy_map = (icp_poses[_id + 1, 1] - icp_poses[_id, 1]) / dt
            #     v_map = np.array([vx_map, vy_map, 0])
            #     body_to_map = rigid_tranformation(np.array([0, 0, icp_poses[_id - 1, 4]]))
            #     map_to_body = np.linalg.inv(body_to_map)
            #     v_body = np.matmul(map_to_body, v_map)  # apply map_to_body rigid transform for velocities
            #     self.data_x[i, 1] = v_body[0]  # body v_x
            #     self.data_x[i, 2] = v_body[1]  # body v_y

            if self.motion_data_np[i, 3] != _id: #if new icp hit --> compute velocity from previous icp pose
                dt = icp_poses[_id, 0] - icp_poses[_id - 1, 0]
                vx_map = (icp_poses[_id, 0] - icp_poses[_id - 1, 0]) / dt
                vy_map = (icp_poses[_id, 1] - icp_poses[_id - 1, 1]) / dt
                v_map = np.array([vx_map, vy_map, 0])
                body_to_map = rigid_tranformation(np.array([0, 0, icp_poses[_id - 1, 4]]))
                map_to_body = np.linalg.inv(body_to_map)
                v_body = np.matmul(map_to_body, v_map)  # apply map_to_body rigid transform for velocities
                self.data_x[i, 1] = v_body[0]  # body v_x
                self.data_x[i, 2] = v_body[1]  # body v_y

                _id = int(self.motion_data_np[i, 3])

            if _id >= len_data:
                break

            self.data_y[i - 1, 0] = self.data_x[i, 0]  # roll
            self.data_y[i - 1, 1] = self.data_x[i, 1]  # body v_x
            self.data_y[i - 1, 2] = self.data_x[i, 2]  # body v_y
            self.data_y[i - 1, 3] = self.data_x[i, 3]  # yaw rate

        # remove last row of both input and output array
        self.data_x = self.data_x[:-1]
        self.data_y = self.data_y[:-1]
        print(self.data_x[:,1])
        print(self.data_x[:,2])

        self.X = torch.from_numpy(self.data_x)
        self.X = self.X.float()
        self.y = torch.from_numpy(self.data_y)
        self.y = self.y.float()


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

