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
        icp_poses = np.zeros((icp_hits, 5))
        icp_poses[0, :2] = self.motion_data_np[0, 7:9]
        icp_poses[0, 2:] = quaternion_to_euler(self.motion_data_np[0, 13], self.motion_data_np[0, 10],
                                            self.motion_data_np[0, 11], self.motion_data_np[0, 12])[2]

        self.data_x = np.zeros((icp_hits, 8))  # input is [roll, vx, vy, yaw_rate, u_left_t0, u_right_t0, u_left_t1, u_right_t1]
        self.data_y = np.zeros((icp_hits, 4))  # output is [roll, vx, vy, yaw_rate at t2]
        id = 1

        for i in range(1, self.motion_data_np.shape[0]):
            # euler_angles[i, :] = quaternion_to_euler(self.motion_data_np[i, 13], self.motion_data_np[i, 10],
            #                                          self.motion_data_np[i, 11], self.motion_data_np[i, 12])[2]
            dt = self.motion_data_np[i, 1] - self.motion_data_np[i - 1, 1]
            if self.motion_data_np[i, 3] != id:
                icp_poses[id, :2] = self.motion_data_np[i, 7:9]
                icp_poses[id, 2:] = quaternion_to_euler(self.motion_data_np[i, 13], self.motion_data_np[i, 10],
                                                       self.motion_data_np[i, 11], self.motion_data_np[i, 12])[2]
                self.data_x[id, 0] = icp_poses[id, 2] # roll
                # we compute velocity
                vx_map = (icp_poses[id, 0] - icp_poses[id - 1, 0]) / dt
                vy_map = (icp_poses[id, 1] - icp_poses[id - 1, 1]) / dt
                v_map = np.array([vx_map, vy_map, 0])
                body_to_map = rigid_tranformation(np.array([0, 0, icp_poses[id - 1, 4]]))
                map_to_body = np.linalg.inv(body_to_map)
                v_body = np.matmul(map_to_body, v_map) # apply map_to_body rigid transform for velocities
                self.data_x[id, 1] = v_body[0] # body v_x
                self.data_x[id, 2] = v_body[1] # body v_y
                self.data_x[id, 3] = (icp_poses[id, 4] - icp_poses[id - 1, 4]) / dt  # yaw rate, no need for transform
                self.data_x[id, 4] = self.motion_data_np[i - 1, 4]  # u_left_t0
                self.data_x[id, 5] = self.motion_data_np[i - 1, 6]  # u_right_t0
                self.data_x[id, 6] = self.motion_data_np[i, 4]  # u_left_t1
                self.data_x[id, 7] = self.motion_data_np[i, 6]  # u_right_t1

                # The output ground truth is the same as input, but offset by 1 icp hit
                self.data_y[id-1, 0] = self.data_x[id, 0] # roll
                self.data_y[id-1, 1] = self.data_x[id, 1]  # body v_x
                self.data_y[id-1, 2] = self.data_x[id, 2] # body v_y
                self.data_y[id-1, 3] = self.data_x[id, 3] # yaw rate

                id = int(self.motion_data_np[i, 3])
            if id >= len_data:
                break

        # remove last row of both input and output array
        self.data_x = self.data_x[:-1]
        self.data_y = self.data_y[:-1]

        self.X = torch.from_numpy(self.data_x)
        self.X = self.X.float()
        self.y = torch.from_numpy(self.data_y)
        self.y = self.y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

