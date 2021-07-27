import pandas as pd
import numpy as np

# import all data

low_speed_CCW = pd.read_csv("../data/low_speed_CCW_data_raw.csv")
low_speed_CCW = low_speed_CCW.to_numpy()
low_speed_CW = pd.read_csv("../data/low_speed_CW_data_raw.csv")
low_speed_CW = low_speed_CW.to_numpy()
zigzag_CCW = pd.read_csv("../data/zigzag_CCW_data_raw.csv")
zigzag_CCW = zigzag_CCW.to_numpy()
zigzag_CW = pd.read_csv("../data/zigzag_CW_data_raw.csv")
zigzag_CW = zigzag_CW.to_numpy()
acceleration_CCW = pd.read_csv("../data/acceleration_CCW_data_raw.csv")
acceleration_CCW = acceleration_CCW.to_numpy()
acceleration_CW = pd.read_csv("../data/acceleration_CW_data_raw.csv")
acceleration_CW = acceleration_CW.to_numpy()
high_speed_CCW = pd.read_csv("../data/high_speed_CCW_data_raw.csv")
high_speed_CCW = high_speed_CCW.to_numpy()
high_speed_CW = pd.read_csv("../data/high_speed_CW_data_raw.csv")
high_speed_CW = high_speed_CW.to_numpy()
drift_CCW = pd.read_csv("../data/drift_CCW_data_raw.csv")
drift_CCW = drift_CCW.to_numpy()
drift_CW = pd.read_csv("../data/drift_CW_data_raw.csv")
drift_CW = drift_CW.to_numpy()

# create run lits

run_list = [low_speed_CCW, low_speed_CW, zigzag_CCW, zigzag_CW, acceleration_CCW, acceleration_CW,
            high_speed_CCW, high_speed_CW, drift_CCW, drift_CW]

# set time of all arrays to start at 0, convert time in seconds / same for icp index
for run in run_list:
    run[:, 1] = run[:, 1] - run[0, 1]
    run[:,1] = run[:, 1] / 10**9
    run[:, 3] = run[:, 3] - run[0, 3]

# set the time of all runs to the end of the previous run / same for icp index
for i in range(1, len(run_list)):
    run_list[i][:, 1] = run_list[i][:, 1] + run_list[i-1][-1, 1] + 0.05
    run_list[i][:, 3] = run_list[i][:, 3] + run_list[i-1][-1, 3] + 1

# concatenate the entire dataset

all_runs = np.copy(low_speed_CCW)

for i in range(1, len(run_list)):
    all_runs = np.vstack((all_runs, run_list[i]))

# delete first column of array

all_runs = np.delete(all_runs, 0, 1)

# re-convert all data to pandas dataframe

df = pd.DataFrame(data=all_runs, columns=['ros_time', 'joy_switch', 'icp_index',
                                               'cmd_lin', 'cmd_ang',
                                               'imu_quat_x', 'imu_quat_y',
                                               'imu_quat_z', 'imu_quat_w',
                                               'imu_yaw_rate',
                                               'icp_pos_x', 'icp_pos_y',
                                               'icp_quat_x', 'icp_quat_y',
                                               'icp_quat_z', 'icp_quat_w'])
df.to_csv('../data/all_runs.csv')