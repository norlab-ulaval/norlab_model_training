# coding=utf-8
import torch

from util.dataset import *
from models.neural_network import *

motion_dataset = MotionDataset(csv_file="data/low_speed_CCW_data_raw.csv")

class TestDataset:

    def test_dataset_output_is_offset_from_input(self):
        motion_dataset = MotionDataset(csv_file="data/low_speed_CCW_data_raw.csv")
        assert motion_dataset.__getitem__(401)[0][0] == motion_dataset.__getitem__(400)[1][0]
        assert motion_dataset.__getitem__(401)[0][1] == motion_dataset.__getitem__(400)[1][1]
        assert motion_dataset.__getitem__(401)[0][2] == motion_dataset.__getitem__(400)[1][2]
        assert motion_dataset.__getitem__(401)[0][3] == motion_dataset.__getitem__(400)[1][3]

    def test_dataset_input_size_is_8(self):
        assert motion_dataset.__getitem__(400)[0].shape[0] == 8

    def test_dataset_output_size_is_4(self):
        assert motion_dataset.__getitem__(400)[1].shape[0] == 4

motion_network = MotionNetwork()

class TestNetwork:

    def test_motion_prediction_output_is_4(self):
        pred = motion_network(torch.zeros(8))
        assert pred.shape[0] == 4