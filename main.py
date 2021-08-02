# coding=utf-8

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

from util import *
import util.prep_dataset
from util.dataset import MotionDataset
from models.neural_network import *
import models.neural_network
from torch.utils.data import DataLoader


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    test_weights = np.load('params/original_files/dynamics_W1.npy')
    test_biases = np.load('params/original_files/dynamics_b1.npy')

    motion_dataset = MotionDataset(csv_file="data/all_runs.csv")

    motion_network = MotionNetwork()

    train_dataloader = DataLoader(dataset = motion_dataset, batch_size = 8, shuffle = True)

    optimizer = torch.optim.Adam(params = motion_network.parameters(), lr = 0.001)
    optimizer.zero_grad()

    criterion = nn.MSELoss()

    max_epochs = 100
    running_loss = 0.0
    for epoch in range(max_epochs):
        # Training
        for i, data in enumerate(train_dataloader, 0):
            inputs, gt = data
            # Transfer to GPU
            optimizer.zero_grad()

            outputs = motion_network(inputs)

            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # export weights
    l = motion_network.state_dict()
    w1_export = l['fc1.weight']
    w1_export = w1_export.numpy()
    w1_export = np.float64(w1_export)
    np.save('params/norlab_autorally_nn/dynamics_W1.npy', w1_export)
    b1_export = l['fc1.bias']
    b1_export = b1_export.numpy()
    b1_export = np.float64(b1_export)
    np.save('params/norlab_autorally_nn/dynamics_b1.npy', b1_export)

    w2_export = l['fc2.weight']
    w2_export = w2_export.numpy()
    w2_export = np.float64(w2_export)
    np.save('params/norlab_autorally_nn/dynamics_W2.npy', w2_export)
    b2_export = l['fc2.bias']
    b2_export = b2_export.numpy()
    b2_export = np.float64(b2_export)
    np.save('params/norlab_autorally_nn/dynamics_b2.npy', b2_export)

    w3_export = l['fc3.weight']
    w3_export = w3_export.numpy()
    w3_export = np.float64(w3_export)
    np.save('params/norlab_autorally_nn/dynamics_W3.npy', w3_export)
    b3_export = l['fc3.bias']
    b3_export = b3_export.numpy()
    b3_export = np.float64(b3_export)
    np.save('params/norlab_autorally_nn/dynamics_b3.npy', b3_export)

    print(w1_export.shape)
    print(b1_export.shape)

    np.savez('params/norlab_autorally_nn/norlab_autorally_nn_02_08_2021.npz', dynamics_W1=w1_export, dynamics_b1=b1_export,
             dynamics_W2=w2_export, dynamics_b2=b2_export,
             dynamics_W3=w3_export, dynamics_b3=b3_export)
