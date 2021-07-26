# coding=utf-8

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from util import *
import util.prep_dataset
from util.dataset import MotionDataset
from models.neural_network import *
import models.neural_network
from torch.utils.data import DataLoader


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    motion_dataset = MotionDataset(csv_file="data/all_runs.csv")

    motion_network = MotionNetwork()

    train_dataloader = DataLoader(dataset = motion_dataset, batch_size = 8, shuffle = True)

    optimizer = torch.optim.RMSprop(params = motion_network.parameters(), lr = 0.001)
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