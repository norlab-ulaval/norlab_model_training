# coding=utf-8
import torch
import numpy as np
from torch import nn

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('Using {} device'.format(device))

class MotionNetwork(nn.Module):
    def __init__(self):
        super(MotionNetwork, self).__init__()

        # Parameters
        self.inputSize = 8
        self.outputSize = 4
        self.hiddenSize = 32

        # First fully connected layer, taking 8 input channels (r, vx, vy, yaw_rate, cmd_left_t0,
        #                                                   cmd_right_t0, cmd_left_t0, cmd_right_t0)
        # outputting 32 features
        self.fc1 = nn.Linear(self.inputSize, self.hiddenSize)
        # Second fully connected (first hidden) layer that inputs / outputs 32 features
        self.fc2 = nn.Linear(self.hiddenSize, self.hiddenSize)
        # Third fully connected layer that inputs 32 features
        # outputting 4 next dynamic states (r, vx, vy, yaw_rate)
        self.fc3 = nn.Linear(self.hiddenSize, self.outputSize)

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.W2 = torch.randn(self.hiddenSize, self.hiddenSize)
        self.W3 = torch.randn(self.hiddenSize, self.outputSize)


    def forward(self, x):
        activation = nn.Tanh()
        # self.z = torch.matmul(x, self.W1) # multiply input by first weights
        # self.z2 = activation(self.z) # hyperbolic tangent nonlinearities
        # self.z3 = torch.matmul(self.z2, self.W2) # multiply by second weights
        # self.z4 = activation(self.z3)  # hyperbolic tangent nonlinearities
        # self.z5 = torch.matmul(self.z4, self.W3)  # multiply by second weights
        # out = activation(self.z5)
        self.z1 = self.fc1(x)
        self.z2 = activation(self.z1)
        self.z3 = self.fc2(self.z2)
        self.z4 = activation(self.z3)
        self.z5 = self.fc3(self.z4)
        out = activation(self.z5)
        return out

    # def backward(self, X, y, o): ## Investigate minibatch gradient descent
    #     self.o_error = y - 0 # output error
    #     self.o_delta = self.o_error * nn.Tanh(o) ## NEED to define backprop function
    #     self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
    #     self.z2_delta = self.z2_error * nn.Tanh(o) ## NEED to define backprop function        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
    #     self.z4_error = torch.matmul(self.o_delta, torch.t(self.W3))
    #     self.z4_delta = self.z4_error * nn.Tanh(o) ## NEED to define backprop function
    #     self.W1 += torch.matmul(torch.t(X), self.z2_delta)
    #     self.W2 += torch.matmul(torch.t(self.z2), self.z4_delta)
    #     self.W3 += torch.matmul(torch.t(self.z4), self.o_delta)
    #
    # def train(self, X, y):
    #     o = self.forward(X)
    #     self.backward(X, y, o)
    #
    # def saveWeights(self, model):
    #     torch.save(model, 'motionNN')
    #
    # def predict(self, X):
    #     o = self.forward(X)
    #     return o

def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss =criterion(output,y)
    loss.backward()
    optimizer.step()

    return loss, output