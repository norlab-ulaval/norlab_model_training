# coding=utf-8
import torch
import numpy as np
from torch import nn
from util.util_func import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('Using {} device'.format(device))

class MotionNetwork(nn.Module):
    def __init__(self):
        super(MotionNetwork, self).__init__()

        # Parameters
        self.inputSize = 6
        self.outputSize = 4
        self.hiddenSize = 32

        # First fully connected layer, taking 6 input channels (r, vx, vy, yaw_rate, cmd_lin, cmd_ang)
        # outputting 32 features
        self.fc1 = nn.Linear(self.inputSize, self.hiddenSize).double()
        # Second fully connected (first hidden) layer that inputs / outputs 32 features
        self.fc2 = nn.Linear(self.hiddenSize, self.hiddenSize).double()
        # Third fully connected layer that inputs 32 features
        # outputting 4 next dynamic states (r, vx, vy, yaw_rate)
        self.fc3 = nn.Linear(self.hiddenSize, self.outputSize).double()

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

    def predict(self, X, curr_kinematic_pose, dt):
        """
        A method that predicts the next kinematic state based on current state and input
        :param X: input vector (dynamic state and command)
        :param curr_kinematic_pose: current kinematic pose [x, y, yaw] in world frame
        :param dt: time step
        :return:
        """
        with torch.no_grad():
            nn_output = self.forward(X) # predict next dynamic state
            next_dynamic_states = nn_output.detach()
            next_dynamic_states = next_dynamic_states.numpy()

            # compute and apply transform from body to world frame
            body_to_world = rigid_tranformation(np.array([0, 0, curr_kinematic_pose[2]]))
            body_vel = np.array([next_dynamic_states[1], next_dynamic_states[2], next_dynamic_states[3]])
            world_vel = body_to_world @ body_vel

            # apply simple unicycle model to compute next body frame pose
            next_kinematic_states = np.array([curr_kinematic_pose[0] + world_vel[0] * dt,
                                               curr_kinematic_pose[1] + world_vel[1] * dt,
                                               curr_kinematic_pose[2] + world_vel[2] * dt])
        return next_kinematic_states, next_dynamic_states