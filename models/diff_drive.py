import numpy as np
from util.util_func import *

class Diff_drive():
    """
    ideal differential class
    :k: numpy array, k[0]=r, k[1]=b
    """
    def __init__(self, k):
        """
        :param k:
        """

        self.r = k[0]
        self.b = k[1]
        self.J = self.r * np.array([[0.5, 0.5], [0.0, 0.0], [-1 / self.r, 1 / self.r]])

    def pred_kinematic(self, u, curr_kinematic_pose, dt):
        """
        A method that predicts the next kinematic state based on current state and input
        :param u: input vector (command)
        :param curr_kinematic_pose: current kinematic pose [x, y, yaw] in world frame
        :param dt: time step
        :return:
        """

        x_dot = self.J @ u
        body_to_world = rigid_tranformation(np.array([0, 0, curr_kinematic_pose[2]]))
        return curr_kinematic_pose + body_to_world @ x_dot * dt