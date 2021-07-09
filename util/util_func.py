# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

print('test')


def quaternion_to_euler(w, x, y, z):
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x ** 2 + y ** 2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2*(w*y - z*x)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp)*np.pi/2,
                     np.arcsin(sinp))

    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y ** 2 + z ** 2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def wrap2pi(angle):
    if angle <= np.pi and angle >= -np.pi:
        return (angle)
    elif angle < -np.pi:
        return (wrap2pi(angle + 2*np.pi))
    else:
        return (wrap2pi(angle - 2*np.pi))


def rigid_tranformation(params):
    """Returns a rigid transformation matrix

    :params: numpy array, params[0]=tx, params[1]=ty, params[2]=theta
    :returns: LaTeX bmatrix as a string
    """
    return np.array([[np.cos(params[2]), -np.sin(params[2]), params[0]],
                     [np.sin(params[2]), np.cos(params[2]), params[1]],
                     [0, 0, 1]])
