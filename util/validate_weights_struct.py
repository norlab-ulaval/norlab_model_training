import numpy as np

GT_weights = np.load('../params/original_files/autorally_nnet_09_12_2018.npz')
norlab_weights = np.load('../params/norlab_autorally_nn/norlab_autorally_nn_02_08_2021.npz')

w1 = np.float64(norlab_weights['dynamics_W1'])
w2 = np.float64(norlab_weights['dynamics_W2'])
w3 = np.float64(norlab_weights['dynamics_W3'])
b1 = np.float64(norlab_weights['dynamics_b1'])
b2 = np.float64(norlab_weights['dynamics_b2'])
b3 = np.float64(norlab_weights['dynamics_b3'])

np.savez('../params/norlab_autorally_nn/norlab_autorally_nn_02_08_2021.npz', dynamics_b1=b1, dynamics_b2=w2,
         dynamics_b3=b3, dynamics_W1=w1, dynamics_W2=w2, dynamics_W3=w3)

print(GT_weights['dynamics_W1'].dtype)
print(norlab_weights['dynamics_W1'].dtype)
print(w1.dtype)