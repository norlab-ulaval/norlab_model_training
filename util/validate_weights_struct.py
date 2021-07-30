import numpy as np

GT_weights = np.load('../params/original_files/autorally_nnet_09_12_2018.npz')
norlab_weights = np.load('../params/norlab_autorally_nn/norlab_autorally_nn.npz')

np.savez('../params/norlab_autorally_nn/norlab_autorally_nn.npz', dynamics_b1=norlab_weights['dynamics_b1'],
         dynamics_b2=norlab_weights['dynamics_b2'],
         dynamics_b3=norlab_weights['dynamics_b3'],
         dynamics_W1=norlab_weights['dynamics_W1'],
         dynamics_W2=norlab_weights['dynamics_W2'],
         dynamics_W3=norlab_weights['dynamics_W3'])

print(GT_weights.files)
print(norlab_weights.files)