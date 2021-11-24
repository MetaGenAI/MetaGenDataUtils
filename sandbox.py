
import numpy as np

import matplotlib.pyplot as plt

a = np.load("data\data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S_d03a6c7b.npy")
b = np.load("data\example_numpy_frames.npy")

plt.matshow(a[:30,0,:])
plt.matshow(b[:,:])
