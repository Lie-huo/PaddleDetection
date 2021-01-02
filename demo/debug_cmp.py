import numpy as np

pd_x = np.load('2021_debug_x_128.npy')

torch_x = np.load('/Users/liuhui29/Downloads/npy_2021/2021_debug_x_128.npy')

print(pd_x.shape, pd_x.sum(), pd_x.mean(), torch_x.shape, torch_x.sum(), torch_x.mean())
print(np.sum(np.abs(torch_x - pd_x)))
print(np.mean(np.abs(torch_x - pd_x)))