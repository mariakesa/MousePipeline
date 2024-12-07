path="/home/maria/Documents/CarsenMariusData/6845348/natimg2800_M161025_MP030_2017-05-29.mat"

import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat(path)
print(mat.keys())

#print(mat['stat'])
print(mat['db'])
print(mat['stim'][0].shape)
print(mat['stim'][0][0][0].shape)