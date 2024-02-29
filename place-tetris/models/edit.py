import numpy as np

file ='./place-tetris/models/models_12x8.npy'
data = np.load(file, allow_pickle=True)
data[:,1] -= 8
np.save(file, data)