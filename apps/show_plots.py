import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

plt.ion()

data_files = list(glob.glob(sys.argv[1]+'/mnist_net_*_train.log'))
valid_data_files = list(glob.glob(sys.argv[1]+'/mnist_net_*_valid.log'))

for fname in data_files:
  data = np.loadtxt(fname).reshape(-1, 3)
  name = fname.split('/')[-1]
  plt.plot(data[:, 0], 1-data[:, 2], label=name)

for fname in valid_data_files:
  data = np.loadtxt(fname).reshape(-1, 2)
  name = fname.split('/')[-1]
  plt.plot(data[:, 0], 1-data[:, 1], label=name)

plt.legend(loc=1)

raw_input('Press Enter.')
