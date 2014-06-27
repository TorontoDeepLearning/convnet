import h5py
import numpy as np
import sys
import os

pred_file = sys.argv[1]
print os.path.splitext(pred_file)
if os.path.splitext(pred_file)[1] == '.txt':
  predictions = np.loadtxt(pred_file)
else:
  predictions = h5py.File(pred_file, 'r')['output'].value
class_names = []
for line in open('class_names_CLS.txt', 'r'):
  class_names.append(line.strip())
K = 10
for i in xrange(predictions.shape[0]):
  print '----------------'
  p = predictions[i, :]
  p_sorted = (-p).argsort()
  for label in p_sorted[:K]:
    print class_names[label], p[label]


