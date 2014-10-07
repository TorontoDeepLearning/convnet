import numpy as np
import cudamat as cm
from cudamat import cudamat_conv as cc
from cudamat import gpu_lock2 as gpu_lock
import sys
import h5py
from time import sleep
from google.protobuf import text_format
import convnet_config_pb2

"""
This script uses the GPU locking system used at University of Toronto.
Please modify this accordingly for your GPU resource environment.
"""

def LockGPU(max_retries=10):
  """ Locks a free GPU board and returns its id. """
  for retry_count in range(max_retries):
    board = gpu_lock.obtain_lock_id()
    if board != -1:
      break
    sleep(1)
  if board == -1:
    print 'No GPU board available.'
    sys.exit(1)
  else:
    cm.cuda_set_device(board)
    cm.cublas_init()
  return board

def FreeGPU(board):
  """ Frees the board. """
  cm.cublas_shutdown()
  gpu_lock.free_lock(board)
