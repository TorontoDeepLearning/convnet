import ctypes as ct
import math
import pdb
_ConvNet = ct.cdll.LoadLibrary('libcudamat_conv_gemm.so')

def DivUp(a, b):
  return (a + b - 1) / b

def AddAtAllLocs(h, b):
  batch_size, size_x, size_y, num_channels = h.shape4d
  b_shape = b.shape
  h.reshape((-1, num_channels))
  b.reshape((1, -1))
  assert b.shape[1] == num_channels
  h.add_row_vec(b)
  h.reshape((batch_size, -1))
  b.reshape(b_shape)

def AddAtAllLocs3D(h, b):
  batch_size, size_x, size_y, num_channels_mult_size_t = h.shape4d
  num_channels = b.shape[1]
  size_t = num_channels_mult_size_t / num_channels
  assert size_t * num_channels == num_channels_mult_size_t
  b_shape = b.shape
  h.reshape((-1, num_channels_mult_size_t))
  b.reshape((1, -1))
  for i in xrange(size_t):
    h.slice(i*num_channels, (i+1)*num_channels).add_row_vec(b)
  h.reshape((batch_size, -1))
  b.reshape(b_shape)

def AddUpAllLocs(h, b, scaleTargets=0):
  batch_size, size_x, size_y, num_channels = h.shape4d
  b_shape = b.shape
  h.reshape((-1, num_channels))
  b.reshape((1, -1))
  assert b.shape[1] == num_channels
  if scaleTargets == 0:
    h.sum(axis=0, target=b)
  else:
    b.mult(scaleTargets)
    b.add_sums(h, axis=0)
  h.reshape((batch_size, -1))
  b.reshape(b_shape)

def AddUpAllLocs3D(h, b, scaleTargets=0):
  batch_size, size_x, size_y, num_channels_mult_size_t = h.shape4d
  num_channels = b.shape[1]
  size_t = num_channels_mult_size_t / num_channels
  assert size_t * num_channels == num_channels_mult_size_t
  b_shape = b.shape
  h.reshape((-1, num_channels_mult_size_t))
  b.reshape((1, -1))
  
  b.mult(scaleTargets)
  for i in xrange(size_t):
    b.add_sums(h.slice(i*num_channels, (i+1)*num_channels), axis=0)
  h.reshape((batch_size, -1))
  b.reshape(b_shape)

def convUp(images, filters, targets, conv_desc, scaleTargets=0):
  _ConvNet.convUpGemm(images.p_mat, filters.p_mat, targets.p_mat,
                  images.p_shape4d, filters.p_shape4d, targets.p_shape4d,
                  conv_desc, ct.c_float(scaleTargets))

def convDown(hidSums, filters, targets, conv_desc, scaleTargets=0):
  _ConvNet.convDownGemm(hidSums.p_mat, filters.p_mat, targets.p_mat,
                    hidSums.p_shape4d, filters.p_shape4d, targets.p_shape4d,
                    conv_desc, ct.c_float(scaleTargets))

def convOutp(images, hidSums, targets, conv_desc, scaleTargets=0):
  _ConvNet.convOutpGemm(
    images.p_mat, hidSums.p_mat, targets.p_mat,
    images.p_shape4d, hidSums.p_shape4d, targets.p_shape4d,
    conv_desc, ct.c_float(scaleTargets), ct.c_float(1))

def MaxPool(images, targets, conv_desc):
  _ConvNet.MaxPoolGemm(images.p_mat, targets.p_mat, images.p_shape4d,
                       targets.p_shape4d, conv_desc, ct.c_float(0.0),
                       ct.c_float(1.0))

def MaxPoolUndo(images, grad, maxes, targets, conv_desc, scaleTargets=0):
  _ConvNet.MaxPoolUndoGemm(images.p_mat, grad.p_mat, maxes.p_mat, targets.p_mat,
                       images.p_shape4d, grad.p_shape4d, conv_desc,
                       ct.c_float(scaleTargets))

def AvgPool(images, targets, conv_desc):
  _ConvNet.AvgPoolGemm(images.p_mat, targets.p_mat, images.p_shape4d,
                       targets.p_shape4d, conv_desc, ct.c_float(0.0),
                       ct.c_float(1.0))

def AvgPoolUndo(avgGrads, targets, conv_desc, scaleTargets=0):
  _ConvNet.AvgPoolUndoGemm(avgGrads.p_mat, targets.p_mat, avgGrads.p_shape4d,
                       targets.p_shape4d, conv_desc, ct.c_float(scaleTargets))

def ResponseNormCrossMap(images, targets, sizeF, addScale, powScale, blocked):
  _, _, _, num_filters = images.shape4d
  _ConvNet.ResponseNormCrossMapGemm(
    images.p_mat, targets.p_mat, ct.c_int(num_filters), ct.c_int(sizeF),
    ct.c_float(addScale), ct.c_float(powScale), ct.c_int(blocked))

def ResponseNormCrossMapUndo(derivs, images, targets, sizeF, addScale, powScale, blocked):
  _, _, _, num_filters = images.shape4d
  _ConvNet.ResponseNormCrossMapUndoGemm(
    derivs.p_mat, images.p_mat, targets.p_mat, ct.c_int(num_filters), ct.c_int(sizeF),
    ct.c_float(addScale), ct.c_float(powScale), ct.c_int(blocked))

def convUp3D(images, filters, targets, conv_desc, scaleTargets=0, bias=None):
  _ConvNet.convUp3DGemm(images.p_mat, filters.p_mat, targets.p_mat,
                  images.p_shape4d, filters.p_shape4d, targets.p_shape4d,
                  conv_desc, ct.c_float(scaleTargets))
  if bias is not None:
    AddAtAllLocs3D(targets, bias)

def convDown3D(hidSums, filters, targets, conv_desc, scaleTargets=0):
  _ConvNet.convDown3DGemm(hidSums.p_mat, filters.p_mat, targets.p_mat,
                    hidSums.p_shape4d, filters.p_shape4d, targets.p_shape4d,
                    conv_desc, ct.c_float(scaleTargets))

def convOutp3D(images, hidSums, targets, conv_desc, scaleTargets=0, dbias=None):
  _ConvNet.convOutp3DGemm(
    images.p_mat, hidSums.p_mat, targets.p_mat,
    images.p_shape4d, hidSums.p_shape4d, targets.p_shape4d,
    conv_desc, ct.c_float(scaleTargets), ct.c_float(1))
  if dbias is not None:
    AddUpAllLocs3D(hidSums, dbias)

def MaxPool3D(images, targets, conv_desc):
  MaxPool(images, targets, conv_desc)

def MaxPool3DUndo(images, grad, maxes, targets, conv_desc, scaleTargets=0):
  MaxPoolUndo(images, grad, maxes, targets, conv_desc, scaleTargets=scaleTargets)

def AvgPool3D(images, targets, conv_desc):
  AvgPool(images, targets, conv_desc)

def AvgPool3DUndo(avgGrads, targets, conv_desc, scaleTargets=0):
  AvgPoolUndo(avgGrads, targets, conv_desc, scaleTargets=scaleTargets)
