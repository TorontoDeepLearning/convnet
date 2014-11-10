import ctypes as ct
import math
import pdb
_ConvNet = ct.cdll.LoadLibrary('libcudamat_conv.so')

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

def convUp(images, filters, targets, conv_desc, scaleTargets=0):
  _ConvNet.convUp(images.p_mat, filters.p_mat, targets.p_mat,
                  images.p_shape4d, filters.p_shape4d, targets.p_shape4d,
                  conv_desc, ct.c_float(scaleTargets))

def localUp(images, filters, targets, conv_desc, scaleTargets=0):
  _ConvNet.localUp(images.p_mat, filters.p_mat, targets.p_mat,
                   images.p_shape4d, filters.p_shape4d, targets.p_shape4d,
                   conv_desc, ct.c_float(scaleTargets))

def convDown(hidSums, filters, targets, conv_desc, scaleTargets=0):
  _ConvNet.convDown(hidSums.p_mat, filters.p_mat, targets.p_mat,
                    hidSums.p_shape4d, filters.p_shape4d, targets.p_shape4d,
                    conv_desc, ct.c_float(scaleTargets))

def localDown(hidSums, filters, targets, conv_desc, scaleTargets=0):
  _ConvNet.localDown(hidSums.p_mat, filters.p_mat, targets.p_mat,
                     hidSums.p_shape4d, filters.p_shape4d, targets.p_shape4d,
                     conv_desc, ct.c_float(scaleTargets))

def convOutp(images, hidSums, targets, conv_desc, scaleTargets=0, partialSumY=0, partialSumX=0, temp=None):
  num_images, num_modules_x, num_modules_y, num_output_channels = hidSums.shape4d
  num_output_channels2, kernel_size_x, kernel_size_y, num_input_channels = targets.shape4d
  if partialSumY == 0:
    partialSumY = num_modules_y
  if partialSumX == 0:
    partialSumX = num_modules_x
  temp_alloc = False
  num_locs = DivUp(num_modules_x, partialSumX) * DivUp(num_modules_y, partialSumY)
  if num_locs == 1:
    outp = targets
    scale_targets = scaleTargets
  else:
    if temp is None:
      temp_alloc = True
      temp = cm.empty((num_output_channels, kernel_size_x * kernel_size_y * num_input_channels * num_locs))
      temp.set_shape4d((num_output_channels, kernel_size_x, kernel_size_y, num_input_channels * num_locs))
    outp = temp
    scale_targets = 0

  if temp is not None:
    num_output_channels3, kernel_size_x2, kernel_size_y2, num_input_channels_mult_partial_sum = temp.shape4d
    assert kernel_size_y2 == kernel_size_y
    assert kernel_size_x2 == kernel_size_x
    assert num_output_channels3 == num_output_channels
    assert num_input_channels_mult_partial_sum % num_input_channels == 0
    assert num_locs == num_input_channels_mult_partial_sum / num_input_channels
 
  _ConvNet.convOutp(
    images.p_mat, hidSums.p_mat, outp.p_mat,
    images.p_shape4d, hidSums.p_shape4d, outp.p_shape4d,
    conv_desc, ct.c_int(partialSumY), ct.c_int(partialSumX),
    ct.c_float(scale_targets), ct.c_float(1))

  if num_locs > 1:
    temp.reshape((-1, num_locs))
    targets.reshape((-1, 1))
    targets.mult(scaleTargets)
    targets.add_sums(temp, axis=1)
    temp.reshape((num_output_channels, -1))
    targets.reshape((num_output_channels, -1))
    if temp_alloc:
      temp.free_device_memory()
  elif temp is not None:
    temp.assign(outp)

def localOutp(images, hidSums, targets, conv_desc, scaleTargets=0):
  _ConvNet.localOutp(
    images.p_mat, hidSums.p_mat, targets.p_mat,
    images.p_shape4d, hidSums.p_shape4d, targets.p_shape4d,
    conv_desc, ct.c_float(scale_targets), ct.c_float(1))

def MaxPool(images, targets, conv_desc):
  _ConvNet.MaxPool(images.p_mat, targets.p_mat, images.p_shape4d,
                   targets.p_shape4d, conv_desc)

def AvgPool(images, targets, conv_desc):
  _ConvNet.AvgPool(images.p_mat, targets.p_mat, images.p_shape4d,
                   targets.p_shape4d, conv_desc)

def MaxPoolUndo(images, grad, maxes, targets, conv_desc, scaleTargets=0):
  _ConvNet.MaxPoolUndo(images.p_mat, grad.p_mat, maxes.p_mat, targets.p_mat,
                       images.p_shape4d, grad.p_shape4d, conv_desc,
                       ct.c_float(scaleTargets))

def AvgPoolUndo(avgGrads, targets, conv_desc, scaleTargets=0):
  _ConvNet.AvgPoolUndo(avgGrads.p_mat, targets.p_mat, avgGrads.p_shape4d,
                       targets.p_shape4d, conv_desc, ct.c_float(scaleTargets))

def ResponseNorm(images, denoms, targets, numChannels, sizeX, addScale, powScale):
  assert targets.shape == images.shape
  assert targets.shape == denoms.shape
  _ConvNet.ResponseNorm(images.p_mat, denoms.p_mat, targets.p_mat,
             numChannels, sizeX, ct.c_float(addScale),
             ct.c_float(powScale))

def ResponseNormCrossMap(images, targets, numChannels, sizeF, addScale, powScale, blocked):
  assert targets.shape == images.shape
  _ConvNet.ResponseNormCrossMap(images.p_mat, targets.p_mat, numChannels, sizeF, ct.c_float(addScale),
                                ct.c_float(powScale), blocked)

def ResponseNormUndo(outGrad, denoms, inGrad, acts, targets, numChannels, sizeX,
                     addScale, powScale):
  assert targets.shape == outGrad.shape
  assert targets.shape == denoms.shape
  assert targets.shape == inGrad.shape
  assert targets.shape == acts.shape
  _ConvNet.ResponseNormUndo(outGrad.p_mat, denoms.p_mat, inGrad.p_mat,
               acts.p_mat, targets.p_mat, numChannels, sizeX,
               ct.c_float(addScale), ct.c_float(powScale))

def ResponseNormCrossMapUndo(outGrad, inGrad, acts, targets, numChannels, sizeF,
                             addScale, powScale, blocked):
  assert targets.shape == outGrad.shape
  assert targets.shape == inGrad.shape
  assert targets.shape == acts.shape
  _ConvNet.ResponseNormUndo(outGrad.p_mat, inGrad.p_mat,
                            acts.p_mat, targets.p_mat, numChannels, sizeF,
                            ct.c_float(addScale), ct.c_float(powScale), blocked)
