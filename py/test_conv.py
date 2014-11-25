import sys
from convnet import *
import numpy as np
import conv_cpu

test_gemm = True

def DivUp(a, b):
  return (a + b - 1) / b

def TestConvUp(images_shape, conv_desc):
  filters_shape = (conv_desc.num_output_channels, conv_desc.kernel_size_x, conv_desc.kernel_size_y, conv_desc.num_input_channels)
  output_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  
  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  filters = np.random.randn(filters_shape[0], filters_shape[1] * filters_shape[2] * filters_shape[3]).astype(np.float32)
 
  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  filters_gpu = cm.CUDAMatrix(filters, shape=filters_shape)
  output_gpu = cm.empty(output_shape)

  if test_gemm:
    cc_gemm.convUp(images_gpu, filters_gpu, output_gpu, conv_desc)
  else:
    cc.convUp(images_gpu, filters_gpu, output_gpu, conv_desc)
  output_cpu = conv_cpu.ConvUp(images, filters, images_shape, cm.GetConvDescTuple(conv_desc))

  diff = Diff(output_cpu, output_gpu.asarray())
  images_gpu.free_device_memory()
  filters_gpu.free_device_memory()
  output_gpu.free_device_memory()
  return diff

def TestConvDown(images_shape, conv_desc):
  deriv_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  filters_shape = (conv_desc.num_output_channels, conv_desc.kernel_size_x, conv_desc.kernel_size_y, conv_desc.num_input_channels)
  
  derivs = np.random.randn(deriv_shape[0], deriv_shape[1] * deriv_shape[2] * deriv_shape[3]).astype(np.float32)
  filters = np.random.randn(filters_shape[0], filters_shape[1] * filters_shape[2] * filters_shape[3]).astype(np.float32)
 
  derivs_gpu = cm.CUDAMatrix(derivs, shape=deriv_shape)
  filters_gpu = cm.CUDAMatrix(filters, shape=filters_shape)
  images_gpu = cm.empty(images_shape)

  if test_gemm:
    cc_gemm.convDown(derivs_gpu, filters_gpu, images_gpu, conv_desc)
  else:
    cc.convDown(derivs_gpu, filters_gpu, images_gpu, conv_desc)
  images_cpu = conv_cpu.ConvDown(derivs, filters, images_shape, cm.GetConvDescTuple(conv_desc))

  diff = Diff(images_cpu, images_gpu.asarray())
  images_gpu.free_device_memory()
  filters_gpu.free_device_memory()
  derivs_gpu.free_device_memory()
  return diff

def TestConvOutp(images_shape, conv_desc, partial_sum_y=0, partial_sum_x=0):
  filters_shape = (conv_desc.num_output_channels, conv_desc.kernel_size_x, conv_desc.kernel_size_y, conv_desc.num_input_channels)
  deriv_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  batch_size, num_modules_x, num_modules_y, num_output_channels = deriv_shape

  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  derivs = np.random.randn(deriv_shape[0], deriv_shape[1] * deriv_shape[2] * deriv_shape[3]).astype(np.float32)
 
  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  filters_gpu = cm.empty(filters_shape)
  derivs_gpu = cm.CUDAMatrix(derivs, shape=deriv_shape)

  if test_gemm:
    cc_gemm.convOutp(images_gpu, derivs_gpu, filters_gpu, conv_desc)
  else:
    if partial_sum_x == 0:
      partial_sum_x = num_modules_x
    if partial_sum_y == 0:
      partial_sum_y = num_modules_y
    partial_sum_locs_y = DivUp(num_modules_y, partial_sum_y)
    partial_sum_locs_x = DivUp(num_modules_x, partial_sum_x)
    filters_temp_gpu = cm.empty((filters_shape[0], filters_shape[1], filters_shape[2], filters_shape[3] * partial_sum_locs_x * partial_sum_locs_y))
    cc.convOutp(images_gpu, derivs_gpu, filters_gpu, conv_desc, partialSumY=partial_sum_y, partialSumX=partial_sum_x, temp=filters_temp_gpu)

  filters_cpu, filters_temp_cpu = conv_cpu.ConvOutp(images, derivs, images_shape, cm.GetConvDescTuple(conv_desc), partial_sum_y=partial_sum_y, partial_sum_x=partial_sum_x)

  diff1 = Diff(filters_gpu.asarray(), filters_cpu)
  if test_gemm:
    diff2 = 0
  else:
    diff2 = Diff(filters_temp_gpu.asarray(), filters_temp_cpu)
    filters_temp_gpu.free_device_memory()
  
  images_gpu.free_device_memory()
  filters_gpu.free_device_memory()
  derivs_gpu.free_device_memory()
  return diff1, diff2

def TestMaxPool(images_shape, conv_desc):
  output_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  output_gpu = cm.empty(output_shape)

  if test_gemm:
    cc_gemm.MaxPool(images_gpu, output_gpu, conv_desc)
  else:
    cc.MaxPool(images_gpu, output_gpu, conv_desc)
  output_cpu = conv_cpu.MaxPool(images, images_shape, cm.GetConvDescTuple(conv_desc))

  diff = Diff(output_cpu, output_gpu.asarray())
  images_gpu.free_device_memory()
  output_gpu.free_device_memory()
  return diff

def TestMaxPoolUndo(images_shape, conv_desc):
  deriv_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  images = np.random.rand(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  derivs = np.random.randn(deriv_shape[0], deriv_shape[1] * deriv_shape[2] * deriv_shape[3]).astype(np.float32)
  maxes  = conv_cpu.MaxPool(images, images_shape, cm.GetConvDescTuple(conv_desc))

  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  derivs_gpu = cm.CUDAMatrix(derivs, shape=deriv_shape)
  maxes_gpu = cm.CUDAMatrix(maxes, shape=deriv_shape)
  targets_gpu = cm.empty(images_shape)

  if test_gemm:
    cc_gemm.MaxPoolUndo(images_gpu, derivs_gpu, maxes_gpu, targets_gpu, conv_desc)
  else:
    cc.MaxPoolUndo(images_gpu, derivs_gpu, maxes_gpu, targets_gpu, conv_desc)
  output_cpu = conv_cpu.MaxPoolUndo(images, maxes, derivs, images_shape, deriv_shape, cm.GetConvDescTuple(conv_desc))

  diff = Diff(output_cpu, targets_gpu.asarray())
  images_gpu.free_device_memory()
  derivs_gpu.free_device_memory()
  maxes_gpu.free_device_memory()
  targets_gpu.free_device_memory()
  return diff

def TestAvgPool(images_shape, conv_desc):
  output_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  output_gpu = cm.empty(output_shape)

  if test_gemm:
    cc_gemm.AvgPool(images_gpu, output_gpu, conv_desc)
  else:
    cc.AvgPool(images_gpu, output_gpu, conv_desc)
  output_cpu = conv_cpu.AvgPool(images, images_shape, cm.GetConvDescTuple(conv_desc))

  diff = Diff(output_cpu, output_gpu.asarray())
  images_gpu.free_device_memory()
  output_gpu.free_device_memory()
  return diff

def TestAvgPoolUndo(images_shape, conv_desc):
  deriv_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  derivs = np.random.randn(deriv_shape[0], deriv_shape[1] * deriv_shape[2] * deriv_shape[3]).astype(np.float32)

  derivs_gpu = cm.CUDAMatrix(derivs, shape=deriv_shape)
  targets_gpu = cm.empty(images_shape)

  if test_gemm:
    cc_gemm.AvgPoolUndo(derivs_gpu, targets_gpu, conv_desc)
  else:
    cc.AvgPoolUndo(derivs_gpu, targets_gpu, conv_desc)
  output_cpu = conv_cpu.AvgPoolUndo(derivs, images_shape, cm.GetConvDescTuple(conv_desc))

  diff = Diff(output_cpu, targets_gpu.asarray())
  derivs_gpu.free_device_memory()
  targets_gpu.free_device_memory()
  return diff

def TestResponseNormCrossMap(images_shape, numF, add_scale, pow_scale, blocked):
  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  targets_gpu = cm.empty(images_shape)
  if test_gemm:
    cc_gemm.ResponseNormCrossMap(images_gpu, targets_gpu, numF, add_scale, pow_scale, blocked)
  else:
    cc.ResponseNormCrossMap(images_gpu, targets_gpu, numF, add_scale, pow_scale, blocked)
  output_cpu = conv_cpu.ResponseNormCrossMap(images, images_shape, numF, add_scale, pow_scale, blocked)
  diff = Diff(output_cpu, targets_gpu.asarray())
  images_gpu.free_device_memory()
  targets_gpu.free_device_memory()
  return diff

def TestResponseNormCrossMapUndo(images_shape, numF, add_scale, pow_scale, blocked):
  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  derivs = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  derivs_gpu = cm.CUDAMatrix(derivs, shape=images_shape)
  targets_gpu = cm.empty(images_shape)
  if test_gemm:
    cc_gemm.ResponseNormCrossMapUndo(derivs_gpu, images_gpu, targets_gpu, numF, add_scale, pow_scale, blocked)
  else:
    acts_gpu = cm.empty(images_shape)
    cc.ResponseNormCrossMap(images_gpu, acts_gpu, numF, add_scale, pow_scale, blocked)
    cc.ResponseNormCrossMapUndo(derivs_gpu, images_gpu, acts_gpu, targets_gpu, numF, add_scale, pow_scale, blocked)
    acts_gpu.free_device_memory()
  output_cpu = conv_cpu.ResponseNormCrossMapUndo(derivs, images, images_shape, numF, add_scale, pow_scale, blocked)
  diff = Diff(output_cpu, targets_gpu.asarray())
  images_gpu.free_device_memory()
  targets_gpu.free_device_memory()
  return diff

def TestConvUp3D(images_shape, conv_desc):
  filters_shape = (conv_desc.num_output_channels, conv_desc.kernel_size_x, conv_desc.kernel_size_y, conv_desc.num_input_channels * conv_desc.kernel_size_t)
  output_shape = cm.GetOutputShape5D(images_shape, conv_desc)
  
  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3] * images_shape[4]).astype(np.float32)
  filters = np.random.randn(filters_shape[0], filters_shape[1] * filters_shape[2] * filters_shape[3]).astype(np.float32)
 
  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  filters_gpu = cm.CUDAMatrix(filters, shape=filters_shape)
  output_gpu = cm.empty(output_shape)

  assert test_gemm
  cc_gemm.convUp3D(images_gpu, filters_gpu, output_gpu, conv_desc)
  output_cpu = conv_cpu.ConvUp3D(images, filters, images_shape, cm.GetConvDescTuple3D(conv_desc))

  diff = Diff(output_cpu, output_gpu.asarray())
  images_gpu.free_device_memory()
  filters_gpu.free_device_memory()
  output_gpu.free_device_memory()
  return diff

def TestConvDown3D(images_shape, conv_desc):
  filters_shape = (conv_desc.num_output_channels, conv_desc.kernel_size_x, conv_desc.kernel_size_y, conv_desc.num_input_channels * conv_desc.kernel_size_t)
  deriv_shape = cm.GetOutputShape5D(images_shape, conv_desc)
  
  derivs = np.random.randn(deriv_shape[0], deriv_shape[1] * deriv_shape[2] * deriv_shape[3] * deriv_shape[4]).astype(np.float32)
  filters = np.random.randn(filters_shape[0], filters_shape[1] * filters_shape[2] * filters_shape[3]).astype(np.float32)
 
  derivs_gpu = cm.CUDAMatrix(derivs, shape=deriv_shape)
  filters_gpu = cm.CUDAMatrix(filters, shape=filters_shape)
  images_gpu = cm.empty(images_shape)

  assert test_gemm
  cc_gemm.convDown3D(derivs_gpu, filters_gpu, images_gpu, conv_desc)
  images_cpu = conv_cpu.ConvDown3D(derivs, filters, images_shape, cm.GetConvDescTuple3D(conv_desc))

  diff = Diff(images_cpu, images_gpu.asarray())
  images_gpu.free_device_memory()
  filters_gpu.free_device_memory()
  derivs_gpu.free_device_memory()
  return diff

def TestConvOutp3D(images_shape, conv_desc, partial_sum_y=0, partial_sum_x=0):
  filters_shape = (conv_desc.num_output_channels, conv_desc.kernel_size_x, conv_desc.kernel_size_y, conv_desc.num_input_channels * conv_desc.kernel_size_t)
  deriv_shape = cm.GetOutputShape5D(images_shape, conv_desc)
  batch_size, num_modules_x, num_modules_y, num_output_channels, num_modules_t = deriv_shape

  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3] * images_shape[4]).astype(np.float32)
  derivs = np.random.randn(deriv_shape[0], deriv_shape[1] * deriv_shape[2] * deriv_shape[3] * deriv_shape[4]).astype(np.float32)
 
  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  filters_gpu = cm.empty(filters_shape)
  derivs_gpu = cm.CUDAMatrix(derivs, shape=deriv_shape)
  
  cc_gemm.convOutp3D(images_gpu, derivs_gpu, filters_gpu, conv_desc)
  filters_cpu = conv_cpu.ConvOutp3D(images, derivs, images_shape, cm.GetConvDescTuple3D(conv_desc))
  diff = Diff(filters_gpu.asarray(), filters_cpu)
  
  images_gpu.free_device_memory()
  filters_gpu.free_device_memory()
  derivs_gpu.free_device_memory()
  return diff

def TestMaxPool3D(images_shape, conv_desc):
  output_shape = cm.GetOutputShape5D(images_shape, conv_desc)
  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3] * images_shape[4]).astype(np.float32)
  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  output_gpu = cm.empty(output_shape)

  assert test_gemm
  cc_gemm.MaxPool3D(images_gpu, output_gpu, conv_desc)
  output_cpu = conv_cpu.MaxPool3D(images, images_shape, cm.GetConvDescTuple3D(conv_desc))

  diff = Diff(output_cpu, output_gpu.asarray())
  images_gpu.free_device_memory()
  output_gpu.free_device_memory()
  return diff

def TestMaxPool3DUndo(images_shape, conv_desc):
  deriv_shape = cm.GetOutputShape5D(images_shape, conv_desc)
  images = np.random.rand(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3] * images_shape[4]).astype(np.float32)
  derivs = np.random.randn(deriv_shape[0], deriv_shape[1] * deriv_shape[2] * deriv_shape[3] * deriv_shape[4]).astype(np.float32)
  maxes  = conv_cpu.MaxPool3D(images, images_shape, cm.GetConvDescTuple3D(conv_desc))

  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  derivs_gpu = cm.CUDAMatrix(derivs, shape=deriv_shape)
  maxes_gpu = cm.CUDAMatrix(maxes, shape=deriv_shape)
  targets_gpu = cm.empty(images_shape)
  
  assert test_gemm
  cc_gemm.MaxPool3DUndo(images_gpu, derivs_gpu, maxes_gpu, targets_gpu, conv_desc)
  output_cpu = conv_cpu.MaxPool3DUndo(images, maxes, derivs, images_shape, deriv_shape, cm.GetConvDescTuple3D(conv_desc))

  diff = Diff(output_cpu, targets_gpu.asarray())
  images_gpu.free_device_memory()
  derivs_gpu.free_device_memory()
  maxes_gpu.free_device_memory()
  targets_gpu.free_device_memory()
  return diff

def TestAvgPool3D(images_shape, conv_desc):
  output_shape = cm.GetOutputShape5D(images_shape, conv_desc)
  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3] * images_shape[4]).astype(np.float32)
  images_gpu = cm.CUDAMatrix(images, shape=images_shape)
  output_gpu = cm.empty(output_shape)

  assert test_gemm
  cc_gemm.AvgPool3D(images_gpu, output_gpu, conv_desc)
  output_cpu = conv_cpu.AvgPool3D(images, images_shape, cm.GetConvDescTuple3D(conv_desc))

  diff = Diff(output_cpu, output_gpu.asarray())
  images_gpu.free_device_memory()
  output_gpu.free_device_memory()
  return diff

def TestAvgPool3DUndo(images_shape, conv_desc):
  deriv_shape = cm.GetOutputShape5D(images_shape, conv_desc)
  derivs = np.random.randn(deriv_shape[0], deriv_shape[1] * deriv_shape[2] * deriv_shape[3] * deriv_shape[4]).astype(np.float32)

  derivs_gpu = cm.CUDAMatrix(derivs, shape=deriv_shape)
  targets_gpu = cm.empty(images_shape)

  assert test_gemm
  cc_gemm.AvgPool3DUndo(derivs_gpu, targets_gpu, conv_desc)
  output_cpu = conv_cpu.AvgPool3DUndo(derivs, images_shape, cm.GetConvDescTuple3D(conv_desc))

  diff = Diff(output_cpu, targets_gpu.asarray())
  derivs_gpu.free_device_memory()
  targets_gpu.free_device_memory()
  return diff

def Diff(a, b):
  scale = np.abs(a + b).mean()
  diff = np.abs(a - b).max() / scale
  return diff

def Check(diff, tol=1e-4):
  if diff < tol:
    result = 'PASSED'
  else:
    result = 'FAILED'
  print diff, result

def Test2D():
  batch_size = 128
  image_size_x = 12
  image_size_y = 12
  num_input_channels = 32
  sizeF = 8
  add_scale = 0.005
  pow_scale = 0.75
  blocked = False
  num_output_channels = 64
  kernel_size_y = 3
  kernel_size_x = 3
  stride_y = 2
  stride_x = 2
  padding_y = 1
  padding_x = 1
  partial_sum = 0

  images_shape = (batch_size, image_size_x, image_size_y, num_input_channels)
  conv_desc = cm.GetConvDesc(num_input_channels, num_output_channels,
                             kernel_size_y, kernel_size_x, stride_y,
                             stride_x, padding_y, padding_x)
  pool_desc = cm.GetConvDesc(num_input_channels, num_input_channels,
                             kernel_size_y, kernel_size_x, stride_y,
                             stride_x, padding_y, padding_x)
  print 'ConvUp'
  Check(TestConvUp(images_shape, conv_desc))
  print 'ConvDown'
  Check(TestConvDown(images_shape, conv_desc))
  print 'ConvOutp'
  d1, d2 = TestConvOutp(images_shape, conv_desc, partial_sum_y=partial_sum, partial_sum_x=partial_sum)
  Check(d1)
  print 'MaxPool'
  Check(TestMaxPool(images_shape, pool_desc))
  print 'AvgPool'
  Check(TestAvgPool(images_shape, pool_desc))
  print 'MaxPoolUndo'
  Check(TestMaxPoolUndo(images_shape, pool_desc))
  print 'AvgPoolUndo'
  Check(TestAvgPoolUndo(images_shape, pool_desc))
  print 'ResponseNormCrossMap'
  Check(TestResponseNormCrossMap(images_shape, sizeF, add_scale, pow_scale, blocked))
  print 'ResponseNormCrossMapUndo'
  Check(TestResponseNormCrossMapUndo(images_shape, sizeF, add_scale, pow_scale, blocked))

def Test3D():
  batch_size = 128
  image_size_x = 32
  image_size_y = 24
  image_size_t = 12
  num_input_channels = 3
  num_output_channels = 64
  kernel_size_y = 7
  kernel_size_x = 7
  kernel_size_t = 3
  stride_y = 2
  stride_x = 2
  stride_t = 2
  padding_y = 1
  padding_x = 1
  padding_t = 0

  images_shape = (batch_size, image_size_x, image_size_y, num_input_channels, image_size_t)
  conv_desc = cm.GetConvDesc(num_input_channels, num_output_channels,
                             kernel_size_y, kernel_size_x,
                             stride_y, stride_x,
                             padding_y, padding_x, kernel_size_t, stride_t, padding_t)
  pool_desc = cm.GetConvDesc(num_input_channels, num_input_channels,
                             kernel_size_y, kernel_size_x,
                             stride_y, stride_x,
                             padding_y, padding_x, kernel_size_t, stride_t, padding_t)
  print 'ConvUp'
  Check(TestConvUp3D(images_shape, conv_desc))
  print 'ConvDown'
  Check(TestConvDown3D(images_shape, conv_desc))
  print 'ConvOutp'
  Check(TestConvOutp3D(images_shape, conv_desc))
  print 'MaxPool'
  Check(TestMaxPool3D(images_shape, pool_desc))
  print 'MaxPoolUndo'
  Check(TestMaxPool3DUndo(images_shape, pool_desc))
  print 'AvgPool'
  Check(TestAvgPool3D(images_shape, pool_desc))
  print 'AvgPoolUndo'
  Check(TestAvgPool3DUndo(images_shape, pool_desc))

def main():
  print "Testing 2D convolutions"
  Test2D()
  print "Testing 3D convolutions"
  Test3D()

if __name__ == '__main__':
  board = LockGPU()
  print 'Using board', board
  main()
  FreeGPU(board)
