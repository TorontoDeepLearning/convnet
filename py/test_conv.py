import sys
from convnet import *
import numpy as np
import conv_cpu

test_gemm = False

def DivUp(a, b):
  return (a + b - 1) / b

class TestConvNet(ConvNet):
  def __init__(self, model_pbtxt):
    super(TestConvNet, self).__init__(model_pbtxt)

  def TestFprop(self, input_data, cpu_check=False):
    batch_size = input_data.shape[0]
    if self.batch_size_ != batch_size:
      self.SetBatchSize(batch_size)

    for l in self.layer_:
      overwrite = True
      if cpu_check:
        cpu_out = 0
      for e in l.incoming_edge_:
        e.ComputeUp(e.GetSource(), l, overwrite)
        if cpu_check:
          cpu_out += e.ComputeUpCPU(e.GetSource().asarray())
        overwrite = False
      if cpu_check:
        Check(Diff(l.GetState().asarray(), cpu_out))
      if l.IsInput():
        state = l.GetState()
        state.overwrite(input_data)
        self.Normalize(state)
        l.ApplyDropout()
      else:
        l.ApplyActivation()

def TestConvUp(images_shape, conv_desc):
  filters_shape = (conv_desc.num_output_channels, conv_desc.kernel_size_x, conv_desc.kernel_size_y, conv_desc.num_input_channels)
  output_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  
  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  filters = np.random.randn(filters_shape[0], filters_shape[1] * filters_shape[2] * filters_shape[3]).astype(np.float32)
 
  images_gpu = cm.CUDAMatrix(images)
  filters_gpu = cm.CUDAMatrix(filters)
  output_gpu = cm.empty((output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]))

  images_gpu.set_shape4d(images_shape)
  filters_gpu.set_shape4d(filters_shape)
  output_gpu.set_shape4d(output_shape)

  if test_gemm:
    cc_gemm.convUp(images_gpu, filters_gpu, output_gpu, conv_desc)
  else:
    cc.convUp(images_gpu, filters_gpu, output_gpu, conv_desc)
  output_cpu = conv_cpu.ConvUp(images, filters, images_shape, cm.GetConvDescTuple2(conv_desc))

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
 
  derivs_gpu = cm.CUDAMatrix(derivs)
  filters_gpu = cm.CUDAMatrix(filters)
  images_gpu = cm.empty((images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]))

  derivs_gpu.set_shape4d(deriv_shape)
  filters_gpu.set_shape4d(filters_shape)
  images_gpu.set_shape4d(images_shape)

  if test_gemm:
    cc_gemm.convDown(derivs_gpu, filters_gpu, images_gpu, conv_desc)
  else:
    cc.convDown(derivs_gpu, filters_gpu, images_gpu, conv_desc)
  images_cpu = conv_cpu.ConvDown(derivs, filters, images_shape, cm.GetConvDescTuple2(conv_desc))

  diff = Diff(images_cpu, images_gpu.asarray())
  images_gpu.free_device_memory()
  filters_gpu.free_device_memory()
  derivs_gpu.free_device_memory()
  return diff

def TestConvOutp(images_shape, conv_desc, partial_sum_y=0, partial_sum_x=0):
  filters_shape = (conv_desc.num_output_channels, conv_desc.kernel_size_x, conv_desc.kernel_size_y, conv_desc.num_input_channels)
  deriv_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  batch_size, num_modules_x, num_modules_y, num_output_channels = deriv_shape

  if partial_sum_x == 0:
    partial_sum_x = num_modules_x
  if partial_sum_y == 0:
    partial_sum_y = num_modules_y
  partial_sum_locs_y = DivUp(num_modules_y, partial_sum_y)
  partial_sum_locs_x = DivUp(num_modules_x, partial_sum_x)

  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  derivs = np.random.randn(deriv_shape[0], deriv_shape[1] * deriv_shape[2] * deriv_shape[3]).astype(np.float32)
 
  images_gpu = cm.CUDAMatrix(images)
  filters_gpu = cm.empty((filters_shape[0], filters_shape[1] * filters_shape[2] * filters_shape[3]))
  filters_temp_gpu = cm.empty((filters_shape[0], filters_shape[1] * filters_shape[2] * filters_shape[3] * partial_sum_locs_x * partial_sum_locs_y))
  derivs_gpu = cm.CUDAMatrix(derivs)

  images_gpu.set_shape4d(images_shape)
  filters_gpu.set_shape4d(filters_shape)
  filters_temp_gpu.set_shape4d((filters_shape[0], filters_shape[1], filters_shape[2], filters_shape[3] * partial_sum_locs_x * partial_sum_locs_y))
  derivs_gpu.set_shape4d(deriv_shape)

  if test_gemm:
    cc_gemm.convOutp(images_gpu, derivs_gpu, filters_gpu, conv_desc, partialSumY=partial_sum_y, partialSumX=partial_sum_x, temp=filters_temp_gpu)
  else:
    cc.convOutp(images_gpu, derivs_gpu, filters_gpu, conv_desc, partialSumY=partial_sum_y, partialSumX=partial_sum_x, temp=filters_temp_gpu)
  filters_cpu, filters_temp_cpu = conv_cpu.ConvOutp(images, derivs, images_shape, cm.GetConvDescTuple2(conv_desc), partial_sum_y=partial_sum_y, partial_sum_x=partial_sum_x)

  diff1 = Diff(filters_gpu.asarray(), filters_cpu)
  diff2 = Diff(filters_temp_gpu.asarray(), filters_temp_cpu)
  
  images_gpu.free_device_memory()
  filters_gpu.free_device_memory()
  filters_temp_gpu.free_device_memory()
  derivs_gpu.free_device_memory()
  return diff1, diff2

def TestMaxPool(images_shape, conv_desc):
  output_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  images_gpu = cm.CUDAMatrix(images)
  output_gpu = cm.empty((output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]))

  images_gpu.set_shape4d(images_shape)
  output_gpu.set_shape4d(output_shape)
  if test_gemm:
    cc_gemm.MaxPool(images_gpu, output_gpu, conv_desc)
  else:
    cc.MaxPool(images_gpu, output_gpu, conv_desc)
  output_cpu = conv_cpu.MaxPool(images, images_shape, cm.GetConvDescTuple2(conv_desc))

  diff = Diff(output_cpu, output_gpu.asarray())
  images_gpu.free_device_memory()
  output_gpu.free_device_memory()
  return diff

def TestMaxPoolUndo(images_shape, conv_desc):
  deriv_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  images = np.random.rand(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  derivs = np.random.randn(deriv_shape[0], deriv_shape[1] * deriv_shape[2] * deriv_shape[3]).astype(np.float32)
  maxes  = conv_cpu.MaxPool(images, images_shape, cm.GetConvDescTuple2(conv_desc))

  images_gpu = cm.CUDAMatrix(images)
  derivs_gpu = cm.CUDAMatrix(derivs)
  maxes_gpu = cm.CUDAMatrix(maxes)
  targets_gpu = cm.empty((images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]))

  images_gpu.set_shape4d(images_shape)
  derivs_gpu.set_shape4d(deriv_shape)
  maxes_gpu.set_shape4d(deriv_shape)
  targets_gpu.set_shape4d(images_shape)
  if test_gemm:
    cc_gemm.MaxPoolUndo(images_gpu, derivs_gpu, maxes_gpu, targets_gpu, conv_desc)
  else:
    cc.MaxPoolUndo(images_gpu, derivs_gpu, maxes_gpu, targets_gpu, conv_desc)
  output_cpu = conv_cpu.MaxPoolUndo(images, maxes, derivs, images_shape, deriv_shape, cm.GetConvDescTuple2(conv_desc))

  diff = Diff(output_cpu, targets_gpu.asarray())
  images_gpu.free_device_memory()
  derivs_gpu.free_device_memory()
  maxes_gpu.free_device_memory()
  targets_gpu.free_device_memory()
  return diff

def TestAvgPool(images_shape, conv_desc):
  output_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  images = np.random.randn(images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]).astype(np.float32)
  images_gpu = cm.CUDAMatrix(images)
  output_gpu = cm.empty((output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]))

  images_gpu.set_shape4d(images_shape)
  output_gpu.set_shape4d(output_shape)
  if test_gemm:
    cc_gemm.AvgPool(images_gpu, output_gpu, conv_desc)
  else:
    cc.AvgPool(images_gpu, output_gpu, conv_desc)
  output_cpu = conv_cpu.AvgPool(images, images_shape, cm.GetConvDescTuple2(conv_desc))

  diff = Diff(output_cpu, output_gpu.asarray())
  images_gpu.free_device_memory()
  output_gpu.free_device_memory()
  return diff

def TestAvgPoolUndo(images_shape, conv_desc):
  deriv_shape = cm.GetOutputShape4D(images_shape, conv_desc)
  derivs = np.random.randn(deriv_shape[0], deriv_shape[1] * deriv_shape[2] * deriv_shape[3]).astype(np.float32)

  derivs_gpu = cm.CUDAMatrix(derivs)
  targets_gpu = cm.empty((images_shape[0], images_shape[1] * images_shape[2] * images_shape[3]))

  derivs_gpu.set_shape4d(deriv_shape)
  targets_gpu.set_shape4d(images_shape)
  if test_gemm:
    cc_gemm.AvgPoolUndo(derivs_gpu, targets_gpu, conv_desc)
  else:
    cc.AvgPoolUndo(derivs_gpu, targets_gpu, conv_desc)
  output_cpu = conv_cpu.AvgPoolUndo(derivs, images_shape, cm.GetConvDescTuple2(conv_desc))

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

def Test():
  batch_size = 128
  image_size_x = 14
  image_size_y = 28
  num_input_channels = 32
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

def TestModel(model_pbtxt):
  model = cn.ConvNet(model_pbtxt)

def main():
  Test()

if __name__ == '__main__':
  board = LockGPU()
  print 'Using board', board
  main()
  FreeGPU(board)
