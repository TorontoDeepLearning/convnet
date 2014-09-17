import ctypes as ct
import math
import pdb
_ConvNet = ct.cdll.LoadLibrary('libcudamat_conv.so')

def DivUp(a, b):
  return (a + b - 1) / b

def convUp(images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, scaleTargets=0, numGroups=1):
  """
  images - (n_images, img_w**2 * n_chans)
  filters - (n_filters, filter_w**2 * n_chans)
  targets - (n_images, n_locs**2 * n_filters)
  numModulesX - Number of filter locations along an axis. = n_locs
  paddingStart - Set to k for a k-pixel border of zeros. Usually set to 0.
  moduleStride - stride to move the filters by. 
  numImgColors - n_chans
  """
  numImages = images.shape[0]
  numFilters = filters.shape[0]

  assert targets.shape == (numImages, numFilters * numModulesX * numModulesY), '%s %d %d-%d-%d' % (targets.shape.__str__(), numImages, numFilters, numModulesX, numModulesY)

  _ConvNet.convUp(images.p_mat, filters.p_mat, targets.p_mat, ct.c_int(imgSizeY), ct.c_int(numModulesY), ct.c_int(numModulesX),
                  ct.c_int(-paddingStart), ct.c_int(moduleStride), ct.c_int(numImgColors), ct.c_int(numGroups), ct.c_float(scaleTargets))

def convDown(hidSums, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, scaleTargets=0, numGroups=1):
  """
  hidSums - (n_images, n_locs**2 * n_filters)
  filters - (n_filters, filter_w**2 * n_chans)
  targets - (n_images, img_w**2 * n_chans)
  """
  numImages = hidSums.shape[0] 

  assert paddingStart >= 0
  assert targets.shape == (numImages, numImgColors * imgSizeX * imgSizeY)

  _ConvNet.convDown(hidSums.p_mat, filters.p_mat, targets.p_mat, ct.c_int(imgSizeY), ct.c_int(imgSizeX), ct.c_int(numModulesY),
                    ct.c_int(-paddingStart), ct.c_int(moduleStride), ct.c_int(numImgColors), ct.c_int(numGroups), ct.c_float(scaleTargets))

def convOutp(images, hidSums, targets, imgSizeY, numModulesY, numModulesX, filterSize, paddingStart, moduleStride, numImgColors, scaleTargets=0, numGroups=1, partialSum=0):
  """
  images - (n_images, img_w**2 * n_chans)
  hidSums - (n_images, n_locs**2 * n_filters)
  targets - (n_filters, filter_w**2 * n_chans)
  """
  if partialSum == 0:
    partialSum = max(numModulesX, numModulesY)
  sum_locs = DivUp(numModulesX, partialSum) * DivUp(numModulesY, partialSum)
  numImages  = images.shape[0]
  numFilters = hidSums.shape[1] / (numModulesX * numModulesY)

  assert targets.shape == (numFilters, sum_locs * numImgColors * filterSize * filterSize), '%s %d %d-%d-%d-%d' % (targets.shape.__str__(), sum_locs, numFilters, numImgColors, filterSize, filterSize)
  _ConvNet.convOutp(images.p_mat, hidSums.p_mat, targets.p_mat, ct.c_int(imgSizeY), ct.c_int(numModulesY), ct.c_int(numModulesX), ct.c_int(filterSize), ct.c_int(-paddingStart), ct.c_int(moduleStride), ct.c_int(numImgColors), ct.c_int(numGroups), ct.c_int(partialSum), ct.c_float(scaleTargets), ct.c_float(1))

def localUp(images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numGroups=1, scaleTargets=0):
  """
  images - (n_images, img_w**2 * n_chans)
  filters - (n_filters, filter_w**2 * n_chans)
  targets - (n_images, n_locs**2 * n_filters)
  numModulesX - Number of filter locations along an axis. = n_locs
  paddingStart - Set to k for a k-pixel border of zeros. Usually set to 0.
  moduleStride - stride to move the filters by. 
  numImgColors - n_chans
  """
  numImages = images.shape[0]
  numFilters = filters.shape[0]

  assert targets.shape == (numImages, numFilters * numModulesX * numModulesY), '%s %d %d-%d-%d' % (targets.shape.__str__(), numImages, numFilters, numModulesX, numModulesY)

  _ConvNet.localUp(images.p_mat, filters.p_mat, targets.p_mat, imgSizeY, numModulesY, numModulesX,
                  -paddingStart, moduleStride, numImgColors, numGroups, scaleTargets)


def localDown(hidSums, filters, targets, imgSizeY, imgSizeX, numModulesY, paddingStart, moduleStride, numImgColors, numGroups=1, scaleTargets=0):
  """
  hidSums - (n_images, n_locs**2 * n_filters)
  filters - (n_filters, filter_w**2 * n_chans)
  targets - (n_images, img_w**2 * n_chans)
  """
  numImages = hidSums.shape[0] 

  assert paddingStart >= 0
  assert targets.shape == (numImages, numImgColors * imgSizeX * imgSizeY)

  _ConvNet.localDown(hidSums.p_mat, filters.p_mat, targets.p_mat, imgSizeY, imgSizeX, numModulesY,
                    -paddingStart, moduleStride, numImgColors, numGroups, scaleTargets)

def localOutp(images, hidSums, targets, imgSizeY, numModulesY, numModulesX, filterSize, paddingStart, moduleStride, numImgColors, numGroups=1, scaleTargets=0):
  """
  images - (n_images, img_w**2 * n_chans)
  hidSums - (n_images, n_locs**2 * n_filters)
  targets - (n_filters, filter_w**2 * n_chans)
  """
  numImages = images.shape[0]
  numFilters = hidSums.shape[1] / (numModulesX * numModulesY)

  assert targets.shape == (numFilters, numImgColors * filterSize * filterSize * numModulesX * numModulesY), '%s %d %d-%d-%d' % (targets.shape.__str__(), numFilters, numImgColors, filterSize, filterSize)
  _ConvNet.convOutp(images.p_mat, hidSums.p_mat, targets.p_mat, imgSizeY, numModulesY, numModulesX, filterSize, -paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, 1)

def MaxPool(images, targets, numChannels, subsX, startX, strideX, outputsX):
  """
  images - (n_images, img_w**2 * n_chans)
  numChannels - number of filter/color channels
  subsX - width of pooling area
  startX - pixel where pooling starts
  strideX - stride
  outputsX - number of pooling sites
  """
  numImages = images.shape[0]

  assert targets.shape == (numImages, numChannels * outputsX * outputsX)
  
  _ConvNet.MaxPool(images.p_mat, targets.p_mat, numChannels, subsX, startX, strideX, outputsX)

def ProbMaxPool(images, rnd, targets, numChannels, subsX, startX, strideX, outputsX):
  """
  images - (n_images, img_w**2 * n_chans)
  rnd - (n_images, img_w**2 * n_chans)
  numChannels - number of filter/color channels
  subsX - width of pooling area
  startX - pixel where pooling starts
  strideX - stride
  outputsX - number of pooling sites
  """
  numImages = images.shape[0]

  assert targets.shape == (numImages, numChannels * outputsX * outputsX)
  assert rnd.shape == images.shape

  raise Exception('Not implemented.')
  """
  _ConvNet.ProbMaxPool(images.p_mat, rnd.p_mat, targets.p_mat,
           numChannels, subsX, startX, strideX, outputsX)
  """


def MaxPoolUndo(images, targets, grad, maxes,
        subsX, startX, strideX, outputsX):
  """
  images - (n_images, img_w**2 * n_chans)
  grad - (n_images, outputsX**2 * n_chans) cudamat of deltas/gradients of loss wrt layer outputs.
  maxes - (n_images, outputsX**2 * n_chans) cudamat of layer outputs.
  subsX - width of pooling area
  startX - pixel where pooling starts
  strideX - stride
  outputsX - number of pooling sites
  """
  assert targets.shape == images.shape

  _ConvNet.MaxPoolUndo(images.p_mat, grad.p_mat, maxes.p_mat, targets.p_mat, subsX, startX, strideX, outputsX)

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
