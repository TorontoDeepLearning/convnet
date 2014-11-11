import ctypes as ct
import math
import pdb
_ConvNet = ct.cdll.LoadLibrary('libcudamat_conv_gemm.so')

def DivUp(a, b):
  return (a + b - 1) / b

def convUp(images, filters, targets, conv_desc, scaleTargets=0):
  _ConvNet.convUpGemm(images.p_mat, filters.p_mat, targets.p_mat,
                  images.p_shape4d, filters.p_shape4d, targets.p_shape4d,
                  conv_desc, ct.c_float(scaleTargets))

def convDown(hidSums, filters, targets, conv_desc, scaleTargets=0):
  _ConvNet.convDownGemm(hidSums.p_mat, filters.p_mat, targets.p_mat,
                    hidSums.p_shape4d, filters.p_shape4d, targets.p_shape4d,
                    conv_desc, ct.c_float(scaleTargets))

def convOutp(images, hidSums, targets, conv_desc, scaleTargets=0, partialSumY=0, partialSumX=0, temp=None):
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

