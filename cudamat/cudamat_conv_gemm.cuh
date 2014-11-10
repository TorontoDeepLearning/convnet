/** Kernels for convUp, convDown, convOutp, maxpool, avgpool, maxpoolundo,
 *  avgpoolundo.
 *  These kernels are 10-20% slower than cuda-convnet2, but have no constraints
 *  on number of channels and support rectangular images and rectangular kernels.
 *  They use cublasSgemm for convUp, convDown, convOutp.
 *  Data layout : Column-major
 *  data : (num_images, image_size_x, image_size_y, num_input_channels)
 *  filters : (num_output_channels, kernel_size_x, kernel_size_y, num_input_channels)
 */
#ifndef CUDAMAT_CONV_GEMM_CUH_
#define CUDAMAT_CONV_GEMM_CUH_
#include "cudamat.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include <math.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif
void convUpGemm(cudamat* images, cudamat* filters, cudamat* targets,
                Shape4D* images_shape, Shape4D* filters_shape,
                Shape4D* targets_shape, ConvDesc conv_desc,
                float scaleTargets);

void convDownGemm(cudamat* derivs, cudamat* filters, cudamat* targets,
                  Shape4D* derivs_shape, Shape4D* filters_shape,
                  Shape4D* targets_shape, ConvDesc conv_desc,
                  float scaleTargets);

void convOutpGemm(cudamat* images, cudamat* derivs, cudamat* targets,
                  Shape4D* images_shape, Shape4D* derivs_shape,
                  Shape4D* targets_shape, ConvDesc conv_desc,
                  float scaleTargets, float scaleOutput);

void localUpGemm(cudamat* images, cudamat* filters, cudamat* targets,
                Shape4D* images_shape, Shape4D* filters_shape,
                Shape4D* targets_shape, ConvDesc conv_desc,
                float scaleTargets);

void localDownGemm(cudamat* derivs, cudamat* filters, cudamat* targets,
                  Shape4D* derivs_shape, Shape4D* filters_shape,
                  Shape4D* targets_shape, ConvDesc conv_desc,
                  float scaleTargets);

void localOutpGemm(cudamat* images, cudamat* derivs, cudamat* targets,
                  Shape4D* images_shape, Shape4D* derivs_shape,
                  Shape4D* targets_shape, ConvDesc conv_desc,
                  float scaleTargets, float scaleOutput);


void MaxPoolGemm(cudamat* images, cudamat* targets, Shape4D* images_shape,
                 Shape4D* targets_shape, ConvDesc conv_desc, float scaleTargets,
                 float scaleOutput);

void MaxPoolUndoGemm(cudamat* images, cudamat* maxGrads, cudamat* maxActs,
                     cudamat* targets, Shape4D* images_shape,
                     Shape4D* maxGrads_shape, ConvDesc conv_desc,
                     float scaleTargets);

void AvgPoolGemm(cudamat* images, cudamat* targets, Shape4D* images_shape,
                 Shape4D* targets_shape, ConvDesc conv_desc, float scaleTargets,
                 float scaleOutput);

void AvgPoolUndoGemm(cudamat* avgGrads, cudamat* targets,
                     Shape4D* avgGrads_shape, Shape4D* targets_shape,
                     ConvDesc conv_desc, float scaleTargets);


void UpSampleGemm(cudamat* images, cudamat* targets, Shape4D* images_shape,
              Shape4D* targets_shape, int factor, float scaleTargets);
 
void DownSampleGemm(cudamat* images, cudamat* targets, Shape4D* images_shape,
                Shape4D* targets_shape, int factor);

void ResponseNormCrossMapGemm(
  cudamat* images, cudamat* targets, int numFilters, int sizeF, float addScale,
  float powScale, bool blocked);

void ResponseNormCrossMapUndoGemm(
  cudamat* outGrads, cudamat* inputs, cudamat* targets,
  int numFilters, int sizeF, float addScale, float powScale, bool blocked);


#ifdef __cplusplus
}
#endif
#endif
