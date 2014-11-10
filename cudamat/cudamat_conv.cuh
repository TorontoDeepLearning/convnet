#ifndef CUDAMAT_CONV_CUH_
#define CUDAMAT_CONV_CUH_
#include "cudamat.cuh"
#ifdef __cplusplus
extern "C" {
#endif

void SetupTexture(cudamat* mat);

void convUp(cudamat* images, cudamat* filters, cudamat* targets,
            Shape4D* images_shape, Shape4D* filters_shape, Shape4D* targets_shape,
            ConvDesc conv_desc, float scaleTargets);

void localUp(cudamat* images, cudamat* filters, cudamat* targets,
             Shape4D* images_shape, Shape4D* filters_shape, Shape4D* targets_shape,
             ConvDesc conv_desc, float scaleTargets);

void convDown(cudamat* derivs, cudamat* filters, cudamat* targets,
              Shape4D* derivs_shape, Shape4D* filters_shape, Shape4D* targets_shape,
              ConvDesc conv_desc, float scaleTargets);

void localDown(cudamat* derivs, cudamat* filters, cudamat* targets,
               Shape4D* derivs_shape, Shape4D* filters_shape, Shape4D* targets_shape,
               ConvDesc conv_desc, float scaleTargets);

void convOutp(cudamat* images, cudamat* derivs, cudamat* targets,
              Shape4D* images_shape, Shape4D* derivs_shape, Shape4D* targets_shape,
              ConvDesc conv_desc, int partialSumY, int partialSumX, float scaleTargets,
              float scaleOutput);

void localOutp(cudamat* images, cudamat* derivs, cudamat* targets,
               Shape4D* images_shape, Shape4D* derivs_shape, Shape4D* targets_shape,
               ConvDesc conv_desc, float scaleTargets, float scaleOutput);

void ResponseNormCrossMap(cudamat* images, cudamat* targets,
                          int numFilters, int sizeF, float addScale,
                          float powScale, bool blocked);

void ResponseNormCrossMapUndo(cudamat* outGrads,
                              cudamat* inputs, cudamat* acts, cudamat* targets,
                              int numFilters, int sizeF, float addScale,
                              float powScale, bool blocked);

void ResponseNorm(cudamat* images, cudamat* denoms, cudamat* targets,
                  int numFilters, int sizeX, float addScale, float powScale);

void ResponseNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* inputs,
                      cudamat* acts, cudamat* targets, int numFilters,
                      int sizeX, float addScale, float powScale);

void ContrastNorm(cudamat* images, cudamat* meanDiffs, cudamat* denoms,
                  cudamat* targets, int numFilters, int sizeX, float addScale,
                  float powScale);

void ContrastNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* meanDiffs,
                      cudamat* acts, cudamat* targets, int numFilters,
                      int sizeX, float addScale, float powScale);

void MaxPool(cudamat* images, cudamat* targets, Shape4D* images_shape,
             Shape4D* targets_shape, ConvDesc conv_desc);

void AvgPool(cudamat* images, cudamat* targets, Shape4D* images_shape,
             Shape4D* targets_shape, ConvDesc conv_desc);

void MaxPoolUndo(cudamat* images, cudamat* maxGrads, cudamat* maxActs,
                 cudamat* targets, Shape4D* images_shape, Shape4D* maxGrads_shape,
                 ConvDesc conv_desc, float scaleTargets);

void AvgPoolUndo(cudamat* avgGrads, cudamat* targets, Shape4D* avgGrads_shape,
                 Shape4D* targets_shape, ConvDesc conv_desc, float scaleTargets);

void UpSample(cudamat* images, cudamat* targets, Shape4D* images_shape,
              Shape4D* targets_shape, int factor, float scaleTargets);
 
void DownSample(cudamat* images, cudamat* targets, Shape4D* images_shape,
                Shape4D* targets_shape, int factor);

void RGBToYUV(cudamat* images, cudamat* targets);
#ifdef __cplusplus
}
#endif
#endif
