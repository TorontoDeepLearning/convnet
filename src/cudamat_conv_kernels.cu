#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include <math.h>
#include <assert.h>
#include "cudamat_conv_kernels.cuh"

__device__ float square(const float a) {
  return a*a;
}

__global__ void filterActs_YxX_oneoutput(
    float* images, float* filters, float* targets,
    const int numImages, const int imgSize, const int filterSize, const int paddingStart,
    const int moduleStride, const int numModulesX, const int numImgColors,
    const float scaleTargets, const float scaleOutputs) {

  // Each block does one module. Each thread handles a different image.

  extern __shared__ float shFilter[];
  const int numThreads = blockDim.x;
  const int blockId = blockIdx.x;
  const int moduleIdx = blockId % numModulesX;
  const int moduleIdy = blockId / numModulesX;

  const int filterPixels = filterSize * filterSize; 
  const int imgPixels = imgSize * imgSize; 
  const int y_start = moduleIdy * moduleStride + paddingStart;
  const int x_start = moduleIdx * moduleStride + paddingStart;
  const int tid = threadIdx.x; // image_id


  if (tid < numImages) {
    float res = 0;
    int x, y, x_pos, y_pos;
    for (int c = 0; c < numImgColors; c++) {
      const float* img_c = &images[tid + c * imgPixels * numImages];
      const float* filters_c = &filters[c * filterPixels];

      // Load shared filter.
      #pragma unroll
      for (int i = tid; i < filterPixels; i+=numThreads) {
        shFilter[i] = filters_c[i];
      }
      __syncthreads();
      #pragma unroll
      for (int p = 0; p < filterPixels; p++) {
        y = p / filterSize;
        x = p % filterSize;
        y_pos = y + y_start;
        x_pos = x + x_start;
        if (y_pos >= 0 && y_pos < imgSize && x_pos >= 0 && x_pos < imgSize) {
          res += shFilter[p] * img_c[numImages * (y_pos * imgSize + x_pos)];
        }
      }
      __syncthreads();
    }
    int pos = (moduleIdy * numModulesX + moduleIdx) * numImages + tid;
    targets[pos] = scaleTargets * targets[pos] + scaleOutputs * res;
  }
}

__global__ void conv_img_acts_onefilter(const float* hidActs, const float* filters, float* targets,
                     const int numModulesX, const int numImages, const int numFilters,
                     const int filterSize, const int imgSize, const int paddingStart, const int moduleStride,
                     const int numImgColors, const float scaleTargets, const float scaleOutputs) {
  const int blockId = blockIdx.x;
  const int img_x = blockId % imgSize;
  const int img_y = blockId / imgSize;

  const int filterPixels = filterSize * filterSize; 
  const int imgPixels = imgSize * imgSize; 
  const int tid = threadIdx.x; // image_id

  const int module_y_start = MAX(0, (img_y - filterSize + 1 - paddingStart) / moduleStride);
  const int module_y_end = MIN(numModulesX - 1, (img_y - paddingStart) / moduleStride);
  const int module_x_start = MAX(0, (img_x - filterSize + 1 - paddingStart) / moduleStride);
  const int module_x_end = MIN(numModulesX - 1, (img_x - paddingStart) / moduleStride);

  if (tid < numImages) {
    float* target = &targets[tid + (img_y * imgSize + img_x) * numImages];
    int filter_y, filter_x;
    for (int c = 0; c < numImgColors; c++) {
      float res = 0;
      const float* filters_c = &filters[filterPixels * c];
      const float* hid = &hidActs[tid];
      for (int module_y = module_y_start; module_y <= module_y_end; module_y++) {
        for (int module_x = module_x_start; module_x <= module_x_end; module_x++) {
          filter_y = img_y - module_y * moduleStride - paddingStart;
          filter_x = img_x - module_x * moduleStride - paddingStart;
          if (filter_x >= 0 && filter_x < filterSize && filter_y >= 0 && filter_y < filterSize) {
            float h = hid[numImages * (module_y * numModulesX + module_x)];
            res += h * filters_c[filter_y * filterSize + filter_x];
          }
        }
      }
      target[c * imgPixels * numImages] = scaleTargets * target[c * imgPixels * numImages] + scaleOutputs * res;
    }
  }
}

//TODO: Templatize to make unroll work.
__global__ void conv_weight_acts_oneoutput(
    float* images, float* hidActs, float* targets,
    const int numImages, const int numModulesX,
    const int imgSize, const int filterSize,
    const int paddingStart, const int moduleStride, const int imgStride,
    const float scaleTargets, const float scaleOutputs) {

  int blockId = blockIdx.x;
  const int pixel_x = blockId % filterSize; blockId /= filterSize;
  const int pixel_y = blockId % filterSize;
  const int input_color = blockId / filterSize;

  const int numModules = numModulesX * numModulesX;
  const int imgPixels = imgSize * imgSize; 
  const int tid = threadIdx.x; // image_id

  extern __shared__ float output[];

  if (tid < numImages) {
    float res = 0;
    const float* img = &images[tid + input_color * imgPixels * numImages];
    const float* hid = &hidActs[tid];
    int x, y, img_pixel_x, img_pixel_y, img_pixel;
    #pragma unroll
    for (int hid_pixel = 0; hid_pixel < numModules; hid_pixel++) {
      y = hid_pixel / numModulesX;
      x = hid_pixel % numModulesX;
      img_pixel_y = y * moduleStride + paddingStart + pixel_y;
      img_pixel_x = x * moduleStride + paddingStart + pixel_x;
      img_pixel = img_pixel_y * imgSize + img_pixel_x;
      if (img_pixel_x >= 0 && img_pixel_x < imgSize && img_pixel_y >=0 && img_pixel_y < imgSize) {
        res += hid[numImages * hid_pixel] * img [numImages * img_pixel];
      }
    }
    output[tid] = res;
    __syncthreads();
    if (tid == 0) {  // Do better than this.
      float sum = 0;
      for (int i = 0 ; i < blockDim.x; i++) sum += output[i];
      targets[blockIdx.x] = scaleTargets * targets[blockIdx.x] + scaleOutputs * sum;
    }
  }
}


/*
 * Block size 1x128
 * blockIdx.x determines pixel.x, image idx in batches of 128*imgsPerThread
 * blockIdx.y determines pixel.y
 * 
 * So each block does one output for some number of images and all the fliters.
 * 
 * threadIdx.x determines img idx
 * 
 * imgs:    (numFilters, imgPixels, numImages)
 * meanDiffs:  (numFilters, imgPixels, numImages)
 * denoms:   (numFilters, imgPixels, numImages) (out)
 * target:   (numFilters, imgPixels, numImages) (out)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int imgsPerThread, int numFilters, bool checkCaseBounds>
__global__ void kCNorm_fewfilter(float* imgs, float* meanDiffs, float* denoms, float* target, const int imgSize,
                 const int numImages, const int sizeX, const float addScale, const float powScale) {

  const int imgPixels = imgSize * imgSize;
  const int numImgBlocks = DIVUP(numImages, 128*imgsPerThread);
  const int pxIdxX = blockIdx.x / numImgBlocks;
  const int pxIdxY = blockIdx.y;
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * 128 * imgsPerThread;
  
  const int pxIdx = pxIdxY * imgSize + pxIdxX;
  
  const int startPxX = -sizeX/2 + pxIdxX;
  const int startPxY = -sizeX/2 + pxIdxY;
  const int imgIdx = blockImgIdx + threadIdx.x;
  
  imgs += pxIdx * numImages + imgIdx;
  denoms += pxIdx * numImages + imgIdx;
  meanDiffs += imgIdx;
  target += pxIdx * numImages + imgIdx;
  
  float prod[numFilters][imgsPerThread];
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
      #pragma unroll
      for (int f = 0; f < numFilters; f++) {
        prod[f][i] = 0; 
      }
    }
  }
  const int loopStartY = MAX(0, startPxY);
  const int loopStartX = MAX(0, startPxX);
  const int loopEndY = MIN(imgSize, startPxY + sizeX);
  const int loopEndX = MIN(imgSize, startPxX + sizeX);
    
  for (int y = loopStartY; y < loopEndY; y++) {
    for (int x = loopStartX; x < loopEndX; x++) {
      const int imgPx = y * imgSize + x;
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
          #pragma unroll
          for (int f = 0; f < numFilters; f++) {
            prod[f][i] += square(meanDiffs[(f * imgPixels + imgPx) * numImages + i * 128]);
          }
        }
      }
    }
  }
  
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
      #pragma unroll
      for (int f = 0; f < numFilters; f++) {
        prod[f][i] = 1 + addScale * prod[f][i];
        denoms[f * imgPixels * numImages + i * 128] = prod[f][i];
        target[f * imgPixels * numImages + i * 128] = imgs[f * imgPixels * numImages + i * 128] * __powf(prod[f][i], -powScale);
      }
    }
  }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:    (numFilters, imgPixels, numImages)
 * means:    (numFilters, imgPixels, numImages)
 * denoms:   (numFilters, imgPixels, numImages) (out)
 * target:   (numFilters, imgPixels, numImages) (out)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kCNorm_manyfilter(float* imgs, float* meanDiffs, float* denoms, float* target, const int imgSize,
                 const int numFilters, const int numImages, const int sizeX, const float addScale, const float powScale) {
  const int imgPixels = imgSize * imgSize;
  const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
  const int numFilterBlocks = numFilters/(B_Y*filtersPerThread);
  const int pxIdxX = blockIdx.x / numImgBlocks;
  const int pxIdxY = blockIdx.y / numFilterBlocks;
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
  
  const int pxIdx = pxIdxY * imgSize + pxIdxX;
  
  const int startPxX = -sizeX/2 + pxIdxX;
  const int startPxY = -sizeX/2 + pxIdxY;
  const int imgIdx = blockImgIdx + threadIdx.x;
  
  imgs += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
  meanDiffs += (blockFilterIdx + threadIdx.y) * imgPixels * numImages + imgIdx;
  denoms += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
  target += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
      #pragma unroll
      for (int f = 0; f < filtersPerThread; f++) {
        prod[f][i] = 0;
      }
    }
  }

  const int loopStartY = MAX(0, startPxY);
  const int loopStartX = MAX(0, startPxX);
  const int loopEndY = MIN(imgSize, startPxY + sizeX);
  const int loopEndX = MIN(imgSize, startPxX + sizeX);
  
  for (int y = loopStartY; y < loopEndY; y++) {
    for (int x = loopStartX; x < loopEndX; x++) {
      const int imgPx = y * imgSize + x;
      #pragma unroll
      for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
          #pragma unroll
          for (int f = 0; f < filtersPerThread; f++) {
            prod[f][i] += square(meanDiffs[(f * B_Y * imgPixels + imgPx) * numImages + i * B_X]);
          }
        }
      }
    }
  }
  
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
      #pragma unroll
      for (int f = 0; f < filtersPerThread; f++) {
        prod[f][i] = 1 + addScale * prod[f][i];
        denoms[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
        target[f * B_Y * imgPixels * numImages + i * B_X] = imgs[f * B_Y * imgPixels * numImages + i * B_X] * __powf(prod[f][i], -powScale);
      }
    }
  }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y
 * 
 * So each block does one pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y
 */
template<int B_Y, int B_X, int imgsPerThread, bool checkCaseBounds, bool blocked>
__global__ void kFCNorm(float* imgs, float* meanDiffs, float* denoms, float* target, const int imgSize,
                                  const int numFilters, const int numImages, const int sizeF, 
                                  const float addScale, const float powScale) {
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/B_Y;
    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int filterIdx = (blockIdx.y % numFilterBlocks) * B_Y + threadIdx.y;
    
    const int pxIdx = pxIdxY * imgSize + pxIdxX;

    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    meanDiffs += pxIdx * numImages + imgIdx;
    denoms += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    target += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    
    float prod[imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            prod[i] = 0;
        }
    }

    const int startF = blocked ? (filterIdx / sizeF) * sizeF : -sizeF/2 + filterIdx;
    const int loopStartF = blocked ? startF : MAX(0, startF);
    const int loopEndF = MIN(numFilters, startF + sizeF);
 
    for (int f = loopStartF; f < loopEndF; ++f) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                prod[i] += square(meanDiffs[f * imgPixels * numImages + i * B_X]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            prod[i] = 1 + addScale * prod[i];
            denoms[i * B_X] = prod[i];
            target[i * B_X] = imgs[i * B_X] * __powf(prod[i], -powScale);
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y
 * 
 * TODO: this isn't really ideal
 */
template<int B_Y, int B_X, int imgsPerThread, bool add, bool checkCaseBounds, bool blocked>
__global__ void kFRNormUndo(float* outGrads, float* denoms, float* inputs, float* acts, float* target, const int imgSize, const int numFilters,
                            const int numImages, const int sizeF, const float powScale, const float scaleTargets, const float scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/B_Y;
    
    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int filterIdx = (blockIdx.y % numFilterBlocks) * B_Y + threadIdx.y;
    
    const int imgPixels = imgSize * imgSize;
    const int pxIdx = pxIdxY * imgSize + pxIdxX;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    acts        += pxIdx * numImages + imgIdx;
    inputs      += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    denoms      += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    outGrads    += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    target      += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    
    float prod[imgsPerThread];
//    if (imgIdx != 0 || pxIdx != 0 || filterIdx != 0) {
//        return;
//    }
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        prod[i] = 0;
    }
    
    const int startF = blocked ? (filterIdx / sizeF) * sizeF : -sizeF + sizeF/2 + 1 + filterIdx;
    const int loopStartF = blocked ? startF : MAX(0, startF);
    const int loopEndF = MIN(numFilters, startF + sizeF);
    
    for (int f = loopStartF; f < loopEndF; ++f) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                prod[i] += acts[f * imgPixels * numImages + i * B_X];
            }
        }
    }
//    printf("gpu f start: %d, end: %d\n", loopStartF, loopEndF);

    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                const float inp = inputs[i * B_X];
                const float out = outGrads[i * B_X];
                const float den = denoms[i * B_X];
                prod[i] = inp * prod[i] + out * __powf(den, -powScale);
                target[i * B_X] = prod[i];
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                const float inp = inputs[i * B_X];
                const float out = outGrads[i * B_X];
                const float den = denoms[i * B_X];
                prod[i] = inp * prod[i] + out * __powf(den, -powScale);
                target[i * B_X] = scaleTargets * target[i * B_X] + scaleOutputs * prod[i];
            }
        }
    }
}


/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does 4x4 region of pixels for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 * 
 * imgs:    (numFilters, imgPixels, numImages)
 * means:    (numFilters, imgPixels, numImages)
 * denoms:   (numFilters, imgPixels, numImages) (out)
 * target:   (numFilters, imgPixels, numImages) (out)
 * 
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 * 
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by filtersPerThread
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 */
template<int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kCNorm2(float* imgs, float* meanDiffs, float* denoms, float* target, const int imgSize,
             const int numFilters, const int numImages, const int sizeX, const float addScale, const float powScale) {
  __shared__ float shDiffs[filtersPerThread][B_X*imgsPerThread];
  const int imgPixels = imgSize * imgSize;
  const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
  const int numFilterBlocks = numFilters/(filtersPerThread);
  const int blockPxX = 4*(blockIdx.x / numImgBlocks);
  const int blockPxY = 4*(blockIdx.y / numFilterBlocks);
  const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
  const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
  
  const int tidx = threadIdx.y * B_X + threadIdx.x;
  const int loadY = tidx / 32, loadX = tidx % 32;
  
  const int startPxX = MAX(0, -sizeX/2 + blockPxX);
  const int startPxY = MAX(0, -sizeX/2 + blockPxY);
  const int endPxX = MIN(imgSize, blockPxX + DIVUP(sizeX, 2) + 3);
  const int endPxY = MIN(imgSize, blockPxY + DIVUP(sizeX, 2) + 3);
  
  const int myPxX = blockPxX + threadIdx.y % 4;
  const int myPxY = blockPxY + threadIdx.y / 4;
  const int myPxIdx = myPxY * imgSize + myPxX;
//  const bool doWork = myPxX < imgSize && myPxY < imgSize;
  const int myStartPxY = -sizeX/2 + myPxY;
  const int myStartPxX = -sizeX/2 + myPxX;
  const int myEndPxY = myPxY + DIVUP(sizeX, 2);
  const int myEndPxX = myPxX + DIVUP(sizeX, 2);
  
  const int imgIdx = blockImgIdx + threadIdx.x;
    
  imgs    += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
  meanDiffs  += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
  denoms   += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
  target   += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
  
  float prod[filtersPerThread][imgsPerThread];
  #pragma unroll
  for (int i = 0; i < imgsPerThread; i++) {
    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
      #pragma unroll
      for (int f = 0; f < filtersPerThread; f++) {
        prod[f][i] = 0;
      }
    }
  }

  for (int y = startPxY; y < endPxY; y++) {
    const bool isInY = y >= myStartPxY && y < myEndPxY;
    for (int x = startPxX; x < endPxX; x++) {
      const int px = y * imgSize + x;
      // All the threads load a pixel from memory
      #pragma unroll
      for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
        if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
          #pragma unroll
          for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
            if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
              shDiffs[ly + loadY][lx + loadX] = meanDiffs[(ly * imgPixels + px) * numImages + lx];
            }
          }
        }
      }
      __syncthreads();
      
      // Each row of threads decides if it's interested in this pixel
      if (isInY && x >= myStartPxX && x < myEndPxX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
          if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
              prod[f][i] += square(shDiffs[f][threadIdx.x + i * B_X]);
            }
          }
        }
      }
      __syncthreads();
    }
  }
//  imgs -= (loadY * imgPixels - myPxIdx) * numImages + loadX;
//  imgs += threadIdx.x;
  if (myPxX < imgSize && myPxY < imgSize) {
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
      if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
        #pragma unroll
        for (int f = 0; f < filtersPerThread; f++) {
          prod[f][i] = 1 + addScale * prod[f][i];
          denoms[f * imgPixels * numImages + i * B_X] = prod[f][i];
          target[f * imgPixels * numImages + i * B_X] = imgs[f * imgPixels * numImages + i * B_X] * __powf(prod[f][i], -powScale);
        }
      }
    }
  }
}


/*
 * images:      (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 
 * Note: at present, I have no code to compute the meanDiffs. So it should be set 
 * to be equal to images. In other words, this isn't really doing contrast normalization,
 * just response normalization.
 */
void convContrastNormCrossMap(cudamat* images, cudamat* meanDiffs, cudamat* denoms, cudamat* target,
                             int numFilters, int sizeF, float addScale, float powScale, bool blocked) {
    int numImages = images->size[0];
    int imgPixels = images->size[1] / numFilters;
    assert(images->size[1] == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    //assert(meanDiffs.isSameDims(images));
    assert(sizeF > 0 && sizeF <= numFilters);
    
    assert(!meanDiffs->is_trans);
    assert(!images->is_trans);
    //assert(images.isContiguous());
    //assert(meanDiffs.isContiguous());
    assert(numFilters % 16 == 0);

    //target.resize(images); assuming these shapes are same..
    //denoms.resize(images);
    //assert(target.isContiguous());

    bool checkCaseBounds = numImages % 128 != 0;
        
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / 4) * imgSize);
    if (blocked) {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, true, true>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, true, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, sizeF, addScale, powScale);
        } else {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, false, true>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, false, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, sizeF, addScale, powScale);
        }
    } else {
    if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, true, false>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, true, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, sizeF, addScale, powScale);
        } else {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, false, false>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, false, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, sizeF, addScale, powScale);
        }
    }
    getLastCudaError("convContrastNormCrossMap: kernel execution failed");
}


void convResponseNormCrossMap(cudamat* images, cudamat* denoms, cudamat* target, int numFilters, int sizeF, float addScale, float powScale, bool blocked) {
    convContrastNormCrossMap(images, images, denoms, target, numFilters, sizeF, addScale, powScale, blocked);
}

/*
 * outGrads:    (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages)
 * inputs:      (numFilters, imgPixels, numImages)
 * acts:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * THIS WILL OVERWRITE THE ACTS MATRIX.
 */
void convResponseNormCrossMapUndo(cudamat* outGrads, cudamat* denoms, cudamat* inputs, cudamat* acts, cudamat* target, int numFilters,
                         int sizeF, float addScale, float powScale, bool blocked, float scaleTargets, float scaleOutput) {
    int numImages = outGrads->size[0];
    int imgPixels = outGrads->size[1] / numFilters;

    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    assert(sizeF > 0 && sizeF <= numFilters);
    assert(outGrads->size[1] == numFilters * imgPixels);
    
    //assert(denoms.isSameDims(outGrads));
    //assert(acts.isSameDims(denoms));
    assert(!denoms->is_trans);
    assert(!outGrads->is_trans);
    assert(!acts->is_trans);
    assert(!target->is_trans);
    //assert(outGrads.isContiguous());
    
    assert(numFilters % 16 == 0);
    
    //target.resize(outGrads);
    //assert(target.isContiguous());
    // First do acts := -2 x scale x acts x outGrads / denoms
    // so that the main routine only has to do an addition in its inner loop.
    int prelimEltsPerThread = 4;
    dim3 threads(128);
    const int num_els = outGrads->size[0] * outGrads->size[1];
    dim3 blocks(MIN(512, DIVUP(num_els, (threads.x * prelimEltsPerThread))));
    kRNormUndoPrelims<128, 4><<<blocks, threads>>>(acts->data_device, denoms->data_device, outGrads->data_device, num_els, -2*addScale*powScale);
   
    // Now the main routine
    
    dim3 threads2 = dim3(32, 4);
    dim3 blocks2 = dim3(DIVUP(numImages,32*4) * imgSize, (numFilters / 4) * imgSize);
    bool checkCaseBounds = (numImages % 128) != 0;
    if (blocked) {
        if (scaleTargets == 0 && scaleOutput == 1) { 
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo<4, 32, 4, false, true, true>, cudaFuncCachePreferL1);
                kFRNormUndo<4, 32, 4, false, true, true><<<blocks2, threads2>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, powScale,
                                                                        scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo<4, 32, 4, false, false, true>, cudaFuncCachePreferL1);
                kFRNormUndo<4, 32, 4, false, false, true><<<blocks2, threads2>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, powScale,
                                                                        scaleTargets, scaleOutput);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo<4, 32, 4, true, true, true>, cudaFuncCachePreferL1);
                kFRNormUndo<4, 32, 4, true, true, true><<<blocks2, threads2>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, powScale,
                                                                        scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo<4, 32, 4, true, false, true>, cudaFuncCachePreferL1);
                kFRNormUndo<4, 32, 4, true, false, true><<<blocks2, threads2>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, powScale,
                                                                        scaleTargets, scaleOutput);
            }
        }
    } else {
        if (scaleTargets == 0 && scaleOutput == 1) { 
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo<4, 32, 4, false, true, false>, cudaFuncCachePreferL1);
                kFRNormUndo<4, 32, 4, false, true, false><<<blocks2, threads2>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, powScale,
                                                                        scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo<4, 32, 4, false, false, false>, cudaFuncCachePreferL1);
                kFRNormUndo<4, 32, 4, false, false, false><<<blocks2, threads2>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, powScale,
                                                                        scaleTargets, scaleOutput);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo<4, 32, 4, true, true, false>, cudaFuncCachePreferL1);
                kFRNormUndo<4, 32, 4, true, true, false><<<blocks2, threads2>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, powScale,
                                                                        scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo<4, 32, 4, true, false, false>, cudaFuncCachePreferL1);
                kFRNormUndo<4, 32, 4, true, false, false><<<blocks2, threads2>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, powScale,
                                                                        scaleTargets, scaleOutput);
            }
        }
    }
    getLastCudaError("convResponseNormCrossMapUndo: kernel execution failed");
}


void convContrastNorm(cudamat* images, cudamat* meanDiffs, cudamat* denoms, cudamat* target, int numFilters, int sizeX, float addScale, float powScale) {
  int numImages = images->size[0];
  int imgPixels = images->size[1] / numFilters;
  assert(images->size[1] == numFilters * imgPixels);
  int imgSize = int(sqrt(imgPixels));
  assert(imgSize * imgSize == imgPixels);
  //assert(meanDiffs.isSameDims(images));
  
  //assert(!meanDiffs.isTrans());
  //assert(!images.isTrans());
  //assert(images.isContiguous());
  //assert(meanDiffs.isContiguous());
  assert(numFilters % 16 == 0 || numFilters <= 8);

  //target.resize(images);
  //denoms.resize(images);

  if (sizeX >= 6 && numFilters % 4 == 0) {
    // This one is faster for large regions (my tests show regions >= 6...)
    int imgsPerThread = 8;
    int filtersPerThread = 4;
    int bx = 8;
    bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
    assert((imgsPerThread * bx) % 32 == 0);
    assert(numFilters % filtersPerThread == 0);
    dim3 threads(bx, 16);
    dim3 blocks(DIVUP(imgSize, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(imgSize, 4) * numFilters / filtersPerThread);

    if (checkCaseBounds) {
      cudaFuncSetCacheConfig(kCNorm2<8, 8, 4, true>, cudaFuncCachePreferL1); // L1 faster here
      kCNorm2<8, 8, 4, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                            imgSize, numFilters, numImages, sizeX, addScale, powScale);
    } else {
      cudaFuncSetCacheConfig(kCNorm2<8, 8, 4, false>, cudaFuncCachePreferL1); // L1 faster here
      kCNorm2<8, 8, 4, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                             imgSize, numFilters, numImages, sizeX, addScale, powScale);
    }
  } else {
    bool checkCaseBounds = numImages % 128 != 0;
    if (numFilters <= 8) {
      dim3 threads(128);
      dim3 blocks(DIVUP(numImages,128) * imgSize, imgSize);
      if (numFilters == 1) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 1, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 1, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 1, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 1, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 2) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 2, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 2, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 3) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 3, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 3, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 4) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 4, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 4, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 5) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 5, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 5, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 6) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 6, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 6, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 7) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 7, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 7, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } else if (numFilters == 8) {
        if (checkCaseBounds) {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, true>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 8, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        } else {
          cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, false>, cudaFuncCachePreferL1);
          kCNorm_fewfilter<1, 8, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                   imgSize, numImages, sizeX, addScale, powScale);
        }
      } 
    } else {
      dim3 threads(32, 4);
      dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 2)) * imgSize);
      if (checkCaseBounds) {
        cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, true>, cudaFuncCachePreferL1);
        kCNorm_manyfilter<4, 32, 4, 2, true><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                 imgSize, numFilters, numImages, sizeX, addScale, powScale);
      } else {
        cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, false>, cudaFuncCachePreferL1);
        kCNorm_manyfilter<4, 32, 4, 2, false><<<blocks, threads>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                 imgSize, numFilters, numImages, sizeX, addScale, powScale);
      }
    }
  }
  getLastCudaError("convResponseNorm: kernel execution failed");
}

/*
 * imgs:        (3, imgPixels, numImages) with given imgStride
 * target:      (3, imgPixels, numImages)
 */
void convRGBToYUV(cudamat* images, cudamat* target) {
    assert(!images->is_trans);
    assert(!target->is_trans);
    int imgPixels = images->size[1] / 3;
    int numImages = images->size[0];
    assert(images->size[1] == 3 * imgPixels);
    
    assert(target->size[1] == images->size[1] && target->size[0] == images->size[0]);
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    int imageStride = numImages;  // images.getStride();

    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages, imgsPerThread * 32), DIVUP(imgPixels, 4));
    if (imgsPerThread == 4) {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kRGBToYUV<4, true>, cudaFuncCachePreferL1);
            kRGBToYUV<4, true><<<blocks, threads>>>(images->data_device, target->data_device, imgPixels, numImages, imageStride);
        } else {
            cudaFuncSetCacheConfig(kRGBToYUV<4, false>, cudaFuncCachePreferL1);
            kRGBToYUV<4, false><<<blocks, threads>>>(images->data_device, target->data_device, imgPixels, numImages, imageStride);
        }
    } else if (imgsPerThread == 2) {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kRGBToYUV<2, true>, cudaFuncCachePreferL1);
            kRGBToYUV<2, true><<<blocks, threads>>>(images->data_device, target->data_device, imgPixels, numImages, imageStride);
        } else {
            cudaFuncSetCacheConfig(kRGBToYUV<2, false>, cudaFuncCachePreferL1);
            kRGBToYUV<2, false><<<blocks, threads>>>(images->data_device, target->data_device, imgPixels, numImages, imageStride);
        }
    } else {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kRGBToYUV<1, true>, cudaFuncCachePreferL1);
            kRGBToYUV<1, true><<<blocks, threads>>>(images->data_device, target->data_device, imgPixels, numImages, imageStride);
        } else {
            cudaFuncSetCacheConfig(kRGBToYUV<1, false>, cudaFuncCachePreferL1);
            kRGBToYUV<1, false><<<blocks, threads>>>(images->data_device, target->data_device, imgPixels, numImages, imageStride);
        }
    }
    getLastCudaError("convRGBToYUV: kernel execution failed");
}

