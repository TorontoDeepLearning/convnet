#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include <math.h>
#include <assert.h>

#include "cudamat_conv_kernels.cuh"

extern "C" {
#include "cudamat_conv.cuh"
/*
 * images:   (numImgColors, imgPixels, numImages)
 * filters:   (numFilterColors, filterPixels, numFilters)
 * targets:   (numFilters, numModules, numImages)
 */

void filterActs(cudamat* images, cudamat* filters, cudamat* targets,
          int numModulesX, int paddingStart, int moduleStride,
          int numImgColors, int numGroups,
          float scaleTargets, float scaleOutput, bool conv) {
  int numFilterColors = numImgColors / numGroups;   
  int numFilters = filters->size[0];
  int numModules = numModulesX * numModulesX;
  int numImages = images->size[0];
  int imgPixels = images->size[1]/numImgColors;
  int imgSize = int(sqrt(imgPixels));
  int filterModuleMult = conv ? 1 : numModules;
  
  assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
  assert(numGroups == 1 || numFilterColors % 2 == 0);
  assert(numFilters % (16 * numGroups) == 0 || numFilters == 1);
  assert(numImgColors % numGroups == 0);
  assert(imgSize * imgSize == imgPixels);
  assert(images->size[1] == imgPixels * numImgColors);
  int numFiltersPerGroup = numFilters / numGroups;

  int imgStride = images->size[0]; //images.getStride(); // images does not need to be a contiguous matrix

  int filterPixels = filters->size[1] / (filterModuleMult * numFilterColors);
  int filterSize = int(sqrt(filterPixels));
  assert(filterSize * filterSize == filterPixels);

  assert(filters->size[1] == filterModuleMult * numFilterColors * filterPixels);

  // These routines don't handle the case when only part of the image is visited in the convolution
  assert(paddingStart <= 0 && paddingStart + (numModules-1)*moduleStride + filterSize >= imgSize);
  assert(moduleStride <= filterSize);
  
  /*
  assert(!images.isTrans());
  assert(!filters.isTrans());
  assert(!targets.isTrans());

  assert(filters.isContiguous());
  assert(targets.isContiguous());*/

  dim3 blocks = numFiltersPerGroup % 32 == 0 ? dim3(DIVUP(numImages, 32 * 4), (numModules * numFilters) / (4 * 8))
                : (numFiltersPerGroup % 16 == 0 ? dim3(DIVUP(numImages, 32 * 4), (numModules * numFilters) / (4 * 4))
                    : dim3(DIVUP(numImages, 32 * 4), numModules * numFilters));
  dim3 threads(32, 4);
  bool checkImgBounds = numImages % 128 != 0;
  //if (scaleTargets == 0) {
  //  targets.resize(numFilters * numModules, numImages);
  //} else {
  assert(targets->size[1] == numFilters * numModules);
  assert(targets->size[0] == numImages);
  //}
 
  if (numFilters == 1) {

    int shared_mem_size = filterSize * filterSize * sizeof(float) ;
    filterActs_YxX_oneoutput<<<numModules, numImages, shared_mem_size>>>(
        images->data_device, filters->data_device, targets->data_device,
        numImages, imgSize, filterSize, paddingStart, moduleStride,
        numModulesX, numImgColors, scaleTargets, scaleOutput);

  } else { 
    if (numImgColors <= 3) {
      assert(numGroups == 1); // It has to be based on above definitions, but just to be sure.
      if (scaleTargets == 0) { // don't scale
        if (numImgColors == 1) {
          if (checkImgBounds) {
            if (numFilters % 32 == 0) {
         // WTF is this shit? Why does it set everything to zero? 
         // There has got to be an explanation.
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, false, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 8, 1, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
         numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else if (numFilters % 16 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, false, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 4, 1, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 1, false, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 1, 32, 1, 1, 1, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            }
          } else {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, false, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 8, 1, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else if (numFilters % 16 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, false, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 4, 1, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 1, false, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 1, 32, 1, 1, 1, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            }
          }
        } else if (numImgColors == 2) {
          if (checkImgBounds) {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, false, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 8, 2, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else if (numFilters % 16 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, false, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 4, 2, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 2, false, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 1, 32, 1, 1, 2, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            }
          } else {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, false, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 8, 2, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else if (numFilters % 16 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, false, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 4, 2, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 2, false, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 1, 32, 1, 1, 2, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            }
          }
        } else if (numImgColors == 3) {
          if (checkImgBounds) {
             if (numFilters % 32 == 0) {
               cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, false, true >, cudaFuncCachePreferShared);
               filterActs_YxX_color < 4, 32, 4, 8, 3, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                     numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
             } else if (numFilters % 16 == 0) {
               cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, false, true >, cudaFuncCachePreferShared);
               filterActs_YxX_color < 4, 32, 4, 4, 3, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                     numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
             } else {
               cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 3, false, true >, cudaFuncCachePreferShared);
               filterActs_YxX_color < 1, 32, 1, 1, 3, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                     numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
             }
          } else {
             if (numFilters % 32 == 0) {
               cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, false, false >, cudaFuncCachePreferShared);
               filterActs_YxX_color < 4, 32, 4, 8, 3, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                     numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
             } else if (numFilters % 16 == 0) {
               cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, false, false >, cudaFuncCachePreferShared);
               filterActs_YxX_color < 4, 32, 4, 4, 3, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                     numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
             } else {
               cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 3, false, false >, cudaFuncCachePreferShared);
               filterActs_YxX_color < 1, 32, 1, 1, 3, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                     numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
             }
          }
        }
      } else { // do scale
        if (numImgColors == 1) {
          if (checkImgBounds) {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, true, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 8, 1, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else if (numFilters % 16 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, true, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 4, 1, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 1, true, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 1, 32, 1, 1, 1, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            }
          } else {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 1, true, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 8, 1, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else if (numFilters % 16 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 1, true, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 4, 1, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 1, true, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 1, 32, 1, 1, 1, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            }
          }
        } else if (numImgColors == 2) {
          if (checkImgBounds) {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, true, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 8, 2, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else if (numFilters % 16 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, true, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 4, 2, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 2, true, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 1, 32, 1, 1, 2, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            }
          } else {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 2, true, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 8, 2, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else if (numFilters % 16 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 2, true, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 4, 2, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 2, true, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 1, 32, 1, 1, 2, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            }
          }
        } else if (numImgColors == 3) {
          if (checkImgBounds) {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, true, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 8, 3, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else if (numFilters % 16 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, true, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 4, 3, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 3, true, true >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 1, 32, 1, 1, 3, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            }
          } else {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 8, 3, true, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 8, 3, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else if (numFilters % 16 == 0) {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 4, 32, 4, 4, 3, true, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 4, 32, 4, 4, 3, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            } else {
              cudaFuncSetCacheConfig(filterActs_YxX_color< 1, 32, 1, 1, 3, true, false >, cudaFuncCachePreferShared);
              filterActs_YxX_color < 1, 32, 1, 1, 3, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                    numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
            }
          }
        }
      }
    } else {
      if (scaleTargets == 0) { // don't scale
        if (checkImgBounds) {
          if (numFiltersPerGroup % 32 == 0) {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, true >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 4, 32, 4, 8, 2, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          } else if (numFiltersPerGroup % 16 == 0) {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, false, true >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 4, 32, 4, 4, 2, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          } else {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 1, 128, 1, 1, 4, false, true >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 1, 128, 1, 1, 4, false, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          }
        } else {
          if (numFiltersPerGroup % 32 == 0) {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, false, false >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 4, 32, 4, 8, 2, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          } else if (numFiltersPerGroup % 16 == 0) {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, false, false >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 4, 32, 4, 4, 2, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          } else {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 1, 128, 1, 1, 4, false, false >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 1, 128, 1, 1, 4, false, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          }
        }
      } else { // do scale
        if (checkImgBounds) {
          if (numFiltersPerGroup % 32 == 0) {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, true, true >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 4, 32, 4, 8, 2, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          } else if (numFiltersPerGroup % 16 == 0) {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, true, true >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 4, 32, 4, 4, 2, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          } else {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 1, 128, 1, 1, 4, true, true >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 1, 128, 1, 1, 4, true, true > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          }
        } else {
          if (numFiltersPerGroup % 32 == 0) {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 8, 2, true, false >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 4, 32, 4, 8, 2, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          } else if (numFiltersPerGroup % 16 == 0) {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 4, 32, 4, 4, 2, true, false >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 4, 32, 4, 4, 2, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          } else {
            cudaFuncSetCacheConfig(filterActs_YxX_sparse< 1, 128, 1, 1, 4, true, false >, cudaFuncCachePreferShared);
            filterActs_YxX_sparse < 1, 128, 1, 1, 4, true, false > <<<blocks, threads>>>(images->data_device, filters->data_device, targets->data_device,
                  numImages, numFilters, imgSize, filterSize, paddingStart, moduleStride, numModulesX, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
          }
        }
      }
    }
  }
    
  getLastCudaError("filterActs: kernel execution failed");
}

/*
 * hidActs:   (numFilters, numModules, numImages)
 * filters:   (numFilterColors, filterPixels, numFilters)        if conv
 *       (numModules, numFilterColors, filterPixels, numFilters)  otherwise
 * targets:   (numImageColors, imgPixels, numImages)
 */
void imgActs(cudamat* hidActs, cudamat* filters, cudamat* targets,
       int imgSize, int paddingStart, int moduleStride, int numImgColors, int numGroups,
       float scaleTargets, float scaleOutput, bool conv) {
  int numFilterColors = numImgColors / numGroups;
  int numImages = hidActs->size[0];
  int numFilters = filters->size[0];
  //int numFiltersPerGroup = numFilters / numGroups;
  int numModules = hidActs->size[1] / numFilters;
  int filterModuleMult = conv ? 1 : numModules;
  int filterPixels = filters->size[1] / (filterModuleMult * numFilterColors);
  int filterSize = sqrt(filterPixels);
  int imgPixels = imgSize * imgSize;
  int numModulesX = sqrt(numModules);
  
  assert(numImgColors % numGroups == 0);
  assert(numFilters % (16*numGroups) == 0 || numFilters == 1);
  assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
  assert(numGroups == 1 || numFilterColors % 4 == 0);

  assert(filterPixels == filterSize * filterSize);
  assert(hidActs->size[1] == numModules * numFilters);
  assert(filters->size[1] == filterModuleMult * numFilterColors * filterPixels);
  assert(numModules == numModulesX * numModulesX);

  /*
  assert(hidActs.isContiguous());
  assert(filters.isContiguous());

  assert(!hidActs.isTrans());
  assert(!filters.isTrans());
  assert(!targets.isTrans());*/
  // These routines don't handle the case when only part of the image is visited in the convolution
  assert(paddingStart <= 0 && paddingStart + (numModules-1)*moduleStride + filterSize >= imgSize);
  assert(moduleStride <= filterSize);
  
  //assert(targets.isContiguous()); // no stride support here!

  dim3 blocks;
  dim3 threads(16,16);
  int colorsPerThread;
  bool checkCaseBounds;
  if (numFilterColors % 8 == 0) {
    threads = dim3(32, 4);
    colorsPerThread = numFilterColors % 16 == 0 ? 4 : 2;
    int imgsPerThread = 4;
    assert(numFilterColors % (threads.y * colorsPerThread) == 0);
    checkCaseBounds = numImages % (threads.x * imgsPerThread) != 0;
    blocks = dim3(DIVUP(numImages, threads.x*imgsPerThread) * (numImgColors/(threads.y*colorsPerThread)), imgPixels);
  } else if (numFilterColors > 3) {
    colorsPerThread = numFilterColors % 4 == 0 ? 4 : 2;
    blocks = dim3(DIVUP(numImages,16*8) * (numImgColors / colorsPerThread), DIVUP(imgSize,4) * DIVUP(imgSize,4));
    checkCaseBounds = numImages % (16*8) != 0;
  } else {
    blocks = dim3(DIVUP(numImages,16*8), DIVUP(imgSize,4) * DIVUP(imgSize,4));
    checkCaseBounds = numImages % (16*8) != 0;
  }
  
  //if (scaleTargets == 0) { // do not scale or use targets matrix
  //  targets.resize(numImgColors*imgPixels, numImages);
  //} else {
  assert(targets->size[1] == numImgColors * imgPixels);
  assert(targets->size[0] == numImages);
  //}
  
  if (conv) { // convolutional units
    if (numFilters == 1) {

      conv_img_acts_onefilter<<<imgPixels, numImages>>>(
          hidActs->data_device, filters->data_device, targets->data_device,
          numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart,
          moduleStride, numImgColors, scaleTargets, scaleOutput);

    } else {
      if (scaleTargets == 0) { // do not scale or use targets matrix
        if (numFilterColors % 8 == 0) {
          if (checkCaseBounds) {
            if (numFilterColors % 16 == 0) {
              cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, false, true, true>, cudaFuncCachePreferShared);
              conv_img_acts_manycolor<4, 32, 4, 4, false, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, false, true, true>, cudaFuncCachePreferShared);
              conv_img_acts_manycolor<4, 32, 4, 2, false, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            }
          } else {
            if (numFilterColors % 16 == 0) {
              cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, false, false, true>, cudaFuncCachePreferShared);
              conv_img_acts_manycolor<4, 32, 4, 4, false, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, false, false, true>, cudaFuncCachePreferShared);
              conv_img_acts_manycolor<4, 32, 4, 2, false, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            }
          }
        } else if (numFilterColors > 3) {
          if (checkCaseBounds) {
            if (colorsPerThread == 4) {
              cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 4, false, true, true>, cudaFuncCachePreferShared);
              img_acts_mediumcolor<8, 4, false, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 2, false, true, true>, cudaFuncCachePreferShared);
              img_acts_mediumcolor<8, 2, false, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            }
          } else {
            if (colorsPerThread == 4) {
              cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 4, false, false, true>, cudaFuncCachePreferShared);
              img_acts_mediumcolor<8, 4, false, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 2, false, false, true>, cudaFuncCachePreferShared);
              img_acts_mediumcolor<8, 2, false, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            }
          }
        } else {
          if (checkCaseBounds) {
            if (numFilterColors == 1) {
              cudaFuncSetCacheConfig(img_acts_color<8, 1, false, true, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 1, false, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            } else if (numFilterColors == 2) {
              cudaFuncSetCacheConfig(img_acts_color<8, 2, false, true, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 2, false, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            } else if (numFilterColors == 3) {
              cudaFuncSetCacheConfig(img_acts_color<8, 3, false, true, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 3, false, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            }
          } else {
            if (numFilterColors == 1) {
              cudaFuncSetCacheConfig(img_acts_color<8, 1, false, false, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 1, false, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            } else if (numFilterColors == 2) {
              cudaFuncSetCacheConfig(img_acts_color<8, 2, false, false, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 2, false, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            } else if (numFilterColors == 3) {
              cudaFuncSetCacheConfig(img_acts_color<8, 3, false, false, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 3, false, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            }
          }
        }
      } else { // do scale
        if (numFilterColors % 8 == 0) {
          if (checkCaseBounds) {
            if (numFilterColors % 16 == 0) {
              cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, true, true, true>, cudaFuncCachePreferShared);
              conv_img_acts_manycolor<4, 32, 4, 4, true, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, true, true, true>, cudaFuncCachePreferShared);
              conv_img_acts_manycolor<4, 32, 4, 2, true, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            }
          } else {
            if (numFilterColors % 16 == 0) {
              cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, true, false, true>, cudaFuncCachePreferShared);
              conv_img_acts_manycolor<4, 32, 4, 4, true, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, true, false, true>, cudaFuncCachePreferShared);
              conv_img_acts_manycolor<4, 32, 4, 2, true, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            }
          }
        } else if (numFilterColors > 3) {
          if (checkCaseBounds) {
            if (colorsPerThread == 4) {
              cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 4, true, true, true>, cudaFuncCachePreferShared);
              img_acts_mediumcolor<8, 4, true, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 2, true, true, true>, cudaFuncCachePreferShared);
              img_acts_mediumcolor<8, 2, true, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            }
          } else {
            if (colorsPerThread == 4) {
              cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 4, true, false, true>, cudaFuncCachePreferShared);
              img_acts_mediumcolor<8, 4, true, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 2, true, false, true>, cudaFuncCachePreferShared);
              img_acts_mediumcolor<8, 2, true, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
            }
          }
        } else {
          if (checkCaseBounds) {
            if (numFilterColors == 1) {
              cudaFuncSetCacheConfig(img_acts_color<8, 1, true, true, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 1, true, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            } else if (numFilterColors == 2) {
              cudaFuncSetCacheConfig(img_acts_color<8, 2, true, true, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 2, true, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            } else if (numFilterColors == 3) {
              cudaFuncSetCacheConfig(img_acts_color<8, 3, true, true, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 3, true, true, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            }
          } else {
            if (numFilterColors == 1) {
              cudaFuncSetCacheConfig(img_acts_color<8, 1, true, false, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 1, true, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            } else if (numFilterColors == 2) {
              cudaFuncSetCacheConfig(img_acts_color<8, 2, true, false, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 2, true, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            } else if (numFilterColors == 3) {
              cudaFuncSetCacheConfig(img_acts_color<8, 3, true, false, true>, cudaFuncCachePreferShared);
              img_acts_color<8, 3, true, false, true><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                                numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
            }
          }
        }
      }
    }
  } else { // local, unshared units
    if (scaleTargets == 0) { // do not scale or use targets matrix
      if (numFilterColors % 8 == 0) {
        if (checkCaseBounds) {
          if (numFilterColors % 16 == 0) {
            cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, false, true, false>, cudaFuncCachePreferShared);
            conv_img_acts_manycolor<4, 32, 4, 4, false, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          } else {
            cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, false, true, false>, cudaFuncCachePreferShared);
            conv_img_acts_manycolor<4, 32, 4, 2, false, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          }
        } else {
          if (numFilterColors % 16 == 0) {
            cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, false, false, false>, cudaFuncCachePreferShared);
            conv_img_acts_manycolor<4, 32, 4, 4, false, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          } else {
            cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, false, false, false>, cudaFuncCachePreferShared);
            conv_img_acts_manycolor<4, 32, 4, 2, false, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          }
        }
      } else if (numFilterColors > 3) {
        if (checkCaseBounds) {
          if (colorsPerThread == 4) {
            cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 4, false, true, false>, cudaFuncCachePreferShared);
            img_acts_mediumcolor<8, 4, false, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          } else {
            cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 2, false, true, false>, cudaFuncCachePreferShared);
            img_acts_mediumcolor<8, 2, false, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          }
        } else {
          if (colorsPerThread == 4) {
            cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 4, false, false, false>, cudaFuncCachePreferShared);
            img_acts_mediumcolor<8, 4, false, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          } else {
            cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 2, false, false, false>, cudaFuncCachePreferShared);
            img_acts_mediumcolor<8, 2, false, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          }
        }
      } else {
        if (checkCaseBounds) {
          if (numFilterColors == 1) {
            cudaFuncSetCacheConfig(img_acts_color<8, 1, false, true, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 1, false, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          } else if (numFilterColors == 2) {
            cudaFuncSetCacheConfig(img_acts_color<8, 2, false, true, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 2, false, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          } else if (numFilterColors == 3) {
            cudaFuncSetCacheConfig(img_acts_color<8, 3, false, true, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 3, false, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          }
        } else {
          if (numFilterColors == 1) {
            cudaFuncSetCacheConfig(img_acts_color<8, 1, false, false, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 1, false, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          } else if (numFilterColors == 2) {
            cudaFuncSetCacheConfig(img_acts_color<8, 2, false, false, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 2, false, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          } else if (numFilterColors == 3) {
            cudaFuncSetCacheConfig(img_acts_color<8, 3, false, false, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 3, false, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          }
        }
      }
    } else { // do scale
      if (numFilterColors % 8 == 0) {
        if (checkCaseBounds) {
          if (numFilterColors % 16 == 0) {
            cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, true, true, false>, cudaFuncCachePreferShared);
            conv_img_acts_manycolor<4, 32, 4, 4, true, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          } else {
            cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, true, true, false>, cudaFuncCachePreferShared);
            conv_img_acts_manycolor<4, 32, 4, 2, true, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          }
        } else {
          if (numFilterColors % 16 == 0) {
            cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 4, true, false, false>, cudaFuncCachePreferShared);
            conv_img_acts_manycolor<4, 32, 4, 4, true, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          } else {
            cudaFuncSetCacheConfig(conv_img_acts_manycolor<4, 32, 4, 2, true, false, false>, cudaFuncCachePreferShared);
            conv_img_acts_manycolor<4, 32, 4, 2, true, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          }
        }
      } else if (numFilterColors > 3) {
        if (checkCaseBounds) {
          if (colorsPerThread == 4) {
            cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 4, true, true, false>, cudaFuncCachePreferShared);
            img_acts_mediumcolor<8, 4, true, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          } else {
            cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 2, true, true, false>, cudaFuncCachePreferShared);
            img_acts_mediumcolor<8, 2, true, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          }
        } else {
          if (colorsPerThread == 4) {
            cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 4, true, false, false>, cudaFuncCachePreferShared);
            img_acts_mediumcolor<8, 4, true, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          } else {
            cudaFuncSetCacheConfig(img_acts_mediumcolor<8, 2, true, false, false>, cudaFuncCachePreferShared);
            img_acts_mediumcolor<8, 2, true, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput);
          }
        }
      } else {
        if (checkCaseBounds) {
          if (numFilterColors == 1) {
            cudaFuncSetCacheConfig(img_acts_color<8, 1, true, true, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 1, true, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          } else if (numFilterColors == 2) {
            cudaFuncSetCacheConfig(img_acts_color<8, 2, true, true, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 2, true, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          } else if (numFilterColors == 3) {
            cudaFuncSetCacheConfig(img_acts_color<8, 3, true, true, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 3, true, true, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          }
        } else {
          if (numFilterColors == 1) {
            cudaFuncSetCacheConfig(img_acts_color<8, 1, true, false, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 1, true, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          } else if (numFilterColors == 2) {
            cudaFuncSetCacheConfig(img_acts_color<8, 2, true, false, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 2, true, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          } else if (numFilterColors == 3) {
            cudaFuncSetCacheConfig(img_acts_color<8, 3, true, false, false>, cudaFuncCachePreferShared);
            img_acts_color<8, 3, true, false, false><<<blocks, threads>>>(hidActs->data_device, filters->data_device, targets->data_device,
                              numModulesX, numImages, numFilters, filterSize, imgSize, paddingStart, moduleStride, scaleTargets, scaleOutput);
          }
        }
      }
    }
  }
  
  getLastCudaError("imgActs: kernel execution failed");
}

void weightActs(cudamat* images, cudamat* hidActs, cudamat* targets,
    int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors,
    int numGroups, int partialSum, float scaleTargets, float scaleOutput) {
  int numFilterColors = numImgColors / numGroups;
  int imgStride = images->size[0];
  int numImages = images->size[0];
  int imgPixels = images->size[1] / numImgColors;
  int imgSize = int(sqrt(imgPixels));
  int numModules = numModulesX * numModulesX;
  int numFilters = hidActs->size[1] / numModules;
  int numFiltersPerGroup = numFilters / numGroups;
  
  assert(numImgColors % numGroups == 0);
  assert(numFilters % (16*numGroups) == 0 || numFilters == 1);
  assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 4 == 0)));
  assert(numGroups == 1 || numFilterColors % 4 == 0);
  assert(imgSize * imgSize == imgPixels);
  assert(images->size[1] == imgPixels * numImgColors);

  int filterPixels = filterSize * filterSize;
  partialSum = partialSum == 0 ? numModules : partialSum;

  assert(numModules % partialSum == 0);
  assert(hidActs->size[0] == numImages);

  // These routines don't handle the case when only part of the image is visited in the convolution
  assert(paddingStart <= 0 && paddingStart + (numModules-1)*moduleStride + filterSize >= imgSize);
  assert(moduleStride <= filterSize);
  
  assert(numModules * numFilters == hidActs->size[1]);

  /*
  assert(!images.isTrans());
  assert(!hidActs.isTrans());
  assert(hidActs.isContiguous());

  assert(!targets.isTrans());
  assert(targets.isContiguous());*/
  
  int preloadCases = 32;

  dim3 blocks, threads;
  int bx, by;
  int pixelsPerThread, filtersPerThread, colorsPerThread;
  // Worth playing with these parameters to find best values for your problem.
  // These values work relatively well, but not optimal for all problems.
  if (numFilterColors > 3) {
    filtersPerThread = numFiltersPerGroup % 32 == 0 ? 2 : 1;
    colorsPerThread = numFilterColors % 8 == 0 ? 8 : 4;
    by = numFiltersPerGroup % 64 == 0 ? 4 : 8;
    bx = numFiltersPerGroup % 64 == 0 ? 32 : 16;
    blocks = dim3((numModules/partialSum)*(numFilters/(bx*filtersPerThread)), DIVUP(filterPixels, by) * (numFilterColors / colorsPerThread));
  } else {
    assert(numGroups == 1); // Just for sanity
    pixelsPerThread = numFilters % 32 == 0 ? (numImgColors == 1 ? 8 : 5) : (numImgColors == 1 ? 5 : 2);
    by = numFilters % 32 == 0 ? 4 : 8; // by == 4 seems to work best
    bx = numFilters % 32 == 0 ? 32 : 16; 
    blocks = dim3((numModules/partialSum)*(numFilters/bx), DIVUP(filterPixels, by*pixelsPerThread));
  }
  assert((by * bx) % preloadCases == 0);
  threads = dim3(bx, by);
  bool checkCaseBounds = numImages % 32 != 0;
  
  //if (scaleTargets == 0) {
  //  targets.resize((numModules/partialSum) * numFilterColors*filterPixels, numFilters);
  //} else {
  assert(targets->size[1] == (numModules/partialSum) * numFilterColors*filterPixels);
  assert(targets->size[0] == numFilters);
  //}
  if (numFilters == 1) {
    int filter_dims = filterSize * filterSize * numImgColors;
    conv_weight_acts_oneoutput<<<filter_dims, numImages, numImages * sizeof(float)>>>(
        images->data_device, hidActs->data_device, targets->data_device,
        numImages, numModulesX, imgSize, filterSize,
        paddingStart, moduleStride, imgStride, scaleTargets, scaleOutput);
  } else {
    if (numFilterColors > 3) {
      if (scaleTargets == 0) { // do not scale
        if (numFiltersPerGroup % 64 == 0) {
          if (numFilterColors % 8 == 0) {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,8,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<4,32,2,8,32,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,8,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<4,32,2,8,32,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,4,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<4,32,2,4,32,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,4,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<4,32,2,4,32,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          }
        } else if (numFiltersPerGroup % 32 == 0) {
          if (numFilterColors % 8 == 0) {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,8,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,2,8,32,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,8,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,2,8,32,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,4,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,2,4,32,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,4,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,2,4,32,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          }
        } else if (numFiltersPerGroup == 1) {  // newly added.
          if (numFilterColors % 8 == 0) {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<32,1,1,8,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<32,1,1,8,32,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<32,1,1,8,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<32,1,1,8,32,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<32,1,1,4,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<32,1,1,4,32,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<32,1,1,4,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<32,1,1,4,32,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          }
        } else {
          if (numFilterColors % 8 == 0) {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,8,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,1,8,32,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,8,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,1,8,32,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,4,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,1,4,32,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,4,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,1,4,32,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          }
        }
      } else {
        if (numFiltersPerGroup % 64 == 0) {
          if (numFilterColors % 8 == 0) {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,8,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<4,32,2,8,32,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,8,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<4,32,2,8,32,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,4,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<4,32,2,4,32,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<4,32,2,4,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<4,32,2,4,32,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          }
        } else if (numFiltersPerGroup % 32 == 0) {
          if (numFilterColors % 8 == 0) {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,8,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,2,8,32,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,8,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,2,8,32,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,4,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,2,4,32,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,2,4,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,2,4,32,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          }
       } else if (numFiltersPerGroup == 1) {
          if (numFilterColors % 8 == 0) {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<32,1,1,8,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<32,1,1,8,32,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<32,1,1,8,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<32,1,1,8,32,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<32,1,1,4,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<32,1,1,4,32,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<32,1,1,4,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<32,1,1,4,32,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          }
        } else {
          if (numFilterColors % 8 == 0) {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,8,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,1,8,32,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,8,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,1,8,32,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (checkCaseBounds) {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,4,32, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,1,4,32,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_mc_mf<8,16,1,4,32, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_mc_mf<8,16,1,4,32,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                              numImages, numFilters, numModulesX, imgSize, filterSize,
                                              paddingStart, moduleStride, imgStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
            }
          }
        }
      }
    } else { // numColors in 1,2,3
      if (scaleTargets == 0) { // do not scale
        if (numFilterColors == 1) {
          if (checkCaseBounds) {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,8,32,1, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,8,32,1,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,5,32,1, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,5,32,1,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,8,32,1, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,8,32,1,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,5,32,1, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,5,32,1,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          }
        } else if (numFilterColors == 2) {
          if (checkCaseBounds) {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,2, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,5,32,2,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,2, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,2,32,2,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,2, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,5,32,2,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,2, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,2,32,2,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          }
        } else if (numFilterColors == 3) {
          if (checkCaseBounds) {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,3, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,5,32,3,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,3, false, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,2,32,3,false, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,3, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,5,32,3,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,3, false, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,2,32,3,false, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          }
        }

      } else { // do scale
        if (numFilterColors == 1) {
          if (checkCaseBounds) {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,8,32,1, true, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,8,32,1,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,5,32,1, true, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,5,32,1,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,8,32,1, true, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,8,32,1,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,5,32,1, true, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,5,32,1,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          }
        } else if (numFilterColors == 2) {
          if (checkCaseBounds) {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,2, true, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,5,32,2,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,2, true, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,2,32,2,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,2, true, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,5,32,2,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,2, true, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,2,32,2,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          }
        } else if (numFilterColors == 3) {
          if (checkCaseBounds) {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,3, true, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,5,32,3,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,3, true, true>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,2,32,3,true, true><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          } else {
            if (numFilters % 32 == 0) {
              cudaFuncSetCacheConfig(conv_weight_acts_c<4,32,5,32,3, true, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<4,32,5,32,3,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            } else {
              cudaFuncSetCacheConfig(conv_weight_acts_c<8,16,2,32,3, true, false>, cudaFuncCachePreferShared);
              conv_weight_acts_c<8,16,2,32,3,true, false><<<blocks, threads>>>(images->data_device, hidActs->data_device, targets->data_device,
                                  numImages, numFilters, numModulesX, imgSize, filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleTargets, scaleOutput);
            }
          }
        }
      }
    }
  }
  getLastCudaError("weightActs: kernel execution failed");
}

/*
 * imgs:    (numFilters, imgPixels, numImages)
 * maxGrads:  (numFilters, numOutputs, numImages)
 * rMaxActs:  (numFilters, numOutputs, numImages)
 * target:   (numFilters, imgPixels, numImages)
 */
void convLocalMaxUndo(cudamat* images, cudamat* maxGrads, cudamat* maxActs, cudamat* target,
           int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput) {
  int outputs = outputsX * outputsX;
  int numImages = images->size[0];
  int numFilters = maxGrads->size[1] / outputs;
  int imgPixels = images->size[1] / numFilters;
  assert(images->size[1] == numFilters * imgPixels);
  int imgSize = int(sqrt(imgPixels));
  
  assert(imgSize * imgSize == imgPixels);
  assert(maxGrads->size[1] == numFilters * outputs);
  assert(maxGrads->size[0] == numImages);

  /*
  assert(!images.isTrans());
  assert(!target.isTrans());
  assert(!maxGrads.isTrans());
  assert(!maxActs.isTrans());
  assert(images.isContiguous());
  assert(maxGrads.isContiguous());
  assert(maxActs.isContiguous());
  assert(maxGrads.isSameDims(maxActs));
  */

  assert(numFilters % 16 == 0);
//  assert(numImages % 128 == 0);
  
  assert(strideX <= subsX);
  
  //target.resize(images);
  
  int checkCaseBounds = numImages % 128 != 0;
  dim3 threads(32, 4);
  dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / (4 * 2)) * imgSize);
  
  if (checkCaseBounds) {
    if (scaleTargets == 0 && scaleOutput == 1) {
      kLocalMaxUndo<4, 32, 4, 2, false, true><<<blocks, threads>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                              imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
    } else {
      kLocalMaxUndo<4, 32, 4, 2, true, true><<<blocks, threads>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                              imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
    }
  } else {
    if (scaleTargets == 0 && scaleOutput == 1) {
      kLocalMaxUndo<4, 32, 4, 2, false, false><<<blocks, threads>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                              imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
    } else {
      kLocalMaxUndo<4, 32, 4, 2, true, false><<<blocks, threads>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                              imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
    }
  }

  getLastCudaError("convLocalMaxUndo: kernel execution failed");
}

/*
 * avgGrads:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
void convLocalAvgUndo(cudamat* avgGrads, cudamat* target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize,
                      float scaleTargets, float scaleOutput) {
    int numImages = avgGrads->size[0];

    int outputs = outputsX * outputsX;
    int numFilters = avgGrads->size[1] / outputs;
    assert(avgGrads->size[1] == numFilters * outputs);

    assert(numFilters % 16 == 0);
//    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    int checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * imgSize, (numFilters / (4 * 4)) * imgSize);
    
    if (imgsPerThread == 4) {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 4, 4, false, true><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 4, 4, true, true><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 4, 4, false, false><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 4, 4, true, false><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        }
    } else if (imgsPerThread == 2) {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 2, 4, false, true><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 2, 4, true, true><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 2, 4, false, false><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 2, 4, true, false><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        }
    } else {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 1, 4, false, true><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 1, 4, true, true><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 1, 4, false, false><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 1, 4, true, false><<<blocks, threads>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        }
    }

    getLastCudaError("convLocalAvgUndo: kernel execution failed");
}


void convResponseNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* inputs, cudamat* acts, cudamat* target, int numFilters,
             int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput) {
  int numImages = outGrads->size[0];
  int imgPixels = outGrads->size[1] / numFilters;

  int imgSize = int(sqrt(imgPixels));
  assert(imgSize * imgSize == imgPixels);

  assert(outGrads->size[1] == numFilters * imgPixels);
  
  //assert(denoms.isSameDims(outGrads));
  //assert(acts.isSameDims(denoms));
  //assert(!denoms.isTrans());
  //assert(!outGrads.isTrans());
  //assert(!acts.isTrans());
  //assert(!target.isTrans());
  //assert(outGrads.isContiguous());
  
  assert(numFilters % 16 == 0);
  
  //target.resize(outGrads);
  
  // First do acts := -2 x scale x acts x outGrads / denoms
  // so that the main routine only has to do an addition in its inner loop.
  int prelimEltsPerThread = 4;
  dim3 threads(128);
  dim3 blocks(MIN(512, DIVUP(outGrads->size[0]*outGrads->size[1],(threads.x * prelimEltsPerThread))));
  kRNormUndoPrelims<128, 4><<<blocks, threads>>>(acts->data_device, denoms->data_device, outGrads->data_device, outGrads->size[0]*outGrads->size[1], -2*addScale*powScale);
  
  // Now the main routine
  if (sizeX >= 6 && numFilters % 4 == 0) {
    // This one is faster for large regions (my tests show regions >= 6...)
    int imgsPerThread = 8;
    int filtersPerThread = 4;
    int bx = 16;
    bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
    assert((imgsPerThread * bx) % 32 == 0);

    threads = dim3(bx, 16);
    blocks = dim3(DIVUP(imgSize, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(imgSize, 4) * numFilters / filtersPerThread);
    if (checkCaseBounds) {
      if (scaleTargets == 0 && scaleOutput == 1) {
        cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, true, true>, cudaFuncCachePreferL1);
        kRNormUndo2<16, 8, 4, true, true><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                       target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                       scaleTargets, scaleOutput);
      } else {
        cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, false, true>, cudaFuncCachePreferL1);
        kRNormUndo2<16, 8, 4, false, true><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                       target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                       scaleTargets, scaleOutput);
      }
    } else {
      if (scaleTargets == 0 && scaleOutput == 1) {
        cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, true, false>, cudaFuncCachePreferL1);
        kRNormUndo2<16, 8, 4, true, false><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                       target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                       scaleTargets, scaleOutput);
      } else {
        cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, false, false>, cudaFuncCachePreferL1);
        kRNormUndo2<16, 8, 4, false, false><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                       target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                       scaleTargets, scaleOutput);
      }
    }
  } else {
    bool checkCaseBounds = numImages % 128 != 0;
    threads = dim3(32, 4);
    blocks = dim3(DIVUP(numImages,32*2) * imgSize, (numFilters / (4 * 2)) * imgSize);
    if (checkCaseBounds) { 
      if (scaleTargets == 0 && scaleOutput == 1) {
        cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, false, true>, cudaFuncCachePreferL1);
        kRNormUndo<4, 32, 2, 2, false, true><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                     target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                     scaleTargets, scaleOutput);
      } else {
        cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, true, true>, cudaFuncCachePreferL1);
        kRNormUndo<4, 32, 2, 2, true, true><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                     target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                     scaleTargets, scaleOutput);
      }
    } else {
      if (scaleTargets == 0 && scaleOutput == 1) {
        cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, false, false>, cudaFuncCachePreferL1);
        kRNormUndo<4, 32, 2, 2, false, false><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                     target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                     scaleTargets, scaleOutput);
      } else {
        cudaFuncSetCacheConfig(kRNormUndo<4, 32, 2, 2, true, false>, cudaFuncCachePreferL1);
        kRNormUndo<4, 32, 2, 2, true, false><<<blocks, threads>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                     target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                     scaleTargets, scaleOutput);
      }
    }
  }
  getLastCudaError("kRNormUndo: kernel execution failed");
}

void convContrastNormUndo(cudamat* outGrads, cudamat* denoms,
                          cudamat* meanDiffs, cudamat* acts, cudamat* target,
                          int numFilters, int sizeX, float addScale,
                          float powScale, float scaleTargets, float scaleOutput) {
  convResponseNormUndo(outGrads, denoms, meanDiffs, acts, target, numFilters,
                       sizeX, addScale, powScale, scaleTargets, scaleOutput);
}

void convResponseNorm(cudamat* images, cudamat* denoms, cudamat* target, int numFilters, int sizeX, float addScale, float powScale) {
  convContrastNorm(images, images, denoms, target, numFilters, sizeX, addScale, powScale);
}


  // Convolutions.
void convUp(cudamat* images, cudamat* filters, cudamat* targets, int numModulesX, int paddingStart, int moduleStride, int numImgColors, int numGroups, float scaleTargets){
  filterActs(images, filters, targets, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, 1, true);
}
void convDown(cudamat* images, cudamat* filters, cudamat* targets, int imgSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, float scaleTargets){
  imgActs(images, filters, targets, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, 1, true);
}
void convOutp(cudamat* images, cudamat* hidSums, cudamat* targets, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, int partialSum, float scaleTargets, float scaleOutput){
  weightActs(images, hidSums, targets, numModulesX, filterSize, paddingStart, moduleStride, numImgColors, numGroups, partialSum, scaleTargets, scaleOutput);
  // resize targets_partial_sum so we can add along rows.
}

// Local Connections.
void localUp(cudamat* images, cudamat* filters, cudamat* targets, int numModulesX, int paddingStart, int moduleStride, int numImgColors, int numGroups, float scaleTargets){
 filterActs(images, filters, targets, numModulesX, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, 1, false);
}
void localDown(cudamat* images, cudamat* filters, cudamat* targets, int imgSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, float scaleTargets){
 imgActs(images, filters, targets, imgSize, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, 1, false);
}
void localOutp(cudamat* images, cudamat* hidSums, cudamat* targets, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numGroups, float scaleTargets, float scaleOutput){
 weightActs(images, hidSums, targets, numModulesX, filterSize, paddingStart, moduleStride, numImgColors, numGroups, 1, scaleTargets, scaleOutput);
}

// Response Normalization.
void ResponseNorm(cudamat* images, cudamat* denoms, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convResponseNorm(images, denoms, targets, numFilters, sizeX, addScale, powScale);
}

void ResponseNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* inputs, cudamat* acts, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convResponseNormUndo(outGrads, denoms, inputs, acts, targets, numFilters, sizeX, addScale, powScale, 0, 1);
}

// Response Normalization cross map - computes : images / ((1 + addScale * (sum sq images over neighbourhood))^{powScale})
// blocked : true means divide input into blocks and compete within each, false means compete within a running window centered at self.
void ResponseNormCrossMap(cudamat* images, cudamat* denoms, cudamat* targets, int numFilters, int sizeF, float addScale, float powScale, bool blocked){
  convResponseNormCrossMap(images, denoms, targets, numFilters, sizeF, addScale, powScale, blocked);
}

// overwrites acts.
void ResponseNormCrossMapUndo(cudamat* outGrads, cudamat* denoms, cudamat* inputs, cudamat* acts, cudamat* targets, int numFilters, int sizeF, float addScale, float powScale, bool blocked){
  convResponseNormCrossMapUndo(outGrads, denoms, inputs, acts, targets, numFilters, sizeF, addScale, powScale, blocked, 0, 1);
}

// Contrast Normalization.
void ContrastNorm(cudamat* images, cudamat* meanDiffs, cudamat* denoms, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convContrastNorm(images, meanDiffs, denoms, targets, numFilters, sizeX, addScale, powScale);
}
void ContrastNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* meanDiffs, cudamat* acts, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convContrastNormUndo(outGrads, denoms, meanDiffs, acts, targets, numFilters, sizeX, addScale, powScale, 0, 1);
}

// Pooling.
void MaxPool(cudamat* images, cudamat* targets, int numFilters, int subsX,	int startX,	int strideX, int outputsX){
  MaxPooler mpooler;
  convLocalPool<MaxPooler>(images, targets, numFilters, subsX, startX, strideX, outputsX, mpooler);
}

void AvgPool(cudamat* images, cudamat* targets, int numFilters, int subsX,	int startX,	int strideX, int outputsX){
  AvgPooler pooler = AvgPooler(subsX);
  convLocalPool<AvgPooler>(images, targets, numFilters, subsX, startX, strideX, outputsX, pooler);
}

void ProbMaxPool(cudamat* images, cudamat* rnd, cudamat* targets, int numFilters, int subsX,	int startX,	int strideX, int outputsX){
  ProbMaxPooler mpooler;
  convLocalProbPool<ProbMaxPooler>(images, rnd, targets, numFilters, subsX, startX, strideX, outputsX, mpooler);
}

void MaxPoolUndo(cudamat* images, cudamat* maxGrads, cudamat* maxActs, cudamat* targets, int subsX, int startX, int strideX, int outputsX){
  convLocalMaxUndo(images, maxGrads, maxActs, targets, subsX, startX, strideX, outputsX, 0, 1);
}

void AvgPoolUndo(cudamat* avgGrads, cudamat* targets, int subsX, int startX, int strideX, int outputsX, int imgSize) {
  convLocalAvgUndo(avgGrads, targets, subsX, startX, strideX, outputsX, imgSize, 0, 1);
}

    //MaxPool(input, output, num_input_channels_, kernel_size_, -padding_,
     //       stride_, num_modules_);
void UpSample(cudamat* images, cudamat* targets, int factor, int input_image_size, float scaleTargets) { 
  convLocalAvgUndo(images, targets, factor, 0, factor, input_image_size,
                   factor * input_image_size, scaleTargets, factor * factor);
}

void DownSample(cudamat* images, cudamat* targets, int factor, int input_image_size) {
  AvgPooler pooler = AvgPooler(factor);
  int num_filters = images->size[1] / (input_image_size * input_image_size);
  convLocalPool<AvgPooler>(images, targets, num_filters, factor, 0, factor,
                           input_image_size / factor, pooler);
}

void RGBToYUV(cudamat* images, cudamat* targets) {
  convRGBToYUV(images, targets);
}
}  // end extern "C"
