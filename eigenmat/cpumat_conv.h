#ifndef CPUMAT_CONV_H
#define CPUMAT_CONV_H

#include "eigenmat.h"

void convUp(eigenmat* images, eigenmat* filters, eigenmat* targets,
            Shape4D &images_shape, Shape4D &filters_shape, Shape4D &targets_shape,
            ConvDesc &conv_desc, float scaleTargets, float scaleOutput, bool conv);

void convDown(eigenmat* derivs, eigenmat* filters, eigenmat* targets,
              Shape4D &derivs_shape, Shape4D &filters_shape, Shape4D &targets_shape,
              ConvDesc &conv_desc, float scaleTargets, float scaleOutput, bool conv);

void convOutp(eigenmat* images, eigenmat* derivs, eigenmat* targets,
              Shape4D &images_shape, Shape4D &derivs_shape, Shape4D &targets_shape,
              ConvDesc &conv_desc, float scaleTargets, float scaleOutput, bool conv);

void ResponseNormCrossMap(eigenmat* images, eigenmat* targets,
                          int num_filters, int sizeF, float addScale, float powScale, bool blocked);

void ResponseNormCrossMapUndo(eigenmat* outGrads, eigenmat* inputs, eigenmat* targets,
                              int num_filters, int sizeF, float addScale, float powScale, bool blocked);

#endif

