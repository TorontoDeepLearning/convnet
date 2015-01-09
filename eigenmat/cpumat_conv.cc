#include "cpumat_conv.h"

#include <assert.h>

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif
#ifndef MAX
#define MAX(x,y) ((x > y) ? x : y)
#endif

using namespace std;

void expand(float *images, float* targets,
            int num_images, int num_input_channels,
            int image_size_y, int image_size_x,
            int num_modules_y, int num_modules_x,
            int kernel_size_y, int kernel_size_x,
            int padding_y, int padding_x,
            int stride_y, int stride_x,
            int module_id_offset)
{
    int src_module_id = module_id_offset;
    int dst_module_id = 0;

    int module_id_x = src_module_id % num_modules_x;
    int module_id_y = src_module_id / num_modules_x;
    int startX = module_id_x * stride_x + padding_x;
    int startY = module_id_y * stride_y + padding_y;
    int Y, X;
    long target_id, source_id;

    for (int color=0; color<num_input_channels; ++color)
    {
        float *imgs = images + num_images * image_size_x * image_size_y * color;
        float *trgs = targets + num_images * (dst_module_id + kernel_size_y * kernel_size_x * color);
        for (int y=0; y<kernel_size_y; y++)
        {
            Y = startY + y;
            for (int x=0; x<kernel_size_x; x++)
            {
                X = startX + x;
                target_id = num_images * (x + kernel_size_x * y);
                source_id = num_images * (X + image_size_x * Y);
                if (X < 0 || X >= image_size_x || Y < 0 || Y >= image_size_y)
                {
                    for (int im=0; im<num_images; ++im)
                    {
                        trgs[target_id + im] = 0;
                    }
                } else
                {
                    for (int im=0; im<num_images; ++im)
                    {
                        trgs[target_id + im] = imgs[source_id + im];
                    }
                }
            }
        }
    }
}

void contract(float *expanded_data, float* targets,
              int num_images, int num_input_channels,
              int image_size_y, int image_size_x,
              int num_modules_y, int num_modules_x,
              int kernel_size_y, int kernel_size_x,
              int padding_y, int padding_x,
              int stride_y, int stride_x,
              int module_id_offset)
{
    int dst_module_id = module_id_offset;
    int src_module_id = 0;

    int module_id_x = dst_module_id % num_modules_x;
    int module_id_y = dst_module_id / num_modules_x;
    int startX = module_id_x * stride_x + padding_x;
    int startY = module_id_y * stride_y + padding_y;
    int Y, X;
    long target_id, source_id;

    for (int color=0; color<num_input_channels; ++color)
    {
        float *trgs = targets + num_images * image_size_x * image_size_y * color;
        float *edas = expanded_data + num_images * (src_module_id + kernel_size_y * kernel_size_x * color);
        for (int y=0; y<kernel_size_y; y++)
        {
            Y = startY + y;
            for (int x=0; x<kernel_size_x; x++)
            {
                X = startX + x;
                source_id = num_images * (x + kernel_size_x * y);
                target_id = num_images * (X + image_size_x * Y);
                if (X < 0 || X >= image_size_x || Y < 0 || Y >= image_size_y)
                {
                    // do nothing.
                } else
                {
                    for (int im=0; im<num_images; ++im)
                    {
                        trgs[target_id + im] += edas[source_id + im];
                    }
                }
            }
        }
    }
}

void convUp(eigenmat* images, eigenmat* filters, eigenmat* targets,
            Shape4D &images_shape, Shape4D &filters_shape, Shape4D &targets_shape,
            ConvDesc &conv_desc, float scaleTargets, float scaleOutput, bool conv)
{
    int num_input_channels   = conv_desc.num_input_channels;
    int num_output_channels  = conv_desc.num_output_channels;
    int kernel_size_y        = conv_desc.kernel_size_y;
    int kernel_size_x        = conv_desc.kernel_size_x;
    int stride_y             = conv_desc.stride_y;
    int stride_x             = conv_desc.stride_x;
    int padding_y            = conv_desc.padding_y;
    int padding_x            = conv_desc.padding_x;
    int input_channel_begin  = conv_desc.input_channel_begin;
    int input_channel_end    = conv_desc.input_channel_end;
    int output_channel_begin = conv_desc.output_channel_begin;
    int output_channel_end   = conv_desc.output_channel_end;
    int num_groups           = conv_desc.num_groups;

    if (output_channel_end == 0)
        output_channel_end = num_output_channels;
    if (input_channel_end == 0)
        input_channel_end = num_input_channels;

    int num_output_channels2 = targets_shape.shape[3];
    int num_modules_y        = targets_shape.shape[2];
    int num_modules_x        = targets_shape.shape[1];
    int num_images           = targets_shape.shape[0];

    int num_input_channels2  = images_shape.shape[3];
    int image_size_y         = images_shape.shape[2];
    int image_size_x         = images_shape.shape[1];
    int num_images2          = images_shape.shape[0];

    int num_input_channels3  = filters_shape.shape[3];
    int kernel_size_y2       = filters_shape.shape[2];
    int kernel_size_x2       = filters_shape.shape[1];
    int num_output_channels3 = filters_shape.shape[0];

    int num_modules          = num_modules_y * num_modules_x;
    int filterModuleMult     = conv ? 1 : num_modules;
  
    // Consistency checks.
    assert(num_images == num_images2);
    assert(num_output_channels == num_output_channels2);
    assert(output_channel_end - output_channel_begin == num_output_channels3);
    assert(num_input_channels == num_input_channels2);
    assert(input_channel_end - input_channel_begin == num_input_channels3 / filterModuleMult);
    assert(num_images == images->size[0]);
    assert(num_images == targets->size[0]);
    assert(num_output_channels3 == filters->size[0]);
    assert(image_size_y * image_size_x * num_input_channels == images->size[1]);
    assert(num_modules_y * num_modules_x * num_output_channels == targets->size[1]);
    assert(kernel_size_y * kernel_size_x * num_input_channels3 * filterModuleMult == filters->size[1]);
    assert(kernel_size_y == kernel_size_y2);
    assert(kernel_size_x == kernel_size_x2);
    assert(num_input_channels % num_groups == 0);
    assert(num_groups == 1);
    assert(input_channel_begin  >= 0);
    assert(output_channel_begin >= 0);
    assert(input_channel_end    <= num_input_channels);
    assert(output_channel_end   <= num_output_channels);
    assert(input_channel_begin  <= input_channel_end);
    assert(output_channel_begin <= output_channel_end);
    num_input_channels = input_channel_end - input_channel_begin;
    num_output_channels = output_channel_end - output_channel_begin;
    assert(num_input_channels  > 0);
    assert(num_output_channels > 0);

    float* w = filters->data;
    float* images_data = images->data + input_channel_begin * image_size_y * image_size_x * num_images;
    float* targets_data = targets->data + output_channel_begin * num_modules * num_images;

    int input_size = kernel_size_y * kernel_size_x * num_input_channels;

    float *expanded_images = new float[num_images * input_size];
    float *expanded_target = new float[num_images * num_output_channels];

    for (int i=0; i<num_modules; i++)
    {
        expand(images_data, expanded_images,
               num_images, num_input_channels,
               image_size_y, image_size_x,
               num_modules_y, num_modules_x,
               kernel_size_y, kernel_size_x,
               padding_y, padding_x,
               stride_y, stride_x, i);

        if (!conv)
        {
            w += num_output_channels * input_size;
        }

        sgemm(false, false, true, 
              num_images, num_output_channels, input_size,
              1, expanded_images, num_images,
                 w, num_output_channels,
              0, expanded_target, num_images);

        for (int c=0; c<num_output_channels; ++c)
        {
            float* source = expanded_target + num_images * c;
            float* target = targets_data + num_images * (i + c * num_modules);
            for (int im=0; im<num_images; ++im)
            {
                target[im] = scaleTargets*target[im] + scaleOutput*source[im];
            }
        }
    }

    delete[] expanded_images;
    delete[] expanded_target;
}

void convDown(eigenmat* derivs, eigenmat* filters, eigenmat* targets,
              Shape4D &derivs_shape, Shape4D &filters_shape, Shape4D &targets_shape,
              ConvDesc &conv_desc, float scaleTargets, float scaleOutput, bool conv)
{
    int num_input_channels   = conv_desc.num_input_channels;
    int num_output_channels  = conv_desc.num_output_channels;
    int kernel_size_y        = conv_desc.kernel_size_y;
    int kernel_size_x        = conv_desc.kernel_size_x;
    int stride_y             = conv_desc.stride_y;
    int stride_x             = conv_desc.stride_x;
    int padding_y            = conv_desc.padding_y;
    int padding_x            = conv_desc.padding_x;
    int input_channel_begin  = conv_desc.input_channel_begin;
    int input_channel_end    = conv_desc.input_channel_end;
    int output_channel_begin = conv_desc.output_channel_begin;
    int output_channel_end   = conv_desc.output_channel_end;
    int num_groups           = conv_desc.num_groups;
    if (output_channel_end == 0)
        output_channel_end = num_output_channels;
    if (input_channel_end == 0)
        input_channel_end = num_input_channels;

    int num_output_channels2 = derivs_shape.shape[3];
    int num_modules_y        = derivs_shape.shape[2];
    int num_modules_x        = derivs_shape.shape[1];
    int num_images           = derivs_shape.shape[0];

    int num_input_channels2  = targets_shape.shape[3];
    int image_size_y         = targets_shape.shape[2];
    int image_size_x         = targets_shape.shape[1];
    int num_images2          = targets_shape.shape[0];

    int num_input_channels3  = filters_shape.shape[3];
    int kernel_size_y2       = filters_shape.shape[2];
    int kernel_size_x2       = filters_shape.shape[1];
    int num_output_channels3 = filters_shape.shape[0];

    int num_modules          = num_modules_y * num_modules_x;
    int filterModuleMult     = conv ? 1 : num_modules;
  
    // Consistency checks.
    assert(num_images == num_images2);
    assert(num_output_channels == num_output_channels2);
    assert(output_channel_end - output_channel_begin == num_output_channels3);
    assert(num_input_channels == num_input_channels2);
    assert(input_channel_end - input_channel_begin == num_input_channels3 / filterModuleMult);
    assert(num_images2 == targets->size[0]);
    assert(num_images == derivs->size[0]);
    assert(num_output_channels3 == filters->size[0]);
    assert(image_size_y * image_size_x * num_input_channels2 == targets->size[1]);
    assert(num_modules_y * num_modules_x * num_output_channels2 == derivs->size[1]);
    assert(kernel_size_y * kernel_size_x * num_input_channels3 * filterModuleMult == filters->size[1]);
    assert(kernel_size_y == kernel_size_y2);
    assert(kernel_size_x == kernel_size_x2);
    assert(num_input_channels % num_groups == 0);
    assert(num_groups == 1);
    assert(input_channel_begin  >= 0);
    assert(output_channel_begin >= 0);
    assert(input_channel_end    <= num_input_channels);
    assert(output_channel_end   <= num_output_channels);
    assert(input_channel_begin  <= input_channel_end);
    assert(output_channel_begin <= output_channel_end);
    num_input_channels = input_channel_end - input_channel_begin;
    num_output_channels = output_channel_end - output_channel_begin;
    assert(num_input_channels  > 0);
    assert(num_output_channels > 0);

    float* w = filters->data;
    float* derivs_data = derivs->data + output_channel_begin * num_modules * num_images;
    float* targets_data = targets->data + input_channel_begin * image_size_y * image_size_x * num_images;

    int input_size = kernel_size_y * kernel_size_x * num_input_channels;

    float *expanded_target = new float[num_images * input_size];
    float *expanded_derivs = new float[num_images * num_output_channels];

    for (int i=0; i<targets->size[0]*targets->size[1]; ++i)
    {
        targets->data[i] *= scaleTargets;
    }

    for (int i=0; i<num_modules; i++)
    {
        for (int c=0; c<num_output_channels; ++c)
        {
            int src_idx = num_images * (i + c * num_modules);
            int dst_idx = num_images * c;

            for (int im=0; im<num_images; ++im)
            {
                expanded_derivs[dst_idx + im] = derivs_data[src_idx + im];
            }
        }

        if (!conv)
        {
            w += num_output_channels * input_size;
        }

        sgemm(false, false, false, 
              num_images, kernel_size_x * kernel_size_y * num_input_channels, num_output_channels,
              scaleOutput, expanded_derivs, num_images,
              w, num_output_channels,
              0, expanded_target, num_images);

        contract(expanded_target, targets_data,
                 num_images, num_input_channels,
                 image_size_y, image_size_x,
                 num_modules_y, num_modules_x,
                 kernel_size_y, kernel_size_x,
                 padding_y, padding_x,
                 stride_y, stride_x, i);
    }

    delete[] expanded_derivs;
    delete[] expanded_target;
}

void convOutp(eigenmat* images, eigenmat* derivs, eigenmat* targets,
              Shape4D &images_shape, Shape4D &derivs_shape, Shape4D &targets_shape,
              ConvDesc &conv_desc, float scaleTargets, float scaleOutput, bool conv)
{
    int num_input_channels   = conv_desc.num_input_channels;
    int num_output_channels  = conv_desc.num_output_channels;
    int kernel_size_y        = conv_desc.kernel_size_y;
    int kernel_size_x        = conv_desc.kernel_size_x;
    int stride_y             = conv_desc.stride_y;
    int stride_x             = conv_desc.stride_x;
    int padding_y            = conv_desc.padding_y;
    int padding_x            = conv_desc.padding_x;
    int input_channel_begin  = conv_desc.input_channel_begin;
    int input_channel_end    = conv_desc.input_channel_end;
    int output_channel_begin = conv_desc.output_channel_begin;
    int output_channel_end   = conv_desc.output_channel_end;
    int num_groups           = conv_desc.num_groups;
    if (output_channel_end == 0)
        output_channel_end = num_output_channels;
    if (input_channel_end == 0)
        input_channel_end = num_input_channels;

    int num_output_channels2 = derivs_shape.shape[3];
    int num_modules_y        = derivs_shape.shape[2];
    int num_modules_x        = derivs_shape.shape[1];
    int num_images           = derivs_shape.shape[0];

    int num_input_channels2  = images_shape.shape[3];
    int image_size_y         = images_shape.shape[2];
    int image_size_x         = images_shape.shape[1];
    int num_images2          = images_shape.shape[0];

    int num_input_channels3Mult = targets_shape.shape[3];
    int kernel_size_y2       = targets_shape.shape[2];
    int kernel_size_x2       = targets_shape.shape[1];
    int num_output_channels3 = targets_shape.shape[0];

    int num_modules          = num_modules_y * num_modules_x;
    int filterModuleMult     = conv ? 1 : num_modules;

    // Consistency checks.
    assert(num_images == num_images2);
    assert(num_output_channels == num_output_channels2);
    assert(output_channel_end - output_channel_begin == num_output_channels3);
    assert(num_input_channels == num_input_channels2);
    assert(input_channel_end - input_channel_begin == num_input_channels3Mult / filterModuleMult);
    assert(num_images2 == images->size[0]);
    assert(num_images == derivs->size[0]);
    assert(num_output_channels3 == targets->size[0]);
    assert(image_size_y * image_size_x * num_input_channels2 == images->size[1]);
    assert(num_modules_y * num_modules_x * num_output_channels2 == derivs->size[1]);
    assert(kernel_size_y2 * kernel_size_x2 * num_input_channels3Mult == targets->size[1]);
    assert(kernel_size_y == kernel_size_y2);
    assert(kernel_size_x == kernel_size_x2);
    assert(num_input_channels % num_groups == 0);
    assert(num_groups == 1);
    assert(input_channel_begin  >= 0);
    assert(output_channel_begin >= 0);
    assert(input_channel_end    <= num_input_channels);
    assert(output_channel_end   <= num_output_channels);
    assert(input_channel_begin  <= input_channel_end);
    assert(output_channel_begin <= output_channel_end);
    if (output_channel_end == 0)
        output_channel_end = num_output_channels;
    if (input_channel_end == 0)
        input_channel_end = num_input_channels;
    num_input_channels = input_channel_end - input_channel_begin;
    num_output_channels = output_channel_end - output_channel_begin;
    assert(num_input_channels  > 0);
    assert(num_output_channels > 0);

    float* dw = targets->data;
    float* images_data = images->data + input_channel_begin * image_size_y * image_size_x * num_images;
    float* derivs_data = derivs->data + output_channel_begin * num_modules * num_images;

    int input_size = kernel_size_y * kernel_size_x * num_input_channels;

    float *expanded_images = new float[num_images * input_size];
    float *expanded_derivs = new float[num_images * num_output_channels];

    for (int i=0; i<targets->size[0]*targets->size[1]; ++i)
    {
        targets->data[i] *= scaleTargets;
    }

    for (int i=0; i<num_modules; i++)
    {
        for (int c=0; c<num_output_channels; ++c)
        {
            int src_idx = num_images * (i + c * num_modules);
            int dst_idx = num_images * c;

            for (int im=0; im<num_images; ++im)
            {
                expanded_derivs[dst_idx + im] = derivs_data[src_idx + im];
            }
        }

        expand(images_data, expanded_images,
               num_images, num_input_channels,
               image_size_y, image_size_x,
               num_modules_y, num_modules_x,
               kernel_size_y, kernel_size_x,
               padding_y, padding_x,
               stride_y, stride_x, i);

        if (!conv)
        {
            dw += num_output_channels * input_size;
        }

        sgemm(false, true, false, 
              num_output_channels, input_size, num_images,
              scaleOutput, expanded_derivs, num_images,
              expanded_images, num_images,
              1, dw, num_output_channels);
    }

    delete[] expanded_derivs;
    delete[] expanded_images;
}

void ResponseNormCrossMap(eigenmat* images, eigenmat* targets,
                          int num_filters, int sizeF, float addScale, float powScale, bool blocked)
{
    int num_locs = (images->size[0] * images->size[1]) / num_filters;

    float* data = images->data;
    float* target = targets->data;
    for (int loc_id=0; loc_id<num_locs; ++loc_id)
    {
        float sum = 0;
        int prev_start = 0, prev_end = 0, start, end;
        for (int j=0; j<num_filters; j++)
        {
            start = blocked ? (j / sizeF) * sizeF : -sizeF/2 + j;
            end = MIN(num_filters, start + sizeF);
            start = MAX(0, start);
            for (int i=prev_start; i<start; i++)
            {
                float val = data[i*num_locs + loc_id];
                sum -= val*val;
            }
            for (int i=prev_end; i<end; i++)
            {
                float val = data[i*num_locs + loc_id];
                sum += val*val;
            }
            int idx = j*num_locs + loc_id;
            target[idx] = data[idx] * pow(1 + addScale * sum, -powScale);

            prev_start = start;
            prev_end = end;
        }
    }
}

void ResponseNormCrossMapUndo(eigenmat* outGrads, eigenmat* inputs, eigenmat* targets,
                              int num_filters, int sizeF, float addScale, float powScale, bool blocked)
{
    int num_locs = (inputs->size[0] * inputs->size[1]) / num_filters;
    float *denoms = new float[num_locs * num_filters];

    float* data = inputs->data;
    for (int loc_id=0; loc_id<num_locs; ++loc_id)
    {
        int prev_start = 0, prev_end = 0, start, end;
        float sum = 0;
        for (int j=0; j<num_filters; j++)
        {
            start = blocked ? (j / sizeF) * sizeF : -sizeF/2 + j;
            end = MIN(num_filters, start + sizeF);
            start = MAX(0, start);
            for (int i=prev_start; i<start; i++)
            {
                float val = data[i*num_locs + loc_id];
                sum -= val*val;
            }
            for (int i=prev_end; i<end; i++)
            {
                float val = data[i*num_locs + loc_id];
                sum += val*val;
            }
            denoms[j*num_locs + loc_id] = pow(1 + addScale * sum, -powScale - 1);

            prev_start = start;
            prev_end = end;
        }
    }

    float* deriv = outGrads->data;
    float* target = targets->data;
    for (int loc_id=0; loc_id<num_locs; ++loc_id)
    {
        float sum = 0;
        int prev_start = 0, prev_end = 0, start, end;
        for (int j=0; j<num_filters; j++)
        {
            start = blocked ? (j / sizeF) * sizeF : -sizeF + sizeF/2 + j + 1;
            end = MIN(num_filters, start + sizeF);
            start = MAX(0, start);
            for (int i=prev_start; i<start; i++)
            {
                int idx = i*num_locs + loc_id;
                sum -= deriv[idx] * data[idx] * denoms[idx];
            }
            for (int i=prev_end; i<end; i++)
            {
                int idx = i*num_locs + loc_id;
                sum += deriv[idx] * data[idx] * denoms[idx];
            }
            int idx = j*num_locs + loc_id;
            target[idx] = deriv[idx] * pow(denoms[idx], powScale / (powScale + 1)) - 2 * addScale * powScale * data[idx] * sum;

            prev_start = start;
            prev_end = end;
        }
    }

    delete[] denoms;
}

