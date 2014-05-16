#include "cpuconv.h"
#include <iostream>
#include <cfloat>
#include <cmath>
#include <chrono>

using namespace std;

CPUMatrix::CPUMatrix(): data_(NULL), rows_(0), cols_(0) {
}

CPUMatrix::CPUMatrix(const int rows, const int cols): rows_(rows), cols_(cols) {
  AllocateMemory(rows, cols);
}

void CPUMatrix::FreeMemory() {
  if (rows_ * cols_ > 0) delete data_;
}

void CPUMatrix::AllocateMemory(int rows, int cols) {
  rows_ = rows;
  cols_ = cols;
  data_ = new float[rows * cols];
}

void CPUMatrix::Print() {
  Print(rows_, cols_);
}

void CPUMatrix::Print(int rows, int cols) {
  for (int i = 0; i < rows && i < 10; i++) {
    for (int j = 0; j < cols && j < 10; j++) {
      cout << data_[j + i * cols] << " ";
    }
    if (!(i == 9 || i == rows - 1)) cout << endl;
  }
  float max = -FLT_MAX;
  for (int j = 0; j < rows * cols; j++) if (max < data_[j]) max = data_[j];
  cout << "... Max " << max << endl;
}

void CPUMatrix::Transpose(const float* i_data, float* o_data, int num_filters, int kernel_width, int kernel_height, int num_colors) {
  int f, x, y, c;
  long i, target_ind;
  long long size = num_filters * kernel_width * kernel_height * num_colors;
  for (long ind = 0; ind < size; ind++) {
    i = ind;
    f = i % num_filters; i /= num_filters;
    x = i % kernel_width; i /= kernel_width;
    y = i % kernel_height; i /= kernel_height;
    c = i;
    target_ind = c + num_colors * (x + kernel_width * (y + kernel_height * f));
    o_data[target_ind] = i_data[ind];
  }
}

void CPUMatrix::SetZero(float* data, int length) {
  for (int i = 0; i < length; i++) data[i] = 0;
}

// images : colors * inp_width * inp_height * numimages
// filters: colors * kernel_x * kernel_y * num_filters
// targets : num_filters * out_width * out_height * numimages
void CPUMatrix::ConvUp(const float* images, const float* filters, float* targets,
               const int num_images, const int num_colors, const int num_filters,
               const int inp_width, const int inp_height,
               const int kernel_width, const int kernel_height,
               const int stride_x, const int stride_y,
               const int padding_x, const int padding_y,
               const float scale_outputs,
               const float scale_targets) {
  const int out_height = (inp_height + 2 * padding_y - kernel_height ) / stride_y + 1;
  const int out_width = (inp_width + 2 * padding_x - kernel_width ) / stride_x + 1;

  const int chunk = 16;

  #pragma omp parallel for
  for (long loc = 0; loc < num_images * out_height * out_width; loc++) {
    long i = loc;
    int out_x = i % out_width; i /= out_width;
    int out_y = i % out_height; i /= out_height;
    long image_ind, target_ind, filter_ind;

    for (int f = 0; f < num_filters; f+=chunk) {

      // Do the convolution.
      float res[chunk];
      for (int ff = 0; ff < chunk; ff++) res[ff] = 0;

      for (int k_y = 0, inp_y = out_y * stride_y -padding_y; k_y < kernel_height && inp_y < inp_height; k_y++, inp_y++) {
        if (inp_y < 0) continue;
        for (int k_x = 0, inp_x = out_x * stride_x -padding_x; k_x < kernel_width && inp_x < inp_width; k_x++, inp_x++) {
          if (inp_x < 0) continue;
          image_ind = num_colors * (inp_x + inp_width * (inp_y + inp_height * i));
          for (int c = 0; c < num_colors; c++) {

            for (int ff = 0; ff < chunk; ff++) {
              filter_ind = c + num_colors * (k_x + kernel_width * (k_y + kernel_height * (f + ff)));
              res[ff] += images[image_ind + c] * filters[filter_ind];
            }

          }
        }
      }

      for (int ff = 0; ff < chunk; ff++) {
        target_ind = f + ff + num_filters * loc;
        targets[target_ind] = scale_targets * targets[target_ind] + scale_outputs * res[ff];
      }
    }
  }
}


// images : colors * inp_width * inp_height * numimages
// targets : num_filters * out_width * out_height * numimages
void CPUMatrix::MaxPool(const float* images, float* targets,
               const int num_images, const int num_filters,
               const int inp_width, const int inp_height,
               const int kernel_width, const int kernel_height,
               const int stride_x, const int stride_y,
               const int padding_x, const int padding_y,
               const float scale_outputs,
               const float scale_targets) {
  const int out_height = (inp_height + 2 * padding_y - kernel_height ) / stride_y + 1;
  const int out_width = (inp_width + 2 * padding_x - kernel_width ) / stride_x + 1;

  #pragma omp parallel for
  for (long loc = 0; loc < num_images * out_height * out_width; loc++) {
    long i = loc;
    int out_x = i % out_width; i /= out_width;
    int out_y = i % out_height; i /= out_height;
    long image_ind, target_ind;

    for (int f = 0; f < num_filters; f++) {
      target_ind = f + num_filters * loc;

      // Do the maxpool.
      float res = -FLT_MAX;
      for (int k_y = 0, inp_y = out_y * stride_y - padding_y + k_y;
          k_y < kernel_height && inp_y < inp_height; k_y++, inp_y++) {
        if (inp_y < 0) continue;
        for (int k_x = 0, inp_x = out_x * stride_x - padding_x + k_x;
            k_x < kernel_width && inp_x < inp_width; k_x++, inp_x++) {
          if (inp_x < 0) continue;
          image_ind = f + num_filters * (inp_x + inp_width * (inp_y + inp_height * i));
          if (res < images[image_ind]) res = images[image_ind];
        }
      }

      targets[target_ind] = scale_targets * targets[target_ind] + scale_outputs * res;

    }
  }
}

// images : num_filters * num_locs
// targets : num_filters * num_locs
void CPUMatrix::ResponseNormCrossMap(
    const float* images, float* targets,
    const int num_locs, const int num_filters,
    const int sizeF, const bool blocked,
    const float add_scale, const float pow_scale,
    const float scale_outputs,
    const float scale_targets) {

  #pragma omp parallel for
  for (int i = 0; i < num_locs; i++) {
    for (int f = 0; f < num_filters; f++) {
      int start = blocked ? ((f / sizeF) * sizeF) : (f - sizeF/2);
      int end = start + sizeF;
      if (start < 0) start = 0;
      if (end > num_filters) end = num_filters;
      float sum = 0, act;
      for (int j = start; j < end; j++) {
        act = images[j + i * num_filters];
        sum += act * act;
      }
      sum = pow(1 + add_scale * sum, -pow_scale);
      targets[f + i * num_filters] 
        = scale_targets * targets[f + i * num_filters] +
          scale_outputs * images[f + i * num_filters] * sum;
    }
  }
}

// inputs: num_inputs * num_images
// weights:  num_inputs * num_outputs
// outputs: num_outputs * num_images
void CPUMatrix::FCUp(
    const float* inputs, const float* weights, float* targets,
    const int num_images, const int num_outputs, const int num_inputs,
    const float scale_outputs, const float scale_targets) {

  for (int i = 0; i < num_images; i++) {
    #pragma omp parallel for
    for (int j = 0; j < num_outputs; j++) {
      float res = 0;
      for (int k = 0; k < num_inputs; k++) {
        /*
        __m128 a, b;
        a = _mm_load_ps(inputs + k+ i * num_inputs);
        b = _mm_load_ps(weights + k+ j * num_inputs);
        __m128 res128 = _mm_dp_ps(a, b, 0xF1);
        union { __m128 v; float f[4]; } uf;
        uf.v = res128;
        res += uf.f[0];
        */
        res += inputs[k + i * num_inputs] * weights[k + j * num_inputs];
      }
      int target_ind = j + i * num_outputs;
      targets[target_ind] = scale_targets * targets[target_ind] + scale_outputs * res;
    }
  }
}

void CPUMatrix::AddBias(const float* inputs, const float* bias, float* outputs, const int num_images, const int num_dims) {
  int length = num_dims * num_images;
  #pragma omp parallel for if(length > 10000)
  for (int i = 0; i < length; i++) {
    outputs[i] = inputs[i] + bias[i % num_dims];
  }
}

void CPUMatrix::UpperBound(const float* inputs, float* outputs, const int length, const float limit) {
  #pragma omp parallel for if(length > 10000)
  for (int i = 0; i < length; i++) {
    outputs[i] = inputs[i] > limit ? limit : inputs[i];
  }
}

void CPUMatrix::LowerBound(const float* inputs, float* outputs, const int length, const float limit) {
  #pragma omp parallel for if(length > 10000)
  for (int i = 0; i < length; i++) {
    outputs[i] = inputs[i] < limit ? limit : inputs[i];
  }
}

void CPUMatrix::Argmax(const float* inputs, int* outputs, const int num_images, const int num_dims) {
  #pragma omp parallel for if(num_images > 1000)
  for (int i = 0; i < num_images; i++) {
    const float *inp = inputs + i * num_dims;
    float max = -FLT_MAX;
    int argmax = -1;
    for (int j = 0; j < num_dims; j++) {
      if (max < inp[j]) {
        max = inp[j];
        argmax = j;
      }
    }
    outputs[i] = argmax;
  }
}

void CPUMatrix::Softmax(const float* inputs, float* outputs, const int num_images, const int num_dims) {
  #pragma omp parallel for if(num_images > 1000)
  for (int i = 0; i < num_images; i++) {
    const float *inp = inputs + i * num_dims;
    float *out = outputs + i * num_dims;
    float max = -FLT_MAX, sum = 0;
    for (int j = 0; j < num_dims; j++) if (max < inp[j]) max = inp[j];
    for (int j = 0; j < num_dims; j++) {
      out[j] = exp(inp[j] - max);
      sum += out[j];
    }
    for (int j = 0; j < num_dims; j++) out[j] /= sum;
  }
}

void CPUMatrix::Logistic(const float* inputs, float* outputs, const int length) {
  #pragma omp parallel for if(length > 10000)
  for (int i = 0; i < length; i++) outputs[i] = 1 / (1 + exp(-inputs[i]));
}

void CPUMatrix::ReadHDF5(hid_t file, float* mat, int size, const string& name) {
  hid_t dataset, dataspace;
  dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  dataspace = H5Dget_space(dataset);
  const int ndims = H5Sget_simple_extent_ndims(dataspace);
  hsize_t dims_out[2];
  H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
  int rows = (ndims == 1) ? 1 : dims_out[1];
  int datasize = dims_out[0] * rows;
  if (size != datasize) {
    cerr << "Dimension mismatch: Expected "
         << size << " Got " << rows << "-" << dims_out[0] << endl;
    exit(1);
  }
  H5Dread(dataset, H5T_NATIVE_FLOAT, dataspace, dataspace, H5P_DEFAULT, mat);
  H5Dclose(dataset);
}

void CPUMatrix::ReadHDF5Shape(hid_t file, const string& name, int* rows, int* cols) {
  hid_t dataset, dataspace;
  dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  dataspace = H5Dget_space(dataset);
  const int ndims = H5Sget_simple_extent_ndims(dataspace);
  hsize_t dims_out[2];
  H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
  *rows = dims_out[0];
  *cols = (ndims == 1) ? 1: dims_out[1];
  H5Dclose(dataset);
}
