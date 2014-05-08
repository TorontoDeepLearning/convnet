#include "cpuconv.h"
#include <iostream>
#include <cfloat>
#include <cmath>

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
  for (int i = 0; i < rows_ && i < 10; i++) {
    for (int j = 0; j < cols_ && j < 10; j++) {
      cout << data_[j + i * cols_] << " ";
    }
    cout << endl;
  }
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
  int out_y_ind, out_x_ind, inp_y, inp_x, inp_y_ind, inp_x_ind, filter_y_ind,
      filter_x_ind, image_ind, target_ind, filter_ind;
  const int out_height = (inp_height + 2 * padding_y - kernel_height ) / stride_y + 1;
  const int out_width = (inp_width + 2 * padding_x - kernel_width ) / stride_x + 1;
  for (int i = 0; i < num_images; i++) {
    for (int out_y = 0; out_y < out_height; out_y++) {
      out_y_ind = out_y + out_height * i;
      for (int out_x = 0; out_x < out_width; out_x++) {
        out_x_ind = out_x + out_width * out_y_ind;
        for (int f = 0; f < num_filters; f++) {
          target_ind = out_x_ind * num_filters + f;

          // Do the convolution.
          float res = 0;
          for (int k_y = 0; k_y < kernel_height; k_y++) {
            inp_y = out_y * stride_y - padding_y + k_y;
            if (inp_y < 0 || inp_y > inp_height) continue;
            inp_y_ind = inp_y + inp_height * i;
            filter_y_ind = k_y + kernel_height * f;
            for (int k_x = 0; k_x < kernel_width; k_x++) {
              inp_x = out_x * stride_x - padding_x + k_x;
              if (inp_x < 0 || inp_x > inp_width) continue;
              inp_x_ind = inp_x + inp_width * inp_y_ind;
              filter_x_ind = k_x + kernel_width * filter_y_ind;
              image_ind = inp_x_ind * num_colors;
              filter_ind = filter_x_ind * num_colors;
              for (int c = 0; c < num_colors; c++) {
                res += images[image_ind + c] * filters[filter_ind + c];
              }
            }
          }

          targets[target_ind] = scale_targets * targets[target_ind] + scale_outputs * res;

        }
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
  int out_y_ind, out_x_ind, inp_y, inp_x, inp_y_ind, image_ind, target_ind;
  const int out_height = (inp_height + 2 * padding_y - kernel_height ) / stride_y + 1;
  const int out_width = (inp_width + 2 * padding_x - kernel_width ) / stride_x + 1;
  for (int i = 0; i < num_images; i++) {
    for (int out_y = 0; out_y < out_height; out_y++) {
      out_y_ind = out_y + out_height * i;
      for (int out_x = 0; out_x < out_width; out_x++) {
        out_x_ind = out_x + out_width * out_y_ind;
        for (int f = 0; f < num_filters; f++) {
          target_ind = out_x_ind * num_filters + f;

          // Do the maxpool.
          float res = -FLT_MAX;
          for (int k_y = 0; k_y < kernel_height; k_y++) {
            inp_y = out_y * stride_y - padding_y + k_y;
            if (inp_y < 0 || inp_y > inp_height) continue;
            inp_y_ind = inp_y + inp_height * i;
            for (int k_x = 0; k_x < kernel_width; k_x++) {
              inp_x = out_x * stride_x - padding_x + k_x;
              if (inp_x < 0 || inp_x > inp_width) continue;
              image_ind = f + num_filters * (inp_x + inp_width * inp_y_ind);
              if (res < images[image_ind]) {
                res = images[image_ind];
              }
            }
          }

          targets[target_ind] = scale_targets * targets[target_ind] + scale_outputs * res;

        }
      }
    }
  }
}

// images : num_filters * num_locs
// targets : num_filters * num_locs
void CPUMatrix::ResponseNormCrossMap(
    const float* images, float* targets,
    const int num_locs, const int num_filters,
    const int sizeF, const bool blocked,
    const bool add_scale, const bool pow_scale,
    const float scale_outputs,
    const float scale_targets) {

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
      targets[f + i * num_filters] = images[f + i * num_filters] * sum;
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
    for (int j = 0; j < num_outputs; j++) {
      float res = 0;
      for (int k = 0; k < num_inputs; k++) {
        res += inputs[k + i * num_inputs] * weights[k + j * num_inputs];
      }
      int target_ind = j + i * num_outputs;
      targets[target_ind] = scale_targets * targets[target_ind] + scale_outputs * res;
    }
  }
}

void CPUMatrix::AddBias(const float* inputs, const float* bias, float* outputs, const int num_images, const int num_dims) {
  for (int i = 0; i < num_images; i++) {
    for (int j = 0; j < num_dims; j++) {
      outputs[j + i * num_dims] = inputs[j + i * num_dims] + bias[j];
    }
  }
}

void CPUMatrix::UpperBound(const float* inputs, float* outputs, const int length, const float limit) {
  for (int i = 0; i < length; i++) {
    outputs[i] = inputs[i] > limit ? limit : inputs[i];
  }
}

void CPUMatrix::LowerBound(const float* inputs, float* outputs, const int length, const float limit) {
  for (int i = 0; i < length; i++) {
    outputs[i] = inputs[i] < limit ? limit : inputs[i];
  }
}

void CPUMatrix::Argmax(const float* inputs, int* outputs, const int num_images, const int num_dims) {
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
