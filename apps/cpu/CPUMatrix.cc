#include "CPUMatrix.h"

#include "eigenmat.h"

#include <iostream>
#include <cfloat>
#include <cmath>
#include <chrono>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#include <common.h>
#endif

using namespace std;

string GetStringError(int err_code)
{
  if (err_code == -1)
    return "Incompatible matrix dimensions.";
  if (err_code == -2)
    return "CUBLAS error.";
  if (err_code == -3)
    return "CUDA error ";
  if (err_code == -4)
    return "Operation not supported on views.";
  if (err_code == -5)
    return "Operation not supported on transposed matrices.";
  if (err_code == -6)
    return "";
  if (err_code == -7)
    return "Incompatible transposedness.";
  if (err_code == -8)
    return "Matrix is not in device memory.";
  if (err_code == -9)
    return "Operation not supported.";
  return "Some error";
}

rnd_struct_e rnde_;

CPUMatrix::CPUMatrix()
{
    mat_ = new eigenmat;
    mat_->data = NULL;
    mat_->size[0] = 0;
    mat_->size[1] = 0;
    mat_->is_trans = 0;
    mat_->owns_data = 0;

    mat_t_ = new eigenmat;
    *mat_t_ = *mat_;
}

CPUMatrix::CPUMatrix(const int rows, const int cols)
{
    CPUMatrix();

    AllocateMemory(rows, cols);
}

CPUMatrix::~CPUMatrix()
{
    FreeMemory();

    delete mat_;
    delete mat_t_;
}

void CPUMatrix::FreeMemory()
{
    if (mat_->size[0] * mat_->size[1] > 0)
        delete[] mat_->data;
}

void CPUMatrix::AllocateMemory(int rows, int cols)
{
    mat_->size[0] = rows;
    mat_->size[1] = cols;
    mat_->data = new float[rows * cols];
}

void CPUMatrix::Set(const float val)
{
  int err_code = assign_scalar(mat_, val);
  if (err_code != 0) {
    cerr << "Error: Could not set to scalar : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void CPUMatrix::Print()
{
  int rows = mat_->size[0];
  int cols = mat_->size[1];
  for (int i = 0; i < rows && i < 10; i++) {
    for (int j = 0; j < cols && j < 10; j++) {
      cout <<  mat_->data[j + i * cols] << " ";
    }
    if (!(i == 9 || i == rows - 1)) cout << endl;
  }
  float max = -FLT_MAX;
  for (int j = 0; j < rows * cols; j++)
    if (max <  mat_->data[j])
      max = mat_->data[j];
  cout << "... Max " << max << endl;
}

float* CPUMatrix::GetData()
{
    return mat_->data;
}

int CPUMatrix::GetRows() const
{
    return mat_->size[0];
}

int CPUMatrix::GetCols() const
{
    return mat_->size[1];
}


int CPUMatrix::GetNumEls() const
{
    return mat_->size[1] * mat_->size[0];
}

void CPUMatrix::Add(float val)
{
    add_scalar(mat_, val, mat_);
}

void CPUMatrix::Add(CPUMatrix& m)
{
    add_elementwise(mat_, m.GetMat(), mat_);
}

void CPUMatrix::Add(CPUMatrix& m, float alpha)
{
    add_mult(mat_, m.GetMat(), alpha);
}

void CPUMatrix::AddRowVec(CPUMatrix& v)
{
    add_row_vec(mat_, v.GetMat(), mat_);
}

void CPUMatrix::AddRowVec(CPUMatrix& v, float alpha)
{
    add_row_mult(mat_, v.GetMat(), mat_, alpha);
}

void CPUMatrix::AddColVec(CPUMatrix& v, float alpha)
{
    add_col_mult(mat_, v.GetMat(), mat_, alpha);
}

void CPUMatrix::MultByRowVec(CPUMatrix& val)
{
    mult_by_row_vec(mat_, val.GetMat(), mat_);
}

void CPUMatrix::DivideByColVec(CPUMatrix& v)
{
    div_by_col_vec(mat_, v.GetMat(), mat_);
}

// self *= val
void CPUMatrix::Mult(float val)
{
    mult_by_scalar(mat_, val, mat_);
}

void CPUMatrix::Mult(CPUMatrix& val)
{
    mult_elementwise(mat_, val.GetMat(), mat_);
}

void CPUMatrix::Divide(float val)
{
    divide_by_scalar(mat_, val, mat_);
}

void CPUMatrix::Divide(CPUMatrix& val)
{
    divide_elementwise(mat_, val.GetMat(), mat_);
}

void CPUMatrix::Subtract(CPUMatrix& m, CPUMatrix& target)
{
  int err_code = subtract_elementwise(mat_, m.GetMat(), target.GetMat());
  if (err_code != 0) {
    cerr << "Error in subtract." << endl;
    exit(1);
  }
}

void CPUMatrix::LowerBound(float val)
{
    lower_bound_scalar(mat_, val, mat_);
}

void CPUMatrix::Sqrt()
{
    apply_sqrt(mat_, mat_);
}

// c = alpha * c + beta * a * b
void CPUMatrix::Dot(CPUMatrix& a, CPUMatrix& b, CPUMatrix& c, float alpha, float beta)
{
    dot(a.GetMat(), b.GetMat(), c.GetMat(), alpha, beta);
}

// c = alpha * c + beta * T(a) * T(b)
void CPUMatrix::Dot(CPUMatrix& a, CPUMatrix& b, CPUMatrix& c, float alpha, float beta,
                    bool transpose_a, bool transpose_b)
{
    eigenmat* a_mat = transpose_a ? a.GetMatTranspose() : a.GetMat();
    eigenmat* b_mat = transpose_b ? b.GetMatTranspose() : b.GetMat();
    dot(a_mat, b_mat, c.GetMat(), alpha, beta);
}

void CPUMatrix::ApplyDerivativeOfReLU(CPUMatrix& state)
{
    apply_rectified_linear_deriv(mat_, state.GetMat(), mat_);
}

void CPUMatrix::ApplyLogistic()
{
    apply_sigmoid(mat_, mat_);
}

void CPUMatrix::ApplyDerivativeOfLogistic(CPUMatrix& state)
{
    apply_logistic_deriv(mat_, state.GetMat(), mat_);
}

float CPUMatrix::EuclidNorm()
{
    return euclid_norm(mat_); // TODO: return error code?
}

float CPUMatrix::VDot(CPUMatrix& m)
{
    int err;
    return vdot(mat_, m.GetMat(), &err);
}

void CPUMatrix::CopyTranspose(CPUMatrix& m)
{
    copy_transpose(mat_, m.GetMat());
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

#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
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

#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
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

#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
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

#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_images,
              num_outputs, num_inputs, scale_outputs, inputs, num_inputs,
              weights, num_inputs, scale_targets, targets, num_outputs);
#else
  for (int i = 0; i < num_images; i++) {
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int j = 0; j < num_outputs; j++) {
      float res = 0;
      for (int k = 0; k < num_inputs; k++) {
        res += inputs[k + i * num_inputs] * weights[k + j * num_inputs];
      }
      int target_ind = j + i * num_outputs;
      targets[target_ind] = scale_targets * targets[target_ind] + scale_outputs * res;
    }
  }
#endif
}

void CPUMatrix::AddBias(const float* inputs, const float* bias, float* outputs, const int num_images, const int num_dims) {
  int length = num_dims * num_images;
#ifdef USE_OPENMP
  #pragma omp parallel for if(length > 10000)
#endif
  for (int i = 0; i < length; i++) {
    outputs[i] = inputs[i] + bias[i % num_dims];
  }
}

void CPUMatrix::UpperBound(const float* inputs, float* outputs, const int length, const float limit) {
#ifdef USE_OPENMP
  #pragma omp parallel for if(length > 10000)
#endif
  for (int i = 0; i < length; i++) {
    outputs[i] = inputs[i] > limit ? limit : inputs[i];
  }
}

void CPUMatrix::LowerBound(const float* inputs, float* outputs, const int length, const float limit) {
#ifdef USE_OPENMP
  #pragma omp parallel for if(length > 10000)
#endif
  for (int i = 0; i < length; i++) {
    outputs[i] = inputs[i] < limit ? limit : inputs[i];
  }
}

void CPUMatrix::Argmax(const float* inputs, int* outputs, const int num_images, const int num_dims) {
#ifdef USE_OPENMP
  #pragma omp parallel for if(num_images > 1000)
#endif
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
#ifdef USE_OPENMP
  #pragma omp parallel for if(num_images > 1000)
#endif
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
#ifdef USE_OPENMP
  #pragma omp parallel for if(length > 10000)
#endif
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
