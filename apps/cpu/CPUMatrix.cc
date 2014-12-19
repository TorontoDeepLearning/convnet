#include "CPUMatrix.h"
#include "eigenmat.h"
#include "../../src/util.h"

#include <iostream>
#include <cfloat>
#include <chrono>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#include <common.h>
#endif

using namespace std;

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
    SetupTranspose();

    shape_.shape[0] = 0;
    shape_.shape[1] = 0;
    shape_.shape[2] = 0;
    shape_.shape[3] = 0;
}

CPUMatrix::CPUMatrix(const int rows, const int cols)
{
    CPUMatrix();

    AllocateMainMemory(rows, cols);
}

CPUMatrix::~CPUMatrix()
{
    FreeMemory();

    delete mat_;
    delete mat_t_;
}

void CPUMatrix::Tie(CPUMatrix &m)
{
    cout << "Tying" << endl;
    *mat_ = *(m.GetMat());
    *mat_t_ = *(m.GetMatTranspose());
}

void CPUMatrix::SetupTranspose()
{
    *mat_t_ = *mat_;
    mat_t_->is_trans = 1 - mat_->is_trans;
}

void CPUMatrix::SetShape4D(int d1, int d2, int d3, int d4)
{
    shape_.shape[0] = d1;
    shape_.shape[1] = d2;
    shape_.shape[2] = d3;
    shape_.shape[3] = d4;
}

void CPUMatrix::SetShape4D_like(CPUMatrix& mat)
{
    Shape4D &s = mat.GetShape4D();
    SetShape4D(s.shape[0], s.shape[1], s.shape[2], s.shape[3]);
}

Shape4D& CPUMatrix::GetShape4D()
{
    return shape_;
}

void CPUMatrix::AllocateMainMemory(const size_t rows, const size_t cols)
{
    mat_->size[0] = rows;
    mat_->size[1] = cols;
    mat_->data = new float[rows * cols];
}

void CPUMatrix::FreeMemory()
{
    if (mat_->size[0] * mat_->size[1] > 0)
    {
        delete[] mat_->data;
    }
}

void CPUMatrix::Set(const float val)
{
    int err_code = assign_scalar(mat_, val);
    if (err_code != 0)
    {
        cerr << "Error: Could not set to scalar : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void CPUMatrix::Set(CPUMatrix& val)
{
    int err_code = copy_on_device(val.GetMat(), mat_); // source, dest.
    if (err_code != 0)
    {
        cerr << "Error: Could not set to val : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void CPUMatrix::CopyFromMainMemory(CPUMatrix& mat)
{
    int err_code = copy_on_device(mat.GetMat(), mat_); // source, dest.
    if (err_code != 0)
    {
        cerr << "Error: Could not set to val : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void CPUMatrix::CopyP2PAsync(CPUMatrix& val)
{
    int err_code = copy_on_device(val.GetMat(), mat_); // source, dest.
    if (err_code != 0)
    {
        cerr << "Error: Could not set to val : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void CPUMatrix::GetSlice(CPUMatrix& slice, size_t start, size_t end)
{
    get_slice(mat_, slice.GetMat(), start, end);
    slice.SetupTranspose();
}

void CPUMatrix::FillWithRand()
{
    int err_code = fill_with_rand(&rnde_, mat_);
    if (err_code != 0)
    {
        cerr << "Error: Could not fill with rand : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void CPUMatrix::FillWithRandn()
{
    int err_code = fill_with_randn(&rnde_, mat_);
    if (err_code != 0)
    {
        cerr << "Error: Could not fill with randn : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void CPUMatrix::Reshape(const size_t rows, const size_t cols)
{
    reshape(mat_, rows, cols);
    *mat_t_ = *mat_;
    mat_t_->is_trans = 1;
}

bool CPUMatrix::CheckNaN()
{
    float* data = mat_->data;
    bool is_nan = false;
    size_t i = 0;
    for (; i<mat_->size[0]*mat_->size[1] && !is_nan; i++)
    {
        is_nan = !isfinite(data[i]);
    }
    if (is_nan)
    {
        cout << "Nan at location " << i << " row " << i % mat_->size[0] << " col " << i / mat_->size[0] << endl;
    }
    return is_nan;
}

void CPUMatrix::WriteHDF5(hid_t file, const string& name)
{
    // cols, rows swapped because cudamat is col major, but hdf5 is row major.
    WriteHDF5CPU(file, mat_->data, mat_->size[1], mat_->size[0], name);
}

void CPUMatrix::ReadHDF5(hid_t file, const string& name)
{
    ReadHDF5CPU(file, mat_->data, mat_->size[0] * mat_->size[1], name);
}

void CPUMatrix::AllocateAndReadHDF5(hid_t file, const string& name)
{
    int rows, cols;
    ReadHDF5Shape(file, name, &rows, &cols);
    AllocateMainMemory(rows, cols);
    ReadHDF5(file, name);
}

void CPUMatrix::Print()
{
    int rows = mat_->size[0];
    int cols = mat_->size[1];
    for (int i = 0; i < rows && i < 10; i++)
    {
        for (int j = 0; j < cols && j < 10; j++)
        {
            cout <<  mat_->data[j + i * cols] << " ";
        }
        if (!(i == 9 || i == rows - 1))
        {
            cout << endl;
        }
    }
    float max = -FLT_MAX;
    for (int j = 0; j < rows * cols; j++)
    {
        if (max <  mat_->data[j])
        {
            max = mat_->data[j];
        }
    }
    cout << "... Max " << max << endl;
}

string CPUMatrix::GetShapeString()
{
    stringstream ss;
    ss << mat_->size[0] << " " << mat_->size[1];
    return ss.str();
}

string CPUMatrix::GetShape4DString()
{
    stringstream ss;
    ss << shape_.shape[0] << " " << shape_.shape[1] << " " << shape_.shape[2] << " " << shape_.shape[3];
    return ss.str();
}

float* CPUMatrix::GetHostData()
{
    return mat_->data;
}

size_t CPUMatrix::GetRows() const
{
    return mat_->size[0];
}

size_t CPUMatrix::GetCols() const
{
    return mat_->size[1];
}

size_t CPUMatrix::GetNumEls() const
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

void CPUMatrix::CopyTransposeBig(CPUMatrix& m)
{
    copy_transpose(mat_, m.GetMat());
}

void CPUMatrix::CopyTranspose(CPUMatrix& m)
{
    copy_transpose(mat_, m.GetMat());
}

void CPUMatrix::Transpose(const float* i_data, float* o_data, int num_filters, int kernel_width, int kernel_height, int num_colors)
{
  int f, x, y, c;
  long i, target_ind;
  long long size = num_filters * kernel_width * kernel_height * num_colors;
  for (long ind = 0; ind < size; ind++)
  {
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
void CPUMatrix::ConvUp(CPUMatrix& input, CPUMatrix& w, CPUMatrix& output,
                       ConvDesc &conv_desc, float scale_targets)
{
  int num_images = input.shape_.shape[0];
  int inp_width = input.shape_.shape[1];
  int inp_height = input.shape_.shape[2];
  float *images = input.GetHostData();
  float *filters = w.GetHostData();
  float *targets = output.GetHostData();
  int num_colors = conv_desc.num_input_channels;
  int num_filters = conv_desc.num_output_channels;
  int kernel_width = conv_desc.kernel_size_x;
  int kernel_height = conv_desc.kernel_size_y;
  int stride_y = conv_desc.stride_y;
  int stride_x = conv_desc.stride_x;
  int padding_y = conv_desc.padding_y;
  int padding_x = conv_desc.padding_x;
  float scale_outputs = 1.0;

  const int out_height = (inp_height + 2 * padding_y - kernel_height ) / stride_y + 1;
  const int out_width = (inp_width + 2 * padding_x - kernel_width ) / stride_x + 1;

  const int chunk = 16;

#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for (long loc = 0; loc < num_images * out_height * out_width; loc++)
  {
    long i = loc;
    int out_x = i % out_width; i /= out_width;
    int out_y = i % out_height; i /= out_height;
    long image_ind, target_ind, filter_ind;

    for (int f = 0; f < num_filters; f+=chunk)
    {
      // Do the convolution.
      float res[chunk];
      for (int ff = 0; ff < chunk; ff++)
        res[ff] = 0;

      for (int k_y = 0, inp_y = out_y * stride_y -padding_y; k_y < kernel_height && inp_y < inp_height; k_y++, inp_y++)
      {
        if (inp_y < 0)
          continue;
        for (int k_x = 0, inp_x = out_x * stride_x -padding_x; k_x < kernel_width && inp_x < inp_width; k_x++, inp_x++)
        {
          if (inp_x < 0)
            continue;
          image_ind = num_colors * (inp_x + inp_width * (inp_y + inp_height * i));
          for (int c = 0; c < num_colors; c++)
          {
            for (int ff = 0; ff < chunk; ff++)
            {
              filter_ind = c + num_colors * (k_x + kernel_width * (k_y + kernel_height * (f + ff)));
              res[ff] += images[image_ind + c] * filters[filter_ind];
            }
          }
        }
      }

      for (int ff = 0; ff < chunk; ff++)
      {
        target_ind = f + ff + num_filters * loc;
        targets[target_ind] = scale_targets * targets[target_ind] + scale_outputs * res[ff];
      }
    }
  }
}

// images : colors * inp_width * inp_height * numimages
// targets : num_filters * out_width * out_height * numimages
void CPUMatrix::ConvMaxPool(CPUMatrix& input, CPUMatrix& output,
                            ConvDesc &conv_desc, float scale_targets)
{
  int num_images = input.shape_.shape[0];
  int inp_width = input.shape_.shape[1];
  int inp_height = input.shape_.shape[2];
  float *images = input.GetHostData();
  float *targets = output.GetHostData();
  int num_filters = conv_desc.num_output_channels;
  int kernel_width = conv_desc.kernel_size_x;
  int kernel_height = conv_desc.kernel_size_y;
  int stride_y = conv_desc.stride_y;
  int stride_x = conv_desc.stride_x;
  int padding_y = conv_desc.padding_y;
  int padding_x = conv_desc.padding_x;
  float scale_outputs = 1.0;

  const int out_height = (inp_height + 2 * padding_y - kernel_height ) / stride_y + 1;
  const int out_width = (inp_width + 2 * padding_x - kernel_width ) / stride_x + 1;

#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for (long loc = 0; loc < num_images * out_height * out_width; loc++)
  {
    long i = loc;
    int out_x = i % out_width;
    i /= out_width;
    int out_y = i % out_height;
    i /= out_height;
    long image_ind, target_ind;

    for (int f = 0; f < num_filters; f++)
    {
      target_ind = f + num_filters * loc;

      // Do the maxpool.
      float res = -FLT_MAX;
      for (int k_y = 0, inp_y = out_y * stride_y - padding_y + k_y;
           k_y < kernel_height && inp_y < inp_height; k_y++, inp_y++)
      {
        if (inp_y < 0)
          continue;

        for (int k_x = 0, inp_x = out_x * stride_x - padding_x + k_x;
             k_x < kernel_width && inp_x < inp_width; k_x++, inp_x++)
        {
          if (inp_x < 0)
            continue;

          image_ind = f + num_filters * (inp_x + inp_width * (inp_y + inp_height * i));
          if (res < images[image_ind])
            res = images[image_ind];
        }
      }

      targets[target_ind] = scale_targets * targets[target_ind] + scale_outputs * res;
    }
  }
}

// images : num_filters * num_locs
// targets : num_filters * num_locs
void CPUMatrix::ConvResponseNormCrossMap(CPUMatrix& input, CPUMatrix& output, int num_locs,
    int num_filters, int sizeF, float add_scale, float pow_scale, bool blocked, float scale_targets)
{
  float *images = input.GetHostData();
  float *targets = output.GetHostData();
  float scale_outputs = 1.0;

#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for (int i = 0; i < num_locs; i++)
  {
    for (int f = 0; f < num_filters; f++)
    {
      int start = blocked ? ((f / sizeF) * sizeF) : (f - sizeF/2);
      int end = start + sizeF;
      if (start < 0)
        start = 0;
      if (end > num_filters)
        end = num_filters;
      float sum = 0, act;
      for (int j = start; j < end; j++)
      {
        act = images[j + i * num_filters];
        sum += act * act;
      }
      sum = pow(1 + add_scale * sum, -pow_scale);
      targets[f + i * num_filters] = scale_targets * targets[f + i * num_filters] +
                                     scale_outputs * images[f + i * num_filters] * sum;
    }
  }
}

// inputs: num_inputs * num_images
// weights:  num_inputs * num_outputs
// outputs: num_outputs * num_images
void CPUMatrix::FCUp(CPUMatrix& input, CPUMatrix& w, CPUMatrix& output,
    int num_images, int num_outputs, int num_inputs, float scale_targets)
{
  float *inputs = input.GetHostData();
  float *weights = w.GetHostData();
  float *targets = output.GetHostData();
  float scale_outputs = 1.0;

#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_images,
              num_outputs, num_inputs, scale_outputs, inputs, num_inputs,
              weights, num_inputs, scale_targets, targets, num_outputs);
#else
  for (int i = 0; i < num_images; i++)
  {
#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (int j = 0; j < num_outputs; j++)
    {
      float res = 0;
      for (int k = 0; k < num_inputs; k++)
      {
        res += inputs[k + i * num_inputs] * weights[k + j * num_inputs];
      }
      int target_ind = j + i * num_outputs;
      targets[target_ind] = scale_targets * targets[target_ind] + scale_outputs * res;
    }
  }
#endif
}

void CPUMatrix::AddBias(CPUMatrix& input, CPUMatrix& b, CPUMatrix& output, const int num_images, const int num_dims)
{
  float *inputs = input.GetHostData();
  float *bias = b.GetHostData();
  float *outputs = output.GetHostData();

  int length = num_dims * num_images;
#ifdef USE_OPENMP
  #pragma omp parallel for if(length > 10000)
#endif
  for (int i = 0; i < length; i++)
  {
    outputs[i] = inputs[i] + bias[i % num_dims];
  }
}

void CPUMatrix::UpperBound(CPUMatrix& input, CPUMatrix& output, const int length, const float limit)
{
  float *inputs = input.GetHostData();
  float *outputs = output.GetHostData();

#ifdef USE_OPENMP
  #pragma omp parallel for if(length > 10000)
#endif
  for (int i = 0; i < length; i++)
  {
    outputs[i] = inputs[i] > limit ? limit : inputs[i];
  }
}

void CPUMatrix::LowerBound(CPUMatrix& input, CPUMatrix& output, const int length, const float limit)
{
  float *inputs = input.GetHostData();
  float *outputs = output.GetHostData();

#ifdef USE_OPENMP
  #pragma omp parallel for if(length > 10000)
#endif
  for (int i = 0; i < length; i++)
  {
    outputs[i] = inputs[i] < limit ? limit : inputs[i];
  }
}

void CPUMatrix::Argmax(CPUMatrix& input, int* outputs, const int num_images, const int num_dims)
{
  float *inputs = input.GetHostData();

#ifdef USE_OPENMP
  #pragma omp parallel for if(num_images > 1000)
#endif
  for (int i = 0; i < num_images; i++)
  {
    const float *inp = inputs + i * num_dims;
    float max = -FLT_MAX;
    int argmax = -1;
    for (int j = 0; j < num_dims; j++)
    {
      if (max < inp[j])
      {
        max = inp[j];
        argmax = j;
      }
    }
    outputs[i] = argmax;
  }
}

void CPUMatrix::Softmax(CPUMatrix& input, CPUMatrix& output, const int num_images, const int num_dims)
{
  float *inputs = input.GetHostData();
  float *outputs = output.GetHostData();

#ifdef USE_OPENMP
  #pragma omp parallel for if(num_images > 1000)
#endif
  for (int i = 0; i < num_images; i++)
  {
    const float *inp = inputs + i * num_dims;
    float *out = outputs + i * num_dims;
    float max = -FLT_MAX, sum = 0;
    for (int j = 0; j < num_dims; j++)
    {
      if (max < inp[j])
      {
        max = inp[j];
      }
    }
    for (int j = 0; j < num_dims; j++)
    {
      out[j] = exp(inp[j] - max);
      sum += out[j];
    }
    for (int j = 0; j < num_dims; j++)
    {
      out[j] /= sum;
    }
  }
}

void CPUMatrix::Logistic(CPUMatrix& input, CPUMatrix& output, const int length)
{
    float *inputs = input.GetHostData();
    float *outputs = output.GetHostData();

#ifdef USE_OPENMP
    #pragma omp parallel for if(length > 10000)
#endif
    for (int i = 0; i < length; i++)
    {
        outputs[i] = 1 / (1 + exp(-inputs[i]));
    }
}

void CPUMatrix::SoftmaxDistCE(CPUMatrix& state, CPUMatrix& gt, CPUMatrix& output)
{
    int err = compute_cross_entropy(gt.GetMat(), state.GetMat(), output.GetMat(), 1e-10);
    if (err != 0)
    {
        cerr << "SoftmaxDistCE Error : " << GetStringError(err) << endl;
        exit(1);
    }
}

void CPUMatrix::InitRandom(int seed)
{
    int err_code = init_random(&rnde_, seed);
    if (err_code != 0)
    {
        cerr << "Error init random " << GetStringError(err_code) << endl;
        exit(1);
    }
}

