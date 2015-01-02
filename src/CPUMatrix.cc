#include "CPUMatrix.h"
#include "eigenmat.h"
#include "cpumat_conv.h"
#include "util.h"

#include <iostream>
#include <sstream>
#include <cfloat>
#include <chrono>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

using namespace std;

rnd_struct_e rnde_;
Matrix Matrix::temp_;
Matrix Matrix::ones_;
Matrix Matrix::rgb_to_yuv_mat_;

Matrix::Matrix()
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

Matrix::Matrix(const size_t rows, const size_t cols, const bool on_gpu)
{
    Matrix();

    AllocateMainMemory(rows, cols);
}

Matrix::~Matrix()
{
    if (mat_->owns_data == 1)
    {
        FreeMemory();
    }

    delete mat_;
    delete mat_t_;
}

void Matrix::Tie(Matrix &m)
{
    cout << "Tying" << endl;
    *mat_ = *(m.GetMat());
    *mat_t_ = *(m.GetMatTranspose());
}

void Matrix::SetupTranspose()
{
    *mat_t_ = *mat_;
    mat_t_->is_trans = 1 - mat_->is_trans;
}

void Matrix::SetShape4D(int d1, int d2, int d3, int d4)
{
    shape_.shape[0] = d1;
    shape_.shape[1] = d2;
    shape_.shape[2] = d3;
    shape_.shape[3] = d4;
}

void Matrix::SetShape4D_like(Matrix& mat)
{
    Shape4D &s = mat.GetShape4D();
    SetShape4D(s.shape[0], s.shape[1], s.shape[2], s.shape[3]);
}

Shape4D& Matrix::GetShape4D()
{
    return shape_;
}

void Matrix::AllocateGPUMemory(const size_t rows, const size_t cols, const std::string& name)
{
    if (rows != mat_->size[0] || cols != mat_->size[1])
    {
        AllocateMainMemory(rows, cols);
        SetupTranspose();
    }
}

void Matrix::AllocateGPUMemory(const size_t rows, const size_t cols)
{
    AllocateGPUMemory(rows, cols, "");
}

void Matrix::AllocateMainMemory(const size_t rows, const size_t cols)
{
    FreeMemory();

    mat_->size[0] = rows;
    mat_->size[1] = cols;
    mat_->is_trans = 0;
    mat_->owns_data = 1;
    mat_->data = new float[rows * cols];
}

void Matrix::FreeMemory()
{
    if (mat_->size[0] * mat_->size[1] > 0)
    {
        delete[] mat_->data;
        mat_->size[0] = 0;
        mat_->size[1] = 0;
    }
}

void Matrix::Set(const float val)
{
    int err_code = assign_scalar(mat_, val);
    if (err_code != 0)
    {
        cerr << "Error: Could not set to scalar : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void Matrix::Set(Matrix& val)
{
    int err_code = copy_on_device(val.GetMat(), mat_); // source, dest.
    if (err_code != 0)
    {
        cerr << "Error: Could not set to val : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void Matrix::WriteValue(int index, float val)
{
    WriteValue(index % mat_->size[0], index / mat_->size[0], val);
}

float Matrix::ReadValue(int index)
{
    return ReadValue(index % mat_->size[0], index / mat_->size[0]);
}

void Matrix::WriteValue(int row, int col, float val)
{
    int err_code = write_at(mat_, row, col, val);
    if (err_code != 0)
    {
        cerr << "Error: Could not write value : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

float Matrix::ReadValue(int row, int col)
{
    int err_code;
    float res = read_from(mat_, row, col, &err_code);
    if (err_code != 0)
    {
        cerr << "Error: Could not read value : " << GetStringError(err_code) << endl;
        exit(1);
    }
    return res;
}

void Matrix::CopyFromMainMemory(Matrix& mat)
{
    int err_code = copy_on_device(mat.GetMat(), mat_); // source, dest.
    if (err_code != 0)
    {
        cerr << "Error: Could not set to val : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void Matrix::CopyP2PAsync(Matrix& val)
{
    int err_code = copy_on_device(val.GetMat(), mat_); // source, dest.
    if (err_code != 0)
    {
        cerr << "Error: Could not set to val : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void Matrix::GetSlice(Matrix& slice, size_t start, size_t end)
{
    get_slice(mat_, slice.GetMat(), start, end);
    slice.SetupTranspose();
}

void Matrix::FillWithRand()
{
    int err_code = fill_with_rand(&rnde_, mat_);
    if (err_code != 0)
    {
        cerr << "Error: Could not fill with rand : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void Matrix::FillWithRandn()
{
    int err_code = fill_with_randn(&rnde_, mat_);
    if (err_code != 0)
    {
        cerr << "Error: Could not fill with randn : " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void Matrix::Reshape(const size_t rows, const size_t cols)
{
    reshape(mat_, rows, cols);
    *mat_t_ = *mat_;
    mat_t_->is_trans = 1;
}

bool Matrix::CheckNaN()
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

void Matrix::WriteHDF5(hid_t file, const string& name)
{
    // cols, rows swapped because cudamat is col major, but hdf5 is row major.
    WriteHDF5CPU(file, mat_->data, mat_->size[1], mat_->size[0], name);
}

void Matrix::ReadHDF5(hid_t file, const string& name)
{
    ReadHDF5CPU(file, mat_->data, mat_->size[0] * mat_->size[1], name);
}

void Matrix::AllocateAndReadHDF5(hid_t file, const string& name)
{
    int rows, cols;
    ReadHDF5Shape(file, name, &rows, &cols);
    AllocateGPUMemory(rows, cols);
    ReadHDF5(file, name);
}

void Matrix::Print()
{
    int rows = mat_->size[0];
    int cols = mat_->size[1];
    for (int i=0; i<rows && i<10; i++)
    {
        for (int j=0; j<cols && j<10; j++)
        {
            cout << mat_->data[j*rows + i] << " ";
        }
        if (!(i == 9 || i == rows - 1))
        {
            cout << endl;
        }
    }
    float max = -FLT_MAX;
    for (int j=0; j<rows*cols; j++)
    {
        if (max < mat_->data[j])
        {
            max = mat_->data[j];
        }
    }
    cout << "... Max " << max << endl;
}

string Matrix::GetShapeString()
{
    stringstream ss;
    ss << mat_->size[0] << " " << mat_->size[1];
    return ss.str();
}

string Matrix::GetShape4DString()
{
    stringstream ss;
    ss << shape_.shape[0] << " " << shape_.shape[1] << " " << shape_.shape[2] << " " << shape_.shape[3];
    return ss.str();
}

float* Matrix::GetHostData()
{
    return mat_->data;
}

size_t Matrix::GetRows() const
{
    return mat_->size[0];
}

size_t Matrix::GetCols() const
{
    return mat_->size[1];
}

size_t Matrix::GetNumEls() const
{
    return mat_->size[1] * mat_->size[0];
}

void Matrix::Add(float val)
{
    add_scalar(mat_, val, mat_);
}

void Matrix::Add(Matrix& m)
{
    add_elementwise(mat_, m.GetMat(), mat_);
}

void Matrix::Add(Matrix& m, float alpha)
{
    add_mult(mat_, m.GetMat(), alpha);
}

void Matrix::AddRowVec(Matrix& v)
{
    add_row_vec(mat_, v.GetMat(), mat_);
}

void Matrix::AddRowVec(Matrix& v, float alpha)
{
    add_row_mult(mat_, v.GetMat(), mat_, alpha);
}

void Matrix::AddColVec(Matrix& v, float alpha)
{
    add_col_mult(mat_, v.GetMat(), mat_, alpha);
}

void Matrix::MultByRowVec(Matrix& val)
{
    mult_by_row_vec(mat_, val.GetMat(), mat_);
}

void Matrix::DivideByColVec(Matrix& v)
{
    div_by_col_vec(mat_, v.GetMat(), mat_);
}

float Matrix::Sum()
{
    return sum_all(mat_);
}

// target = alpha * target + beta * sum_rows(self)
void Matrix::SumRows(Matrix& target, float alpha, float beta)
{
    sum_by_axis(mat_, target.GetMat(), 0, beta, alpha);
}

// target = alpha * target + beta * sum_cols(self)
void Matrix::SumCols(Matrix& target, float alpha, float beta)
{
    sum_by_axis(mat_, target.GetMat(), 1, beta, alpha);
}

// target = alpha * target + beta * sum_cols(self**2)
void Matrix::SqSumAxis(Matrix& target, int axis, float beta, float alpha)
{
    int err_code = sqsum_by_axis(mat_, target.GetMat(), axis, beta, alpha);
    if (err_code != 0)
    {
        cerr << "Error in sqsum_by_axis " << GetStringError(err_code) << endl;
        exit(1);
    }
}

// self *= val
void Matrix::Mult(float val)
{
    mult_by_scalar(mat_, val, mat_);
}

void Matrix::Mult(Matrix& val)
{
    mult_elementwise(mat_, val.GetMat(), mat_);
}

void Matrix::Divide(float val)
{
    divide_by_scalar(mat_, val, mat_);
}

void Matrix::Divide(Matrix& val)
{
    divide_elementwise(mat_, val.GetMat(), mat_);
}

void Matrix::Subtract(Matrix& m, Matrix& target)
{
    int err_code = subtract_elementwise(mat_, m.GetMat(), target.GetMat());
    if (err_code != 0)
    {
        cerr << "Error in subtract." << endl;
        exit(1);
    }
}

void Matrix::UpperBoundMod(float val)
{
    upper_bound_mod_scalar(mat_, val, mat_);
}

void Matrix::LowerBound(float val)
{
    lower_bound_scalar(mat_, val, mat_);
}

void Matrix::Sqrt()
{
    apply_sqrt(mat_, mat_);
}

void Matrix::Dropout(float dropprob, float fill_value, float scale_factor)
{
    dropout(&rnde_, mat_, dropprob, fill_value, scale_factor);
}

// c = alpha * c + beta * a * b
void Matrix::Dot(Matrix& a, Matrix& b, Matrix& c, float alpha, float beta)
{
    dot(a.GetMat(), b.GetMat(), c.GetMat(), alpha, beta);
}

// c = alpha * c + beta * T(a) * T(b)
void Matrix::Dot(Matrix& a, Matrix& b, Matrix& c, float alpha, float beta,
                 bool transpose_a, bool transpose_b)
{
    eigenmat* a_mat = transpose_a ? a.GetMatTranspose() : a.GetMat();
    eigenmat* b_mat = transpose_b ? b.GetMatTranspose() : b.GetMat();
    dot(a_mat, b_mat, c.GetMat(), alpha, beta);
}

void Matrix::ApplyDerivativeOfReLU(Matrix& state)
{
    apply_rectified_linear_deriv(mat_, state.GetMat(), mat_);
}

void Matrix::ApplySoftmax()
{
    softmax_row_major(mat_);
}

void Matrix::ApplyLogistic()
{
    apply_sigmoid(mat_, mat_);
}

void Matrix::ApplyDerivativeOfLogistic(Matrix& state)
{
    apply_logistic_deriv(mat_, state.GetMat(), mat_);
}

void Matrix::LogisticCEDeriv(Matrix& state, Matrix& gt, Matrix& deriv)
{
    apply_logistic_grad(state.GetMat(), gt.GetMat(), deriv.GetMat());
}

void Matrix::LogisticCorrect(Matrix& state, Matrix& gt, Matrix& output)
{
    get_logistic_correct_normalized(state.GetMat(), gt.GetMat(), output.GetMat());
}

void Matrix::SoftmaxCEDeriv(Matrix& state, Matrix& gt, Matrix& deriv)
{
    apply_softmax_grad_row_major(state.GetMat(), gt.GetMat(), deriv.GetMat());
}

void Matrix::SoftmaxCorrect(Matrix& state, Matrix& gt, Matrix& output)
{
    get_softmax_correct_row_major(state.GetMat(), gt.GetMat(), output.GetMat());
}

void Matrix::SoftmaxCE(Matrix& state, Matrix& gt, Matrix& output)
{
    get_softmax_cross_entropy_row_major(state.GetMat(), gt.GetMat(), output.GetMat(), 1e-10);
}

void Matrix::SoftmaxDistCE(Matrix& state, Matrix& gt, Matrix& output)
{
    int err = compute_cross_entropy(gt.GetMat(), state.GetMat(), output.GetMat(), 1e-10);
    if (err != 0)
    {
        cerr << "SoftmaxDistCE Error : " << GetStringError(err) << endl;
        exit(1);
    }
}

void Matrix::HingeLossDeriv(Matrix& state, Matrix& gt, Matrix& deriv, bool quadratic, float margin)
{
    int err_code = hinge_loss_row_major(state.GetMat(), gt.GetMat(), deriv.GetMat(), quadratic, margin);
    if (err_code != 0)
    {
        cerr << "Error in hinge loss " << GetStringError(err_code) << endl;
        exit(1);
    }
}

float Matrix::EuclidNorm()
{
    return euclid_norm(mat_); // TODO: return error code?
}

float Matrix::VDot(Matrix& m)
{
    int err;
    return vdot(mat_, m.GetMat(), &err);
}

void Matrix::CopyTransposeBig(Matrix& m)
{
    copy_transpose(mat_, m.GetMat());
}

void Matrix::CopyTranspose(Matrix& m)
{
    copy_transpose(mat_, m.GetMat());
}

// images : colors * inp_width * inp_height * numimages
// targets : num_filters * out_width * out_height * numimages
void Matrix::ConvMaxPool(Matrix& input, Matrix& output, ConvDesc &conv_desc)
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
    int padding_y = -conv_desc.padding_y;
    int padding_x = -conv_desc.padding_x;
    float scale_outputs = 1.0;
    float scale_targets = 0.0;

    int out_height = (inp_height + 2 * padding_y - kernel_height ) / stride_y + 1;
    int out_width = (inp_width + 2 * padding_x - kernel_width ) / stride_x + 1;
    int inp_size = inp_height*inp_width;
    int out_size = out_height*out_width;

    #pragma omp parallel for
    for (int i=0; i<num_images; i++)
    {
        for (int out_x=0; out_x<out_width; out_x++)
        {
            for (int out_y=0; out_y<out_height; out_y++)
            {
                for (int f=0; f<num_filters; f++)
                {
                    long target_ind = i + num_images*(out_size*f + out_width*out_y + out_x);

                    // Do the maxpool.
                    float res = -FLT_MAX;
                    for (int k_y=0, inp_y=out_y*stride_y-padding_y+k_y;
                         k_y<kernel_height && inp_y<inp_height; k_y++, inp_y++)
                    {
                        if (inp_y < 0)
                        {
                            continue;
                        }

                        for (int k_x=0, inp_x=out_x*stride_x-padding_x+k_x;
                             k_x<kernel_width && inp_x<inp_width; k_x++, inp_x++)
                        {
                            if (inp_x < 0)
                            {
                                continue;
                            }

                            long image_ind = i + num_images*(inp_size*f + inp_width*inp_y + inp_x);
                            if (res < images[image_ind])
                            {
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

void Matrix::ConvMaxPoolUndo(Matrix& input, Matrix& deriv_output, Matrix& output,
                             Matrix& deriv_input, ConvDesc &conv_desc, float scale_targets)
{
    int num_images = deriv_output.shape_.shape[0];
    int inp_width = deriv_output.shape_.shape[1];
    int inp_height = deriv_output.shape_.shape[2];
    int numFilters = deriv_output.shape_.shape[3];
    int out_width = deriv_input.shape_.shape[1];
    int out_height = deriv_input.shape_.shape[2];
    int padding_x = conv_desc.padding_x;
    int kernel_size_x = conv_desc.kernel_size_x;
    int stride_x = conv_desc.stride_x;
    float* imgs = input.GetHostData();
    float* maxGrads = deriv_output.GetHostData();
    float* maxActs = output.GetHostData();
    float* target = deriv_input.GetHostData();
    float scaleOutputs = 1;
    int inp_size = inp_height * inp_width;
    int out_size = out_height * out_width;

    #pragma omp parallel for
    for (int i=0; i<num_images; i++)
    {
        for (int out_x=0; out_x<out_width; out_x++)
        {
            for (int out_y=0; out_y<out_height; out_y++)
            {
                for (int f=0; f<numFilters; f++)
                {
                    int startOutputY = out_y - padding_x < kernel_size_x ? 0 : (out_y - padding_x - kernel_size_x) / stride_x + 1;
                    int endOutputY = MIN(inp_height, 1 + (out_y - padding_x) / stride_x);

                    int startOutputX = out_x - padding_x < kernel_size_x ? 0 : (out_x - padding_x - kernel_size_x) / stride_x + 1;
                    int endOutputX = MIN(inp_width, 1 + (out_x - padding_x) / stride_x);

                    long target_ind = i + num_images*(out_size*f + out_width*out_y + out_x);
                    float res = 0;
                    if (out_x<padding_x + stride_x * (inp_width-1) + kernel_size_x &&
                        out_y<padding_x + stride_x * (inp_height-1) + kernel_size_x)
                    {
                        for (int inp_y=startOutputY; inp_y<endOutputY; inp_y++)
                        {
                            for (int inp_x=startOutputX; inp_x<endOutputX; inp_x++)
                            {
                                long image_ind = i + num_images*(inp_size*f + inp_width*inp_y + inp_x);
                                if (imgs[target_ind] == maxActs[image_ind])
                                {
                                    res += maxGrads[image_ind];
                                }
                            }
                        }
                    }

                    target[target_ind] = scale_targets * target[target_ind] + scaleOutputs * res;
                }
            }
        }
    }
}

// images : colors * inp_width * inp_height * numimages
// targets : num_filters * out_width * out_height * numimages
void Matrix::ConvAvgPool(Matrix& input, Matrix& output, ConvDesc &conv_desc)
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
    int padding_y = -conv_desc.padding_y;
    int padding_x = -conv_desc.padding_x;
    float scale_outputs = 1.0;
    float scale_targets = 0.0;

    int out_height = (inp_height + 2 * padding_y - kernel_height ) / stride_y + 1;
    int out_width = (inp_width + 2 * padding_x - kernel_width ) / stride_x + 1;
    int inp_size = inp_height*inp_width;
    int out_size = out_height*out_width;

    #pragma omp parallel for
    for (int i=0; i<num_images; i++)
    {
        for (int out_x=0; out_x<out_width; out_x++)
        {
            for (int out_y=0; out_y<out_height; out_y++)
            {
                for (int f=0; f<num_filters; f++)
                {
                    long target_ind = i + num_images*(out_size*f + out_width*out_y + out_x);

                    // Do the avgpool.
                    float res = 0;
                    int num_elems = 0;
                    for (int k_y=0, inp_y=out_y*stride_y-padding_y+k_y;
                         k_y<kernel_height && inp_y<inp_height; k_y++, inp_y++)
                    {
                        if (inp_y < 0)
                        {
                            continue;
                        }

                        for (int k_x=0, inp_x=out_x*stride_x-padding_x+k_x;
                             k_x<kernel_width && inp_x<inp_width; k_x++, inp_x++)
                        {
                            if (inp_x < 0)
                            {
                                continue;
                            }

                            num_elems++;
                            long image_ind = i + num_images*(inp_size*f + inp_width*inp_y + inp_x);
                            res += images[image_ind];
                        }
                    }
                    res /= num_elems;

                    targets[target_ind] = scale_targets * targets[target_ind] + scale_outputs * res;
                }
            }
        }
    }
}

void Matrix::ConvAvgPoolUndo(Matrix& input, Matrix& deriv_output, ConvDesc &conv_desc, float scale_targets, float scaleOutputs)
{
    int num_images = input.shape_.shape[0];
    int inp_width = input.shape_.shape[1];
    int inp_height = inp_width;
    int numFilters = input.shape_.shape[3];
    int out_width = deriv_output.shape_.shape[1];
    int out_height = out_width;
    int padding_x = conv_desc.padding_x;
    int kernel_size_x = conv_desc.kernel_size_x;
    int stride_x = conv_desc.stride_x;
    float* avgGrads = input.GetHostData();
    float* target = deriv_output.GetHostData();
    int inp_size = inp_height * inp_width;
    int out_size = out_height * out_width;

    #pragma omp parallel for
    for (int i=0; i<num_images; i++)
    {
        for (int out_x=0; out_x<out_width; out_x++)
        {
            for (int out_y=0; out_y<out_height; out_y++)
            {
                for (int f=0; f<numFilters; f++)
                {
                    int startOutputY = out_y - padding_x < kernel_size_x ? 0 : (out_y - padding_x - kernel_size_x) / stride_x + 1;
                    int endOutputY = MIN(inp_height, 1 + (out_y - padding_x) / stride_x);

                    int startOutputX = out_x - padding_x < kernel_size_x ? 0 : (out_x - padding_x - kernel_size_x) / stride_x + 1;
                    int endOutputX = MIN(inp_width, 1 + (out_x - padding_x) / stride_x);

                    float res = 0;
                    if (out_x<padding_x + stride_x * (inp_width-1) + kernel_size_x &&
                        out_y<padding_x + stride_x * (inp_height-1) + kernel_size_x)
                    {
                        for (int inp_y=startOutputY; inp_y<endOutputY; inp_y++)
                        {
                            float regionStartY = fmaxf(0, padding_x + inp_y * stride_x);
                            float regionEndY = fminf(out_height, padding_x + inp_y * stride_x + kernel_size_x);
                            float regionSizeY = regionEndY - regionStartY;
                            for (int inp_x=startOutputX; inp_x<endOutputX; inp_x++)
                            {
                                float regionStartX = fmaxf(0, padding_x + inp_x * stride_x);
                                float regionEndX = fminf(out_width, padding_x + inp_x * stride_x + kernel_size_x);
                                float regionSizeX = regionEndX - regionStartX;
                                float regionSizeInv = 1.0f / (regionSizeX * regionSizeY);

                                long image_ind = i + num_images*(inp_size*f + inp_width*inp_y + inp_x);
                                res += avgGrads[image_ind] * regionSizeInv;
                            }
                        }
                    }

                    long target_ind = i + num_images*(out_size*f + out_width*out_y + out_x);
                    target[target_ind] = scale_targets * target[target_ind] + scaleOutputs * res;
                }
            }
        }
    }
}

void Matrix::ConvUp(Matrix& input, Matrix& w, Matrix& output,
                    ConvDesc &conv_desc, float scale_targets)
{
    convUp(input.GetMat(), w.GetMat(), output.GetMat(),
           input.GetShape4D(), w.GetShape4D(), output.GetShape4D(),
           conv_desc, scale_targets, 1.0, true);
}

void Matrix::ConvDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                      ConvDesc &conv_desc, float scale_targets)
{
    convDown(deriv_output.GetMat(), w.GetMat(), deriv_input.GetMat(),
             deriv_output.GetShape4D(), w.GetShape4D(), deriv_input.GetShape4D(),
             conv_desc, scale_targets, 1.0, true);
}

void Matrix::ConvOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                      ConvDesc &conv_desc, int partial_sum_y, int partial_sum_x,
                      float scale_targets, float scale_outputs)
{
    convOutp(input.GetMat(), deriv_output.GetMat(), dw.GetMat(),
             input.GetShape4D(), deriv_output.GetShape4D(), dw.GetShape4D(),
             conv_desc, scale_targets, scale_outputs, true);
}

void Matrix::ConvResponseNormCrossMap(Matrix& input, Matrix& output, int numFilters,
                                      int sizeF, float addScale, float powScale, bool blocked)
{
    ResponseNormCrossMap(input.GetMat(), output.GetMat(),
                         numFilters, sizeF, addScale, powScale, blocked);
}

void Matrix::ConvResponseNormCrossMapUndo(Matrix& outGrads, Matrix& inputs,
                                          Matrix& acts, Matrix& targets, int numFilters,
                                          int sizeF, float addScale, float powScale, bool blocked)
{
    ResponseNormCrossMapUndo(outGrads.GetMat(), inputs.GetMat(), targets.GetMat(),
                             numFilters, sizeF, addScale, powScale, blocked);
}

void Matrix::ConvUpSample(Matrix& input, Matrix& output, int factor, float scaleTargets)
{
    ConvDesc conv_desc;
    conv_desc.kernel_size_y = factor;
    conv_desc.kernel_size_x = factor;
    conv_desc.stride_y = factor;
    conv_desc.stride_x = factor;
    conv_desc.padding_y = 0;
    conv_desc.padding_x = 0;
    conv_desc.num_input_channels = input.GetShape4D().shape[3];
    conv_desc.num_output_channels = output.GetShape4D().shape[3];
    conv_desc.num_groups = 1;

    ConvAvgPoolUndo(input, output, conv_desc, scaleTargets, factor*factor);
}

void Matrix::ConvDownSample(Matrix& input, Matrix& output, int factor)
{
    ConvDesc conv_desc;
    conv_desc.kernel_size_y = factor;
    conv_desc.kernel_size_x = factor;
    conv_desc.stride_y = factor;
    conv_desc.stride_x = factor;
    conv_desc.padding_y = 0;
    conv_desc.padding_x = 0;
    conv_desc.num_input_channels = input.GetShape4D().shape[3];
    conv_desc.num_output_channels = output.GetShape4D().shape[3];
    conv_desc.num_groups = 1;

    ConvAvgPool(input, output, conv_desc);
}

void Matrix::ConvRGBToYUV(Matrix& input, Matrix& output)
{
    if (Matrix::rgb_to_yuv_mat_.GetNumEls() == 0)
    {
        Matrix::rgb_to_yuv_mat_.AllocateGPUMemory(3, 3);
        float* data = Matrix::rgb_to_yuv_mat_.GetHostData();
        data[0] = 0.2126f;   data[1] = 0.7152f;   data[2] = 0.0722f;
        data[3] = -0.09991f; data[4] = -0.33609f; data[5] = 0.436f;
        data[6] = 0.615f;    data[7] = -0.55861f; data[8] = -0.05639f;
    }

    int batch_size = input.GetRows();
    input.Reshape(-1, 3);
    output.Reshape(-1, 3);
    dot(input.GetMat(), Matrix::rgb_to_yuv_mat_.GetMat(), output.GetMat(), 1, 0);
    output.Reshape(batch_size, -1);
    input.Reshape(batch_size, -1);
}

void Matrix::ExtractPatches(Matrix& source, Matrix& dest, Matrix& width_offset,
                               Matrix& height_offset, Matrix& flip_bit,
                               int image_size_y, int image_size_x, int patch_size_y, int patch_size_x)
{
    int err_code = extract_patches(source.GetMat(), dest.GetMat(),
                                   width_offset.GetMat(), height_offset.GetMat(),
                                   flip_bit.GetMat(), image_size_x, image_size_y,
                                   patch_size_x, patch_size_y);
    if (err_code != 0)
    {
        cerr << "Error extracting patches " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void Matrix::NormLimitByAxis(int axis, float val, bool constraint)
{
    normlimit_by_axis(mat_, mat_, axis, val, constraint);
}

void Matrix::NormalizeColumnwise()
{
    normalize_by_axis(mat_, mat_, 0);
}

void Matrix::ShuffleColumns(Matrix& rand_perm_indices)
{
    shuffleColumns(mat_, rand_perm_indices.GetMat());
}

void Matrix::AddToEachPixel(Matrix& v, float mult)
{
    add_to_each_pixel(mat_, v.GetMat(), mat_, mult);
}

void Matrix::RectifyBBox(Matrix& width_offset, Matrix& height_offset, Matrix& flip, int patch_width, int patch_height)
{
    int err_code = rectify_bounding_boxes(mat_, width_offset.GetMat(), height_offset.GetMat(), flip.GetMat(), patch_width, patch_height);

    if (err_code != 0)
    {
        cerr << "Error rectifying boxes " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void Matrix::GetOnes(size_t rows, size_t cols, Matrix& ones)
{
    Matrix& o = Matrix::ones_;
    size_t size = o.GetNumEls();
    size_t length = rows * cols;
    if (length > size)
    {
        cout << "Allocating new one memory of size " << length << endl;
        o.AllocateGPUMemory(1, length, "ones");
        o.Set(1);
    }

    o.GetSlice(ones, 0, length);
    reshape(ones.GetMat(), rows, cols);
}

void Matrix::GetTemp(size_t rows, size_t cols, Matrix& temp)
{
    Matrix& t = Matrix::temp_;
    size_t size = t.GetNumEls();
    size_t length = rows * cols;
    if (length > size)
    {
        cout << "Allocating new temp memory of size " << length << endl;
        t.AllocateGPUMemory(1, length, "temp");
    }

    t.GetSlice(temp, 0, length);
    reshape(temp.GetMat(), rows, cols);
}

void Matrix::SquashRelu()
{
    apply_relu_squash(mat_, mat_, 2);
}

void Matrix::InitRandom(int seed)
{
    int err_code = init_random(&rnde_, seed);
    if (err_code != 0)
    {
        cerr << "Error init random " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void Matrix::AdagradUpdate(Matrix& adagrad_history, Matrix& gradient, float delta)
{
    int err_code = adagrad(adagrad_history.GetMat(), gradient.GetMat(), delta);
    if (err_code != 0)
    {
        cerr << "Error in adagrad update " << GetStringError(err_code) << endl;
        exit(1);
    }
}

void Matrix::RMSPropUpdate(Matrix& rms_history, Matrix& gradient, float factor)
{
    int err_code = rms_prop(rms_history.GetMat(), gradient.GetMat(), factor);
    if (err_code != 0)
    {
        cerr << "Error in rms update " << GetStringError(err_code) << endl;
        exit(1);
    }
}

