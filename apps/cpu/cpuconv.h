#ifndef CPUCONV_H_
#define CPUCONV_H_
#include "hdf5.h"
//#include <smmintrin.h>
#include <string>
using namespace std;
class CPUMatrix {
 public:
  CPUMatrix();
  CPUMatrix(const int rows, const int cols);
  void AllocateMemory(const int rows, const int cols);
  int GetSize() { return rows_ * cols_; }
  void FreeMemory();
  float* GetData() { return data_;}
  void Print();
  void Print(int rows, int cols);


  static void Transpose(const float* i_data, float* o_data, int num_filters, int kernel_width, int kernel_height, int num_colors);
  static void SetZero(float* data, int length);
  static void ConvUp(
      const float* images, const float* filters, float* targets,
      const int num_images, const int num_colors, const int num_filters,
      const int inp_width, const int inp_height,
      const int kernel_width, const int kernel_height,
      const int stride_x, const int stride_y,
      const int padding_x, const int padding_y,
      const float scale_outputs,
      const float scale_targets);

  static void MaxPool(
      const float* images, float* targets,
      const int num_images, const int num_filters,
      const int inp_width, const int inp_height,
      const int kernel_width, const int kernel_height,
      const int stride_x, const int stride_y,
      const int padding_x, const int padding_y,
      const float scale_outputs,
      const float scale_targets);

  static void ResponseNormCrossMap(
    const float* images, float* targets,
    const int num_locs, const int num_filters,
    const int sizeF, const bool blocked,
    const float add_scale, const float pow_scale,
    const float scale_outputs,
    const float scale_targets);

  static void FCUp(
    const float* inputs, const float* weights, float* targets,
    const int num_images, const int num_outputs, const int num_inputs,
    const float scale_outputs, const float scale_targets);

  static void UpperBound(const float* inputs, float* outputs, const int length, const float limit);
  static void LowerBound(const float* inputs, float* outputs, const int length, const float limit);
  static void AddBias(const float* inputs, const float* bias, float* outputs, const int num_images, const int num_dims);
  static void Softmax(const float* inputs, float* outputs, const int num_images, const int num_dims);
  static void Logistic(const float* inputs, float* outputs, const int length);
  static void Argmax(const float* inputs, int* outputs, const int num_images, const int num_dims);
  static void ReadHDF5(hid_t file, float* mat, int size, const string& name);
  static void ReadHDF5Shape(hid_t file, const string& name, int* rows, int* cols);

 private:
  float* data_;
  int rows_, cols_;
};
#endif
