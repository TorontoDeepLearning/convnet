#ifndef CPUMATRIX_H
#define CPUMATRIX_H

#include <hdf5.h>

#include <string>

extern "C" struct eigenmat;

class CPUMatrix {
public:
  CPUMatrix();
  CPUMatrix(const int rows, const int cols);
  ~CPUMatrix();

  void AllocateMemory(const int rows, const int cols);
  void FreeMemory();
  void Set(const float val);
  void Print();
  static void ReadHDF5(hid_t file, float* mat, int size, const std::string& name);
  static void ReadHDF5Shape(hid_t file, const std::string& name, int* rows, int* cols);

  float* GetData();
  int GetRows() const;
  int GetCols() const;
  int GetNumEls() const;

  // GPU computing methods
  void Add(float val);
  void Add(CPUMatrix& m);
  void Add(CPUMatrix& m, float alpha);
  void SquashRelu() { printf("implement me\n"); exit(1); }
  void AddRowVec(CPUMatrix& v);
  void AddRowVec(CPUMatrix& v, float alpha);
  void AddColVec(CPUMatrix& v, float alpha);
  void MultByRowVec(CPUMatrix& v);
  void DivideByColVec(CPUMatrix& v);
  float Sum() { printf("implement me\n"); exit(1); }
  void SumRows(CPUMatrix& target, float alpha, float beta) { printf("implement me\n"); exit(1); }
  void SumCols(CPUMatrix& target, float alpha, float beta) { printf("implement me\n"); exit(1); }
  void Mult(float val);
  void Mult(CPUMatrix& val);
  void Divide(float val);
  void Divide(CPUMatrix& val);
  void Subtract(CPUMatrix& m, CPUMatrix& target);
  void LowerBound(float val);
  void Sqrt();
  void UpperBoundMod(float val) { printf("implement me\n"); exit(1); }
  void SqSumAxis(CPUMatrix& target, int axis, float beta, float alpha) { printf("implement me\n"); exit(1); }
  void NormLimitByAxis(int axis, float val, bool constraint) { printf("implement me\n"); exit(1); }
  void Dropout(float dropprob, float fill_value, float scale_factor) { printf("implement me\n"); exit(1); }
  void ApplyDerivativeOfReLU(CPUMatrix& state);
  void ApplySoftmax() { printf("implement me\n"); exit(1); }
  void ApplyLogistic();
  void ApplyDerivativeOfLogistic(CPUMatrix& state);
  float EuclidNorm();
  float VDot(CPUMatrix& m);
  void CopyTransposeBig(CPUMatrix& m) { printf("implement me\n"); exit(1); }
  void CopyTranspose(CPUMatrix& m);

  static void Dot(CPUMatrix& a, CPUMatrix& b, CPUMatrix& c, float alpha, float beta);
  static void Dot(CPUMatrix& a, CPUMatrix& b, CPUMatrix& c, float alpha, float beta,
                  bool transpose_a, bool transpose_b);

  static void Transpose(const float* i_data, float* o_data, int num_filters, int kernel_width, int kernel_height, int num_colors);

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

protected:
  eigenmat* GetMat() { return mat_; }
  eigenmat* GetMatTranspose() { return mat_t_; }

private:
  eigenmat *mat_;
  eigenmat *mat_t_;
};

#endif

