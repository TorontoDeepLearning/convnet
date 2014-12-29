#ifndef CPUMATRIX_H
#define CPUMATRIX_H

#include <hdf5.h>

#include <vector>
#include <string>

extern "C" struct eigenmat;

typedef struct Shape4D
{
  int shape[4];
} Shape4D;

typedef struct ConvDesc
{
  int num_input_channels;
  int num_output_channels;
  int kernel_size_y;
  int kernel_size_x;
  int kernel_size_t;
  int stride_y;
  int stride_x;
  int stride_t;
  int padding_y;
  int padding_x;
  int padding_t;

  int input_channel_begin;
  int input_channel_end;
  int output_channel_begin;
  int output_channel_end;
  int num_groups;
} ConvDesc;

inline void notImpl()
{
    printf("implement me\n");
    exit(1);
}

// A CPU matrix class
class Matrix
{
public:
  Matrix();
  Matrix(const size_t rows, const size_t cols, const bool on_gpu);
  ~Matrix();

  void Tie(Matrix &m);
  void SetupTranspose();
  void SetShape4D(int d1, int d2, int d3, int d4);
  void SetShape4D_like(Matrix& mat);
  Shape4D& GetShape4D();
  void AllocateGPUMemory(const size_t rows, const size_t cols, const std::string& name);
  void AllocateGPUMemory(const size_t rows, const size_t cols);
  void AllocateMainMemory(const size_t rows, const size_t cols);
  void Set(const float val);
  void Set(Matrix& val);
  float ReadValue(int row, int col);
  float ReadValue(int index);
  void WriteValue(int row, int col, float val);
  void WriteValue(int index, float val);
  void CopyP2PAsync(Matrix& val);
  void GetSlice(Matrix& slice, size_t start, size_t end);
  void FillWithRand();
  void FillWithRandn();
  void CopyToHost() { /*Do nothing*/ }
  void CopyToDevice() { /*Do nothing*/ }
  void CopyToDeviceSlice(const size_t start, const size_t end) { /*Do nothing*/ }
  void CopyToHostSlice(const size_t start, const size_t end) { /*Do nothing*/ }
  void CopyFromMainMemory(Matrix& mat);
  void Reshape(const size_t rows, const size_t cols);
  void Print();
  bool CheckNaN();
  void WriteHDF5(hid_t file, const std::string& name);
  void ReadHDF5(hid_t file, const std::string& name);
  void AllocateAndReadHDF5(hid_t file, const std::string& name);
  std::string GetShapeString();
  std::string GetShape4DString();

  float* GetHostData();
  size_t GetRows() const;
  size_t GetCols() const;
  size_t GetNumEls() const;

  int GetGPUId() const { return 0; }
  void SetGPUId(int gpu_id) { /*Do nothing*/ }
  void SetReady() { /*Do nothing*/ }
  void WaitTillReady() { /*Do nothing*/ }

  // computing methods
  void Add(float val);
  void Add(Matrix& m);
  void Add(Matrix& m, float alpha);
  void SquashRelu() { notImpl(); }
  void AddRowVec(Matrix& v);
  void AddRowVec(Matrix& v, float alpha);
  void AddColVec(Matrix& v, float alpha);
  void MultByRowVec(Matrix& v);
  void DivideByColVec(Matrix& v);
  float Sum();
  void SumRows(Matrix& target, float alpha, float beta);
  void SumCols(Matrix& target, float alpha, float beta);
  void Mult(float val);
  void Mult(Matrix& val);
  void Divide(float val);
  void Divide(Matrix& val);
  void Subtract(Matrix& m, Matrix& target);
  void LowerBound(float val);
  void Sqrt();
  void UpperBoundMod(float val);
  void SqSumAxis(Matrix& target, int axis, float beta, float alpha);
  void NormLimitByAxis(int axis, float val, bool constraint) { notImpl(); }
  void NormalizeColumnwise() { notImpl(); }
  void Dropout(float dropprob, float fill_value, float scale_factor);
  void ApplyDerivativeOfReLU(Matrix& state);
  void ApplySoftmax();
  void ApplyLogistic();
  void ApplyDerivativeOfLogistic(Matrix& state);
  float EuclidNorm();
  float VDot(Matrix& m);
  void CopyTransposeBig(Matrix& m);
  void CopyTranspose(Matrix& m);
  void ShuffleColumns(Matrix& rand_perm_indices) { notImpl(); }
  void AddToEachPixel(Matrix& v, float mult) { notImpl(); }
  void RectifyBBox(Matrix& width_offset, Matrix& height_offset, Matrix& flip,
                   int patch_width, int patch_height) { notImpl(); }

  static void LogisticCEDeriv(Matrix& state, Matrix& gt, Matrix& deriv) { notImpl(); }
  static void LogisticCorrect(Matrix& state, Matrix& gt, Matrix& output) { notImpl(); }
  static void SoftmaxCEDeriv(Matrix& state, Matrix& gt, Matrix& deriv) { notImpl(); }
  static void SoftmaxCorrect(Matrix& state, Matrix& gt, Matrix& output) { notImpl(); }
  static void SoftmaxCE(Matrix& state, Matrix& gt, Matrix& output) { notImpl(); }
  static void SoftmaxDistCE(Matrix& state, Matrix& gt, Matrix& output);
  static void HingeLossDeriv(Matrix& state, Matrix& gt, Matrix& deriv,
                             bool quadratic, float margin) { notImpl(); }
  static void AdagradUpdate(Matrix& adagrad_history, Matrix& gradient, float delta) { notImpl(); }
  static void RMSPropUpdate(Matrix& rms_history, Matrix& gradient, float factor) { notImpl(); }
  static void Dot(Matrix& a, Matrix& b, Matrix& c, float alpha, float beta);
  static void Dot(Matrix& a, Matrix& b, Matrix& c, float alpha, float beta,
                  bool transpose_a, bool transpose_b);

  static void ConvUp(Matrix& input, Matrix& w, Matrix& output,
                     ConvDesc &conv_desc, float scale_targets);

  static void Conv3DUp(Matrix& input, Matrix& w, Matrix& output,
                       ConvDesc &conv_desc, float scale_targets) { notImpl(); }

  static void ConvDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                       ConvDesc &conv_desc, float scale_targets) { notImpl(); }

  static void Conv3DDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                         ConvDesc &conv_desc, float scale_targets) { notImpl(); }

  static void ConvOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                       ConvDesc &conv_desc, int partial_sum_y, int partial_sum_x,
                       float scale_targets, float scale_outputs) { notImpl(); }

  static void Conv3DOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                         ConvDesc &conv_desc, float scale_targets,
                         float scale_outputs) { notImpl(); }

  static void LocalUp(Matrix& input, Matrix& w, Matrix& output,
                      ConvDesc &conv_desc, float scale_targets) { notImpl(); }

  static void LocalDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                        ConvDesc &conv_desc, float scale_targets) { notImpl(); }

  static void LocalOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                        ConvDesc &conv_desc,
                        float scale_targets, float scale_outputs) { notImpl(); }

  static void ConvMaxPool(Matrix& input, Matrix& output, ConvDesc &conv_desc);

  static void ConvMaxPoolUndo(Matrix& input, Matrix& deriv_output, Matrix& output,
                              Matrix& deriv_input, ConvDesc &conv_desc,
                              float scale_targets);

  static void ConvAvgPool(Matrix& input, Matrix& output, ConvDesc &conv_desc);

  static void ConvAvgPoolUndo(Matrix& input, Matrix& deriv_output,
                              ConvDesc &conv_desc, float scale_targets);

  static void ConvResponseNormCrossMap(
      Matrix& input, Matrix& output, int numFilters, int sizeF, float addScale,
      float powScale, bool blocked);

  static void ConvResponseNormCrossMap3D(
      Matrix& input, Matrix& output, int numFilters, int sizeF, float addScale,
      float powScale, bool blocked, int image_size_t) { notImpl(); }

  static void ConvResponseNormCrossMapUndo(
    Matrix& outGrads, Matrix& inputs, Matrix& acts, Matrix& targets, int numFilters,
    int sizeF, float addScale, float powScale, bool blocked) { notImpl(); }
  
  static void ConvResponseNormCrossMapUndo3D(
    Matrix& outGrads, Matrix& inputs, Matrix& acts, Matrix& targets, int numFilters,
    int sizeF, float addScale, float powScale, bool blocked, int image_size_t) { notImpl(); }

  static void ConvUpSample(Matrix& input, Matrix& output, int factor,
                           float scaleTargets) { notImpl(); }
  
  static void ConvDownSample(Matrix& input, Matrix& output, int factor) { notImpl(); }

  static void ConvRGBToYUV(Matrix& input, Matrix& output) { notImpl(); }

  static void ExtractPatches(Matrix& source, Matrix& dest, Matrix& width_offset,
                             Matrix& height_offset, Matrix& flip_bit,
                             int image_size_y, int image_size_x, int patch_size_y,
                             int patch_size_x);

  void ApplySoftmax2();
  static void ConvUp2(Matrix& input, Matrix& w, Matrix& output,
                     ConvDesc &conv_desc, float scale_targets);
  static void ConvMaxPool2(Matrix& input, Matrix& output, ConvDesc &conv_desc);
  static void ConvResponseNormCrossMap2(
      Matrix& input, Matrix& output, int numFilters, int sizeF, float addScale,
      float powScale, bool blocked);
  static void FCUp(Matrix& input, Matrix& w, Matrix& output,
    int num_images, int num_outputs, int num_inputs, float scale_targets);
  static void AddBias(Matrix& input, Matrix& b, Matrix& output, const int num_images, const int num_dims);
  static void Transpose(const float* i_data, float* o_data, int num_filters,
    int kernel_width, int kernel_height, int num_colors);

  static void GetOnes(size_t rows, size_t cols, Matrix& ones) { notImpl(); }
  static void RegisterTempMemory(size_t size) { /*Do nothing*/ }
  static void RegisterTempMemory(size_t size, const std::string& why) { /*Do nothing*/ }
  static void RegisterOnes(size_t size) { /*Do nothing*/ }
  static void GetTemp(size_t rows, size_t cols, Matrix& temp);
  static void InitRandom(int seed);
  static void SetupCUDADevice(int gpu_id) { /*Do nothing*/ }
  static void SetupCUDADevices(const std::vector<int>& boards) { /*Do nothing*/ }
  static void SetDevice(int gpu_id) { /*Do nothing*/ }
  static void SyncAllDevices() { /*Do nothing*/ }
  static int GetDevice() { return 0; }
  static int GetNumBoards() { return 1; }
  static void ShowMemoryUsage() { notImpl(); }

protected:
  eigenmat* GetMat() { return mat_; }
  eigenmat* GetMatTranspose() { return mat_t_; }
  void FreeMemory();

private:
  eigenmat *mat_;
  eigenmat *mat_t_;
  Shape4D shape_;

  static Matrix temp_;
};

#endif
