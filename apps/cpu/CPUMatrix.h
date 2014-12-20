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
class CPUMatrix
{
public:
  CPUMatrix();
  CPUMatrix(const size_t rows, const size_t cols, const bool on_gpu);
  ~CPUMatrix();

  void Tie(CPUMatrix &m);
  void SetupTranspose();
  void SetShape4D(int d1, int d2, int d3, int d4);
  void SetShape4D_like(CPUMatrix& mat);
  Shape4D& GetShape4D();
  void AllocateGPUMemory(const size_t rows, const size_t cols, const std::string& name);
  void AllocateGPUMemory(const size_t rows, const size_t cols);
  void AllocateMainMemory(const size_t rows, const size_t cols);
  void Set(const float val);
  void Set(CPUMatrix& val);
  float ReadValue(int row, int col);
  float ReadValue(int index);
  void WriteValue(int row, int col, float val);
  void WriteValue(int index, float val);
  void CopyP2PAsync(CPUMatrix& val);
  void GetSlice(CPUMatrix& slice, size_t start, size_t end);
  void FillWithRand();
  void FillWithRandn();
  void CopyToHost() { /*Do nothing*/ }
  void CopyToDevice() { /*Do nothing*/ }
  void CopyToDeviceSlice(const size_t start, const size_t end) { /*Do nothing*/ }
  void CopyToHostSlice(const size_t start, const size_t end) { /*Do nothing*/ }
  void CopyFromMainMemory(CPUMatrix& mat);
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
  void Add(CPUMatrix& m);
  void Add(CPUMatrix& m, float alpha);
  void SquashRelu() { notImpl(); }
  void AddRowVec(CPUMatrix& v);
  void AddRowVec(CPUMatrix& v, float alpha);
  void AddColVec(CPUMatrix& v, float alpha);
  void MultByRowVec(CPUMatrix& v);
  void DivideByColVec(CPUMatrix& v);
  float Sum();
  void SumRows(CPUMatrix& target, float alpha, float beta);
  void SumCols(CPUMatrix& target, float alpha, float beta);
  void Mult(float val);
  void Mult(CPUMatrix& val);
  void Divide(float val);
  void Divide(CPUMatrix& val);
  void Subtract(CPUMatrix& m, CPUMatrix& target);
  void LowerBound(float val);
  void Sqrt();
  void UpperBoundMod(float val);
  void SqSumAxis(CPUMatrix& target, int axis, float beta, float alpha);
  void NormLimitByAxis(int axis, float val, bool constraint) { notImpl(); }
  void NormalizeColumnwise() { notImpl(); }
  void Dropout(float dropprob, float fill_value, float scale_factor);
  void ApplyDerivativeOfReLU(CPUMatrix& state);
  void ApplySoftmax();
  void ApplyLogistic();
  void ApplyDerivativeOfLogistic(CPUMatrix& state);
  float EuclidNorm();
  float VDot(CPUMatrix& m);
  void CopyTransposeBig(CPUMatrix& m);
  void CopyTranspose(CPUMatrix& m);
  void ShuffleColumns(CPUMatrix& rand_perm_indices) { notImpl(); }
  void AddToEachPixel(CPUMatrix& v, float mult) { notImpl(); }
  void RectifyBBox(CPUMatrix& width_offset, CPUMatrix& height_offset, CPUMatrix& flip,
                   int patch_width, int patch_height) { notImpl(); }

  static void LogisticCEDeriv(CPUMatrix& state, CPUMatrix& gt, CPUMatrix& deriv) { notImpl(); }
  static void LogisticCorrect(CPUMatrix& state, CPUMatrix& gt, CPUMatrix& output) { notImpl(); }
  static void SoftmaxCEDeriv(CPUMatrix& state, CPUMatrix& gt, CPUMatrix& deriv) { notImpl(); }
  static void SoftmaxCorrect(CPUMatrix& state, CPUMatrix& gt, CPUMatrix& output) { notImpl(); }
  static void SoftmaxCE(CPUMatrix& state, CPUMatrix& gt, CPUMatrix& output) { notImpl(); }
  static void SoftmaxDistCE(CPUMatrix& state, CPUMatrix& gt, CPUMatrix& output);
  static void HingeLossDeriv(CPUMatrix& state, CPUMatrix& gt, CPUMatrix& deriv,
                             bool quadratic, float margin) { notImpl(); }
  static void AdagradUpdate(CPUMatrix& adagrad_history, CPUMatrix& gradient, float delta) { notImpl(); }
  static void RMSPropUpdate(CPUMatrix& rms_history, CPUMatrix& gradient, float factor) { notImpl(); }
  static void Dot(CPUMatrix& a, CPUMatrix& b, CPUMatrix& c, float alpha, float beta);
  static void Dot(CPUMatrix& a, CPUMatrix& b, CPUMatrix& c, float alpha, float beta,
                  bool transpose_a, bool transpose_b);

  static void ConvUp(CPUMatrix& input, CPUMatrix& w, CPUMatrix& output,
                     ConvDesc &conv_desc, float scale_targets);

  static void Conv3DUp(CPUMatrix& input, CPUMatrix& w, CPUMatrix& output,
                       ConvDesc &conv_desc, float scale_targets) { notImpl(); }

  static void ConvDown(CPUMatrix& deriv_output, CPUMatrix& w, CPUMatrix& deriv_input,
                       ConvDesc &conv_desc, float scale_targets) { notImpl(); }

  static void Conv3DDown(CPUMatrix& deriv_output, CPUMatrix& w, CPUMatrix& deriv_input,
                         ConvDesc &conv_desc, float scale_targets) { notImpl(); }

  static void ConvOutp(CPUMatrix& input, CPUMatrix& deriv_output, CPUMatrix& dw,
                       ConvDesc &conv_desc, int partial_sum_y, int partial_sum_x,
                       float scale_targets, float scale_outputs) { notImpl(); }

  static void Conv3DOutp(CPUMatrix& input, CPUMatrix& deriv_output, CPUMatrix& dw,
                         ConvDesc &conv_desc, float scale_targets,
                         float scale_outputs) { notImpl(); }

  static void LocalUp(CPUMatrix& input, CPUMatrix& w, CPUMatrix& output,
                      ConvDesc &conv_desc, float scale_targets) { notImpl(); }

  static void LocalDown(CPUMatrix& deriv_output, CPUMatrix& w, CPUMatrix& deriv_input,
                        ConvDesc &conv_desc, float scale_targets) { notImpl(); }

  static void LocalOutp(CPUMatrix& input, CPUMatrix& deriv_output, CPUMatrix& dw,
                        ConvDesc &conv_desc,
                        float scale_targets, float scale_outputs) { notImpl(); }

  static void ConvMaxPool(CPUMatrix& input, CPUMatrix& output, ConvDesc &conv_desc);

  static void ConvMaxPoolUndo(CPUMatrix& input, CPUMatrix& deriv_output, CPUMatrix& output,
                              CPUMatrix& deriv_input, ConvDesc &conv_desc,
                              float scale_targets);

  static void ConvAvgPool(CPUMatrix& input, CPUMatrix& output, ConvDesc &conv_desc);

  static void ConvAvgPoolUndo(CPUMatrix& input, CPUMatrix& deriv_output,
                              ConvDesc &conv_desc, float scale_targets);

  static void ConvResponseNormCrossMap(
      CPUMatrix& input, CPUMatrix& output, int numFilters, int sizeF, float addScale,
      float powScale, bool blocked);

  static void ConvResponseNormCrossMap3D(
      CPUMatrix& input, CPUMatrix& output, int numFilters, int sizeF, float addScale,
      float powScale, bool blocked, int image_size_t) { notImpl(); }

  static void ConvResponseNormCrossMapUndo(
    CPUMatrix& outGrads, CPUMatrix& inputs, CPUMatrix& acts, CPUMatrix& targets, int numFilters,
    int sizeF, float addScale, float powScale, bool blocked) { notImpl(); }
  
  static void ConvResponseNormCrossMapUndo3D(
    CPUMatrix& outGrads, CPUMatrix& inputs, CPUMatrix& acts, CPUMatrix& targets, int numFilters,
    int sizeF, float addScale, float powScale, bool blocked, int image_size_t) { notImpl(); }

  static void ConvUpSample(CPUMatrix& input, CPUMatrix& output, int factor,
                           float scaleTargets) { notImpl(); }
  
  static void ConvDownSample(CPUMatrix& input, CPUMatrix& output, int factor) { notImpl(); }

  static void ConvRGBToYUV(CPUMatrix& input, CPUMatrix& output) { notImpl(); }

  static void ExtractPatches(CPUMatrix& source, CPUMatrix& dest, CPUMatrix& width_offset,
                             CPUMatrix& height_offset, CPUMatrix& flip_bit,
                             int image_size_y, int image_size_x, int patch_size_y,
                             int patch_size_x);

  void ApplySoftmax2();
  static void ConvUp2(CPUMatrix& input, CPUMatrix& w, CPUMatrix& output,
                     ConvDesc &conv_desc, float scale_targets);
  static void ConvMaxPool2(CPUMatrix& input, CPUMatrix& output, ConvDesc &conv_desc);
  static void ConvResponseNormCrossMap2(
      CPUMatrix& input, CPUMatrix& output, int numFilters, int sizeF, float addScale,
      float powScale, bool blocked);
  static void FCUp(CPUMatrix& input, CPUMatrix& w, CPUMatrix& output,
    int num_images, int num_outputs, int num_inputs, float scale_targets);
  static void AddBias(CPUMatrix& input, CPUMatrix& b, CPUMatrix& output, const int num_images, const int num_dims);
  static void Transpose(const float* i_data, float* o_data, int num_filters,
    int kernel_width, int kernel_height, int num_colors);

  static void GetOnes(size_t rows, size_t cols, CPUMatrix& ones) { notImpl(); }
  static void RegisterTempMemory(size_t size) { /*Do nothing*/ }
  static void RegisterTempMemory(size_t size, const std::string& why) { /*Do nothing*/ }
  static void RegisterOnes(size_t size) { /*Do nothing*/ }
  static void GetTemp(size_t rows, size_t cols, CPUMatrix& temp);
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

  static CPUMatrix temp_;
};

#endif
