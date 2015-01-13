#ifndef MATRIX_H
#define MATRIX_H

#ifdef USE_GEMM
#include "../cudamat/cudamat_conv_gemm.cuh"
#else
#include "../cudamat/cudamat_conv.cuh"
#endif
#include "../cudamat/cudamat.cuh"

#include <hdf5.h>

#include <vector>
#include <string>
#include <map>

/** A GPU matrix class.*/
class Matrix {
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
  void CopyToHost();
  void CopyToDevice();
  void CopyToDeviceSlice(const size_t start, const size_t end);
  void CopyToHostSlice(const size_t start, const size_t end);
  void CopyFromMainMemory(Matrix& mat);
  void Reshape(const size_t rows, const size_t cols);
  void Print();
  void PrintToFile(const std::string& filename);
  bool CheckNaN();
  void WriteHDF5(hid_t file, const std::string& name);
  void ReadHDF5(hid_t file, const std::string& name);
  void AllocateAndReadHDF5(hid_t file, const std::string& name);
  std::string GetShapeString();
  std::string GetShape4DString();

  float* GetHostData() { return mat_.data_host; }
  size_t GetRows() const {return mat_.size[0];}
  size_t GetCols() const {return mat_.size[1];}
  size_t GetNumEls() const {return mat_.size[1] * mat_.size[0]; }

  int GetGPUId() const { return gpu_id_; }
  void SetGPUId(int gpu_id) { gpu_id_ = gpu_id; }
  void SetReady();
  void WaitTillReady();

  // GPU computing methods.
  void Add(float val);
  void Add(Matrix& m);
  void Add(Matrix& m, float alpha);
  void SquashRelu();
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
  void NormLimitByAxis(int axis, float val, bool constraint);
  void NormalizeColumnwise();
  void Dropout(float dropprob, float fill_value, float scale_factor);
  void ApplyDerivativeOfReLU(Matrix& state);
  void ApplySoftmax();
  void ApplyLogistic();
  void ApplyDerivativeOfLogistic(Matrix& state);
  float EuclidNorm();
  float VDot(Matrix& m);
  void CopyTransposeBig(Matrix& m);
  void CopyTranspose(Matrix& m);
  void ShuffleColumns(Matrix& rand_perm_indices);
  void AddToEachPixel(Matrix& v, float mult);
  void RectifyBBox(Matrix& width_offset, Matrix& height_offset, Matrix& flip,
                   int patch_width, int patch_height);

  static void LogisticCEDeriv(Matrix& state, Matrix& gt, Matrix& deriv);
  static void LogisticCorrect(Matrix& state, Matrix& gt, Matrix& output);
  static void SoftmaxCEDeriv(Matrix& state, Matrix& gt, Matrix& deriv);
  static void SoftmaxCorrect(Matrix& state, Matrix& gt, Matrix& output);
  static void SoftmaxCE(Matrix& state, Matrix& gt, Matrix& output);
  static void SoftmaxDistCE(Matrix& state, Matrix& gt, Matrix& output);
  static void HingeLossDeriv(Matrix& state, Matrix& gt, Matrix& deriv,
                             bool quadratic, float margin);
  static void AdagradUpdate(Matrix& adagrad_history, Matrix& gradient, float delta);
  static void RMSPropUpdate(Matrix& rms_history, Matrix& gradient, float factor);
  static void Dot(Matrix& a, Matrix& b, Matrix& c, float alpha, float beta);
  static void Dot(Matrix& a, Matrix& b, Matrix& c, float alpha, float beta,
                  bool transpose_a, bool transpose_b);

  static void ConvUp(Matrix& input, Matrix& w, Matrix& output,
                     ConvDesc conv_desc, float scale_targets);
  
  static void Conv3DUp(Matrix& input, Matrix& w, Matrix& output,
                     ConvDesc conv_desc, float scale_targets);

  static void ConvDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                       ConvDesc conv_desc, float scale_targets);
  
  static void Conv3DDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                       ConvDesc conv_desc, float scale_targets);

  static void ConvOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                       ConvDesc conv_desc, int partial_sum_y, int partial_sum_x,
                       float scale_targets, float scale_outputs);
  
  static void Conv3DOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                         ConvDesc conv_desc, float scale_targets,
                         float scale_outputs);

  static void LocalUp(Matrix& input, Matrix& w, Matrix& output,
                      ConvDesc conv_desc, float scale_targets);

  static void LocalDown(Matrix& deriv_output, Matrix& w, Matrix& deriv_input,
                        ConvDesc conv_desc, float scale_targets);

  static void LocalOutp(Matrix& input, Matrix& deriv_output, Matrix& dw,
                        ConvDesc conv_desc,
                        float scale_targets, float scale_outputs);

  static void ConvMaxPool(Matrix& input, Matrix& output, ConvDesc conv_desc);

  static void ConvMaxPoolUndo(Matrix& input, Matrix& deriv_output, Matrix& output,
                              Matrix& deriv_input, ConvDesc conv_desc,
                              float scale_targets);

  static void ConvAvgPool(Matrix& input, Matrix& output, ConvDesc conv_desc);

  static void ConvAvgPoolUndo(Matrix& input, Matrix& deriv_output,
                              ConvDesc conv_desc, float scale_targets);

  static void ConvResponseNormCrossMap(
      Matrix& input, Matrix& output, int numFilters, int sizeF, float addScale,
      float powScale, bool blocked);

  static void ConvResponseNormCrossMap3D(
      Matrix& input, Matrix& output, int numFilters, int sizeF, float addScale,
      float powScale, bool blocked, int image_size_t);

  static void ConvResponseNormCrossMapUndo(
    Matrix& outGrads, Matrix& inputs, Matrix& acts, Matrix& targets, int numFilters,
    int sizeF, float addScale, float powScale, bool blocked);
  
  static void ConvResponseNormCrossMapUndo3D(
    Matrix& outGrads, Matrix& inputs, Matrix& acts, Matrix& targets, int numFilters,
    int sizeF, float addScale, float powScale, bool blocked, int image_size_t);

  static void ConvUpSample(Matrix& input, Matrix& output, int factor,
                           float scaleTargets);
  
  static void ConvDownSample(Matrix& input, Matrix& output, int factor);

  static void ConvRGBToYUV(Matrix& input, Matrix& output);

  static void ExtractPatches(Matrix& source, Matrix& dest, Matrix& width_offset,
                             Matrix& height_offset, Matrix& flip_bit,
                             int image_size_y, int image_size_x, int patch_size_y,
                             int patch_size_x);

  static void GetOnes(size_t rows, size_t cols, Matrix& ones);
  static void RegisterTempMemory(size_t size);
  static void RegisterTempMemory(size_t size, const std::string& why);
  static void RegisterOnes(size_t size);
  static void GetTemp(size_t rows, size_t cols, Matrix& temp);
  static void InitRandom(int seed);
  static void SetupCUDADevice(int gpu_id);
  static void SetupCUDADevices(const std::vector<int>& boards);
  static void SetDevice(int gpu_id);
  static void SyncAllDevices();
  static int GetDevice();
  static int GetNumBoards() {return num_boards_;}
  static void ShowMemoryUsage();

protected:
  static std::vector<Matrix> ones_, temp_;
  static std::vector<rnd_struct> rnd_;
  static std::map<std::string, long> gpu_memory_;
  static Matrix rgb_to_yuv_mat_;

  cudamat* GetMat() { return &mat_; }
  cudamat* GetMatTranspose() { return &mat_t_; }

  void WriteToFile(FILE* file);
  void ReadFromFile(FILE* file);
  float Norm(); // duplicated

private:
  cudamat mat_, mat_t_;
  cudaEvent_t ready_;
  Shape4D shape_;
  int gpu_id_;
  std::string name_;
  static int num_boards_;
  static int current_gpu_id_;
  static std::vector<int> boards_;
  static std::vector<size_t> temp_size_, ones_size_;
};

#endif
