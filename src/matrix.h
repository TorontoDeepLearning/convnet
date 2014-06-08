#ifndef MATRIX_H_
#define MATRIX_H_
#include <string>
#include "cudamat.cuh"
#include "cublas.h"
#include "hdf5.h"
#include <vector>
using namespace std;

class Matrix {
 public:
  Matrix();
  Matrix(const int rows, const int cols, const bool on_gpu);
  ~Matrix();
  
  void Tie(Matrix &m);
  void AllocateGPUMemory(const int rows, const int cols, const string& name);
  void AllocateGPUMemory(const int rows, const int cols);
  void AllocateMainMemory(const int rows, const int cols);
  //void ClearMainMemory();
  //void ClearGPUMemory();
  void Set(const float val);
  void Set(Matrix& val);
  void GetSlice(Matrix& slice, int start, int end);
  void FillWithRand();
  void FillWithRandn();
  void CopyToHost();
  void CopyToDevice();
  void CopyToDeviceSlice(const int start, const int end);
  void CopyToHostSlice(const int start, const int end);
  void CopyFromMainMemory(Matrix& mat);
  void Reshape(const int rows, const int cols);
  float Norm();
  void Print();
  void WriteToFile(FILE* file);
  void ReadFromFile(FILE* file);
  void WriteHDF5(hid_t file, const string& name);
  void ReadHDF5(hid_t file, const string& name);
  void AllocateAndReadHDF5(hid_t file, const string& name);
  string GetShapeString();
  cudamat* GetMat() { return &mat_; }
  cudamat* GetMatTranspose() { return &mat_t_; }
  float* GetHostData() { return mat_.data_host; }
  int GetRows() const {return mat_.size[0];}
  int GetCols() const {return mat_.size[1];}
  int GetNumEls() const {return mat_.size[1] * mat_.size[0]; }
  float Sum();
  void Add(Matrix& m);
  void SquashRelu();

  int GetGPUId() const { return gpu_id_; }
  void SetReady();
  void WaitTillReady();

  static void GetOnes(int rows, int cols, Matrix& ones);
  static void RegisterTempMemory(int size);
  static void RegisterTempMemory(int size, const string& why);
  static void RegisterOnes(int size);
  static void GetTemp(int rows, int cols, Matrix& temp);
  static void InitRandom(int seed);
  static void SetupCUDADevice(int gpu_id);
  static void SetupCUDADevices(const vector<int>& boards);
  static void SetDevice(int gpu_id);
  static void SyncAllDevices();
  static int GetDevice();
  static int GetNumBoards() {return num_boards_;}

  static vector<Matrix> ones_, temp_;
  static vector<rnd_struct> rnd_;

 private:
  cudamat mat_, mat_t_;
  cudaEvent_t ready_;
  int gpu_id_;
  string name_;
  static int num_boards_;
  static int current_gpu_id_;
  static vector<int> boards_, temp_size_, ones_size_;
};

#endif
