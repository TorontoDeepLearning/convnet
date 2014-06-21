#include "matrix.h"
#include "util.h"
#include <cstdio>
#include <iostream>
#include <sstream>

vector<rnd_struct> Matrix::rnd_;
vector<Matrix> Matrix::ones_, Matrix::temp_;
vector<int> Matrix::boards_, Matrix::temp_size_, Matrix::ones_size_;
int Matrix::current_gpu_id_ = 0, Matrix::num_boards_ = 0;

Matrix::Matrix() {
  mat_.data_host = NULL;
  mat_.data_device = NULL;
  mat_.on_host = 1;
  mat_.on_device = 0;
  mat_.is_trans = 0;
  mat_.size[0] = 0;
  mat_.size[1] = 0;
  mat_t_ = mat_;
  mat_t_.is_trans = 1;
}


Matrix::~Matrix() {
  //if (mat_.data_host != NULL) free(mat_.data_host);
}

Matrix::Matrix(const int rows, const int cols, const bool on_gpu) {
  Matrix();
  if (on_gpu) {
    AllocateGPUMemory(rows, cols);
  } else {
    AllocateMainMemory(rows, cols);
  }
}

void Matrix::Tie(Matrix &m) {
  cout << "Tying" << endl;
  mat_ = *(m.GetMat());
  mat_t_ = *(m.GetMatTranspose());
}

void Matrix::AllocateGPUMemory(const int rows, const int cols) {
  AllocateGPUMemory(rows, cols, "");
}
void Matrix::AllocateGPUMemory(const int rows, const int cols, const string& name) {
  if (rows != mat_.size[0] || cols != mat_.size[1]) {
    name_ = name;
    gpu_id_ = current_gpu_id_;
    if (gpu_id_ < 0 || gpu_id_ >= num_boards_) {
      cerr << "This should not happen" << endl;
      exit(1);
    }
    if (GetNumEls() > 0) free_device_memory(&mat_);
    AllocateMainMemory(rows, cols);
    CopyToDevice();
    mat_t_ = mat_;
    mat_t_.is_trans = 1;
    //const int size = (rows * cols * sizeof(float)) >> 20;
    //cout << "Allocated GPU memory " << rows << " * " << cols << " " << size << "MB for " << name << endl;
    cuda_create_event(&ready_);
  }
}

void Matrix::AllocateMainMemory(const int rows, const int cols) {
  if (mat_.data_host != NULL) free(mat_.data_host);
  mat_.data_host = (float*)calloc(rows * cols, sizeof(float));
  if (mat_.data_host == NULL) {
    cerr << "Error: Could not allocate main memory for matrix of size "
         << rows << " by " << cols << "." << endl;
    exit(1);
  }
  mat_.size[0] = rows;
  mat_.size[1] = cols;
  mat_.on_device = 0;
  mat_.on_host = 1;
  mat_.is_trans = 0;
  mat_.owns_data = 1;
}

void Matrix::CopyToDevice() {
  CopyToDeviceSlice(0, mat_.size[1]);
}

void Matrix::CopyToHost() {
  CopyToHostSlice(0, mat_.size[1]);
}

void Matrix::CopyFromMainMemory(Matrix& mat) {
  float* src = mat.GetHostData();
  float* dest = GetHostData();
  memcpy(dest, src, sizeof(float) * GetNumEls());
}

void Matrix::Set(const float val) {
  int err_code = assign_scalar(&mat_, val);
  if (err_code != 0) {
    cout << "Error: Could not set to scalar : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::Set(Matrix& val) {
  int err_code = copy_on_device(val.GetMat(), &mat_);  // source, dest.
  if (err_code != 0) {
    cout << "Error: Could not set to scalar : " << GetStringError(err_code) << endl;
    exit(1);
  }
}


void Matrix::FillWithRandn() {
  int err_code = fill_with_randn(&rnd_[current_gpu_id_], &mat_);
  if (err_code != 0) {
    cout << "Error: Could not fill with randn : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::FillWithRand() {
  int err_code = fill_with_rand(&rnd_[current_gpu_id_], &mat_);
  if (err_code != 0) {
    cout << "Error: Could not fill with rand : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

float Matrix::Sum() {
  Matrix ones;
  int rows = mat_.size[0];
  int cols = mat_.size[1];
  reshape(&mat_, 1, -1);
  GetOnes(1, rows * cols, ones);
  int err;
  // cout << "Summing matrix of shape " << rows << " " << cols << endl;
  float res = vdot(ones.GetMat(), &mat_, &err);
  if (err != 0) {
    cerr << "Error in vdot " << GetStringError(err) << endl;
    exit(1);
  }
  reshape(&mat_, rows, cols);
  return res;
}

void Matrix::Add(Matrix& m) {
  add_elementwise(&mat_, m.GetMat(), &mat_);
}

void Matrix::CopyToDeviceSlice(const int start, const int end) {
  int err_code = copy_to_device_slice(&mat_, start, end);
  if (err_code != 0) {
    cerr << "Error copying matrix of size " << mat_.size[0] << " "
         << mat_.size[1] << " slice " << start << ":" << end << " to device: "
         << GetStringError(err_code) << endl;
    exit(1);
  } else {
    //cout << "Successfully copied matrix of size " << mat_.size[0] << " " << mat_.size[1] << " to device." << endl;
  }
}

void Matrix::CopyToHostSlice(const int start, const int end) {
  int err_code = copy_to_host_slice(&mat_, start, end);
  if (err_code != 0) {
    cerr << "Error copying matrix of size " << mat_.size[0] << " "
         << mat_.size[1] << " slice " << start << ":" << end << " to host: "
         << GetStringError(err_code) << endl;
    exit(1);
  } else {
    //cout << "Successfully copied matrix of size " << mat->size[0] << " " << mat->size[1] << " to device." << endl;
  }
}


void Matrix::Reshape(int rows, int cols) {
  reshape(&mat_, rows, cols);
  mat_t_ = mat_;
  mat_t_.is_trans = 1;
}

void Matrix::Print() {
  Matrix::SetDevice(gpu_id_);
  WaitTillReady();
  cuda_sync_threads();
  int err_code = copy_to_host(&mat_);
  if (err_code != 0) {
    cout << "Error: Could not copy to host : " << GetStringError(err_code) << endl;
    exit(1);
  }
  for (int i = 0; i < mat_.size[0]; i++) {
    if (i < 10) {
      for (int j = 0; j < mat_.size[1]; j++) {
        if (j < 10) {
          printf("%.5f ", mat_.data_host[j * mat_.size[0] + i]);
        } else {
          printf(". . . ");
          break;
        }
      }
    } else {
      printf(". . .\n");
      break;
    }
    printf("\n");
  }
}

string Matrix::GetShapeString() {
  stringstream ss;
  ss << mat_.size[0] << " " << mat_.size[1];
  return ss.str();
}

void Matrix::WriteToFile(FILE* file) {
  copy_to_host(&mat_);
  fwrite(mat_.size, sizeof(int), 2, file);
  fwrite(mat_.data_host, sizeof(float), mat_.size[0] * mat_.size[1], file);
}

void Matrix::ReadFromFile(FILE* file) {
  fread(mat_.size, sizeof(int), 2, file);
  fread(mat_.data_host, sizeof(float), mat_.size[0] * mat_.size[1], file);
  int err_code = copy_to_device(&mat_);
  if (err_code != 0) {
    cout << "Error copying matrix to device : " << GetStringError(err_code) << endl;
    exit(1);
  }
}

void Matrix::WriteHDF5(hid_t file, const string& name) {
  copy_to_host(&mat_);
  // cols, rows swapped because cudamat is col major, but hdf5 is row major.
  WriteHDF5CPU(file, mat_.data_host, mat_.size[1], mat_.size[0], name);
}

void Matrix::ReadHDF5(hid_t file, const string& name) {
  ReadHDF5CPU(file, mat_.data_host, mat_.size[0] * mat_.size[1], name);
  copy_to_device(&mat_);
}

void Matrix::AllocateAndReadHDF5(hid_t file, const string& name) {
  int rows, cols;
  ReadHDF5Shape(file, name, &rows, &cols);
  AllocateGPUMemory(rows, cols);
  ReadHDF5(file, name);
}

void Matrix::GetOnes(int rows, int cols, Matrix& ones) {
  Matrix& o = Matrix::ones_[current_gpu_id_];
  int size = o.GetCols();
  if (size == 0) {  // Allocate memory on first call to GetOnes.
    o.AllocateGPUMemory(1, ones_size_[current_gpu_id_]);
    o.Set(1);
    size = ones_size_[current_gpu_id_];
  }
  if (rows * cols > size) {
    cerr << "Ones has only " << size << " elements. Requested was "
         << rows << " * " << cols << endl;
    exit(1);
  }
  get_slice(o.GetMat(), ones.GetMat(), 0, rows * cols);
  ones.Reshape(rows, cols);
}

void Matrix::GetSlice(Matrix& slice, int start, int end) {
  get_slice(&mat_, slice.GetMat(), start, end);
}

void Matrix::GetTemp(int rows, int cols, Matrix& temp) {
  Matrix& t = Matrix::temp_[current_gpu_id_];
  int size = t.GetNumEls();
  const int length = rows * cols;
  if (length > size) {  // Allocate memory as required.
    t.AllocateGPUMemory(1, temp_size_[current_gpu_id_]);
    size = temp_size_[current_gpu_id_];
    //cout << "Allocated " << (temp_size_[current_gpu_id_] >> 18) << " MB for temp." << endl;
  }
  /*
  if (length > size) {
    cerr << "Temp has only " << size << " elements. Requested was " << length << endl;
    exit(1);
  }
  */
  get_slice(t.GetMat(), temp.GetMat(), 0, length);
  reshape(temp.GetMat(), rows, cols);
}

float Matrix::Norm() {
  int err_code;
  float res = euclid_norm(&mat_, &err_code);
  if (err_code != 0) {
    cerr << "Error in Matrix::Norm " << GetStringError(err_code) << endl;
    exit(1);
  }
  return res;
}

void Matrix::SquashRelu() {
  apply_relu_squash(&mat_, &mat_, 2);
}

void Matrix::SetupCUDADevices(const vector<int>& boards) {
  int err_code;
  num_boards_ = boards.size();
  bool check_p2p_fermi  = num_boards_ > 1;
  for (int i = 0; i < num_boards_; i++) {
    boards_.push_back(boards[i]);
    err_code = cuda_set_device(boards[i]);
    if (err_code != 0) {
      cerr << "Error setting device id! " << GetStringError(err_code) << endl;
      exit(1);
    }
    if (check_p2p_fermi && !cuda_is_fermi(boards[i])) {
      cerr << "Error : Board is not Fermi! " << GetStringError(err_code) << endl;
      exit(1);
    }
  }
  if (check_p2p_fermi) {
    // Setup P2P.
    err_code = cuda_set_P2P(boards[0], boards[1]);
    if (err_code != 0) {
      cerr << "Warning : Could not set up P2P, GPU-to-GPU communication will be slow. "
           << GetStringError(err_code) << endl;
    }
  }
  err_code = cublas_init();
  if (err_code != 0) {
    cerr << "Error initializing cublas!" << GetStringError(err_code) << endl;
    exit(1);
  }
  temp_size_.resize(num_boards_);
  ones_size_.resize(num_boards_);
  rnd_.resize(num_boards_);
  ones_.resize(num_boards_);
  temp_.resize(num_boards_);

  for (int i = 0; i < num_boards_; i++) {
    temp_size_[i] = 0;
    ones_size_[i] = 128*256*256;
  }
  cuda_set_device(boards[0]);
}

void Matrix::SetupCUDADevice(int board) {
  vector<int> boards;
  boards.push_back(board);
  SetupCUDADevices(boards);
}

int Matrix::GetDevice() {
  int board;
  cudaError_t err = cudaGetDevice(&board);
  if (err != cudaSuccess) {
    cerr << "Could not get which board is current." << endl;
    exit(1);
  }
  for (int i = 0; i < num_boards_; i++) {
    if (boards_[i] == board) return i;
  }
  cerr << "current board was not set" << endl;
  exit(1);
  return 0;
}

void Matrix::SetDevice(int gpu_id) {
  if (num_boards_ < 2) return;
  if (current_gpu_id_ == gpu_id) return;
  int err_code = cuda_set_device(boards_[gpu_id]);
  if (err_code != 0) {
    cerr << "Error setting device id! " << GetStringError(err_code) << endl;
    exit(1);
  }
  current_gpu_id_ = gpu_id;
}
/*
void Matrix::SyncDevice(int gpu_id) {
  if (num_boards_ < 2) return;
  int old_id = current_gpu_id_;
  SetDevice(gpu_id);
  cuda_sync_threads();
  SetDevice(old_id);
}
*/

void Matrix::SyncAllDevices() {
  if (num_boards_ < 2) return;
  for (int i = 0; i < num_boards_; i++) {
    SetDevice(i);
    cuda_sync_threads();
  }
  SetDevice(0);
}

void Matrix::InitRandom(int seed){
  for (int i = 0; i < num_boards_; i++) {
    SetDevice(i);
    init_random(&rnd_[i], seed + i, NULL);
  }
}

void Matrix::RegisterTempMemory(int size, const string& why) {
  if (size > temp_size_[current_gpu_id_]) {
    temp_size_[current_gpu_id_] = size;
    //cout << "Max for " << why << " " << size << endl;
  }
}

void Matrix::RegisterTempMemory(int size) {
  RegisterTempMemory(size, "");
}

void Matrix::RegisterOnes(int size) {
  if (size > ones_size_[current_gpu_id_]) {
    ones_size_[current_gpu_id_] = size;
  }
}

void Matrix::SetReady() {
  if (num_boards_ < 2) return;
  int err_code;
  if (current_gpu_id_ != gpu_id_) {
    cerr << "Error: Current gpu must be same as the one on which the event was created." << endl;
    exit(1);
  }
  err_code = cuda_record_event(&ready_);
  if (err_code != 0) {
    cerr << "Error: Could not set ready." << endl;
    exit(1);
  }
}

void Matrix::WaitTillReady() {
  if (num_boards_ < 2) return;
  int err_code = cuda_synchronize_event(&ready_);
  if (err_code != 0) {
    cerr << "Error: Could not synchronize." << endl;
    exit(1);
  }
}
