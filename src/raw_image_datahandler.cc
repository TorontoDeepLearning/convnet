#include "raw_image_datahandler.h"
#include <fstream>

template<typename T>
RawImageDataHandler<T>::RawImageDataHandler(const config::DatasetConfig& config):
  DataHandler(config),
  image_size_(config.image_size()),
  pixelwise_normalize_(config.pixelwise_normalize()) {
  string data_file = base_dir_ + config.file_pattern();
  bool flip = config.can_flip();
  bool translate = config.can_translate();
  num_positions_ = ((flip ? 2 : 1) * (translate ? 5 : 1));
  it_ = new RawImageFileIterator<T>(data_file, image_size_,
                                 config.raw_image_size(), flip, translate);
  dataset_size_ = it_->GetDatasetSize();
  int max_dataset_size = config.max_dataset_size();
  if (max_dataset_size > 0 && max_dataset_size < dataset_size_) {
    dataset_size_ = max_dataset_size;
  }
  int num_dims = image_size_ * image_size_ * 3;
  image_buf_ = new T[num_dims];
  LoadMeansFromDisk();
}

template<typename T>
RawImageDataHandler<T>::~RawImageDataHandler() {
  delete it_;
  delete image_buf_;
}

template<typename T>
void RawImageDataHandler<T>::LoadMeansFromDisk() {
  string data_file = base_dir_ + mean_file_;
  hid_t file = H5Fopen(data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (pixelwise_normalize_) {
    Matrix pixel_mean, pixel_std;
    pixel_mean.AllocateAndReadHDF5(file, "pixel_mean");
    pixel_std.AllocateAndReadHDF5(file, "pixel_std");
    pixel_mean.Reshape(1, -1);
    pixel_std.Reshape(1, -1);
    int num_channels = pixel_mean.GetCols();

    mean_.AllocateGPUMemory(image_size_ * image_size_, num_channels);
    std_.AllocateGPUMemory(image_size_ * image_size_, num_channels);

    add_row_vec(mean_.GetMat(), pixel_mean.GetMat(), mean_.GetMat());
    add_row_vec(std_.GetMat(), pixel_std.GetMat(), std_.GetMat());

    mean_.Reshape(-1, 1);
    std_.Reshape(-1, 1);
    mean_.CopyToHost();
    std_.CopyToHost();
  } else {
    mean_.AllocateAndReadHDF5(file, "mean");
    std_.AllocateAndReadHDF5(file, "std");
  }
  H5Fclose(file);
}

template<typename T>
void RawImageDataHandler<T>::GetBatch(vector<Layer*>& data_layers) {
  Matrix::SyncAllDevices();
  Matrix::SetDevice(gpu_id_);
  Matrix& state = data_layers[0]->GetState();
  int batch_size = state.GetRows();
  int num_dims = image_size_ * image_size_ * 3;
  float *data_ptr = state.GetHostData();
  float *mean_ptr = mean_.GetHostData();
  float *std_ptr = std_.GetHostData();
  for (int i = 0; i < batch_size; i++) {
    it_->GetNext(image_buf_);
    for (int j = 0; j < num_dims; j++) {
      data_ptr[j * batch_size + i] = (static_cast<float>(image_buf_[j]) - mean_ptr[j])/std_ptr[j];
    }
  }
  state.CopyToDevice();
}
template class RawImageDataHandler<float>;
template class RawImageDataHandler<unsigned char>;

SlidingWindowDataHandler::SlidingWindowDataHandler(const config::DatasetConfig& config):
  DataHandler(config), image_id_(0) {
  int image_size = config.image_size();
  it_ = new SlidingWindowIterator(config.image_size(), config.stride());
  buf_ = new float[image_size * image_size * 3];
  LoadMeansFromDisk();
  string data_file = base_dir_ + config.file_pattern();

  bool success = ReadLines(data_file, image_file_names_);
  if (!success) exit(1);
  dataset_size_ = 0;
  for (const string& s: image_file_names_) {
    it_->SetImage(s);
    dataset_size_ += it_->GetNumWindows();
  }
  it_->Reset();
}

SlidingWindowDataHandler::~SlidingWindowDataHandler() {
  delete it_;
  delete buf_;
}

void SlidingWindowDataHandler::LoadMeansFromDisk() {
  string data_file = base_dir_ + mean_file_;
  hid_t file = H5Fopen(data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  mean_.AllocateAndReadHDF5(file, "mean");
  std_.AllocateAndReadHDF5(file, "std");
  H5Fclose(file);
}


void SlidingWindowDataHandler::GetBatch(vector<Layer*>& data_layers) {
  Matrix::SyncAllDevices();
  Matrix::SetDevice(gpu_id_);

  Matrix& mat = data_layers[0]->GetState();
  int batch_size = mat.GetRows(), num_dims = mat.GetCols();
  float* data_ptr = mat.GetHostData();
  float *mean_ptr = mean_.GetHostData();
  float *std_ptr = std_.GetHostData();
  for (int i = 0; i < batch_size; i++) {
    if (it_->Done()) {
      cout << "Loading image " << image_id_ << endl;
      it_->SetImage(image_file_names_[image_id_++]);
      if (image_id_ == image_file_names_.size()) image_id_ = 0;
    }
    it_->GetNext(buf_);
    for (int j = 0; j < num_dims; j++) {
      data_ptr[j * batch_size + i] = (buf_[j] - mean_ptr[j]) / std_ptr[j];
    }
  }
  mat.CopyToDevice();
  mat.SetReady();
}
