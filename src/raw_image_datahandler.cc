#include "raw_image_datahandler.h"
#include <fstream>
#include "CImg.h"

RawImageFileIterator::RawImageFileIterator(
    const string& filelist, const int image_size, const int raw_image_size,
    const bool flip, const bool translate) :
    row_(0), image_id_(-1), position_(0), image_size_(image_size),
    num_positions_((flip ? 2 : 1) * (translate ? 5 : 1)),
    raw_image_size_(raw_image_size) {

  bool success = ReadLines(filelist, filenames_);
  if (!success) exit(1);
  dataset_size_ = filenames_.size();

  disp_ = new CImgDisplay();
}

void RawImageFileIterator::GetNext(float* data_ptr) {
  GetNext(data_ptr, row_, position_);
  position_++;
  if (position_ == num_positions_) {
    position_ = 0;
    row_++;
    if (row_ == dataset_size_) row_ = 0;
  }
}

void RawImageFileIterator::GetCoordinates(
    int width, int height, int position, int* left, int* top, bool* flip) {
  *flip = position >= 5;
  position %= 5;
  int x_slack = width - image_size_;
  int y_slack = height - image_size_;
  switch(position) {
    case 0 :  // Center. 
            *left = x_slack / 2;
            *top = y_slack / 2;
            break;
    case 1 :  // Top left.
            *left = 0;
            *top = 0;
            break;
    case 2 :  // Top right.
            *left = x_slack;
            *top = 0;
            break;
    case 3 :  // Bottom right.
            *left = x_slack;
            *top = y_slack;
            break;
    case 4 :  // Bottom left.
            *left = 0;
            *top = y_slack;
            break;
  }
}

void RawImageFileIterator::GetNext(float* data_ptr, const int row, const int position) {
  if (image_id_ != row) {  // Load a new image from disk.
    image_id_ = row;
    string& filename = filenames_[row];
    image_.assign(filename.c_str());
    
    // Resize it so that the shorter side is image_size_.
    int width = image_.width(), height = image_.height();
    int new_width, new_height;
    if (width > height) {
      new_height = raw_image_size_;
      new_width = (width * raw_image_size_) / height;
    } else {
      new_width = raw_image_size_;
      new_height = (height * raw_image_size_) / width;
    }
    image_.resize(new_width, new_height, 1, -100, 3);
  }
  int width = image_.width(), height = image_.height();
  int left, top;
  bool flip;
  GetCoordinates(width, height, position, &left, &top, &flip);

  CImg<float> img = image_.get_crop(
      left, top, left + image_size_ - 1, top + image_size_ - 1, true);

  if (flip) img.mirror('x');
  //img.display(*disp_);

  int num_image_colors = img.spectrum();
  int num_pixels = image_size_ * image_size_;
  if (num_image_colors >= 3) {  // Image has 3 channels.
    memcpy(data_ptr, img.data(), 3 * num_pixels * sizeof(float));
  } else if (num_image_colors == 1) {  // Image has 1 channel.
    for (int i = 0; i < 3; i++) {
      memcpy(data_ptr + i * num_pixels, img.data(), num_pixels * sizeof(float));
    }
  } else {
    cerr << "Image has " << num_image_colors << "colors." << endl;
    exit(1);
  }
}

RawImageDataHandler::RawImageDataHandler(const config::DatasetConfig& config):
  DataHandler(config),
  image_size_(config.image_size()),
  pixelwise_normalize_(config.pixelwise_normalize()) {
  string data_file = base_dir_ + config.file_pattern();
  bool flip = config.can_flip();
  bool translate = config.can_translate();
  num_positions_ = ((flip ? 2 : 1) * (translate ? 5 : 1));
  it_ = new RawImageFileIterator(data_file, image_size_,
                                 config.raw_image_size(), flip, translate);
  dataset_size_ = it_->GetDatasetSize();
  int max_dataset_size = config.max_dataset_size();
  if (max_dataset_size > 0 && max_dataset_size < dataset_size_) {
    dataset_size_ = max_dataset_size;
  }
  int num_dims = image_size_ * image_size_ * 3;
  image_buf_ = new float[num_dims];
  LoadMeansFromDisk();
}

RawImageDataHandler::~RawImageDataHandler() {
  delete it_;
  delete image_buf_;
}

void RawImageDataHandler::LoadMeansFromDisk() {
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

void RawImageDataHandler::GetBatch(vector<Layer*>& data_layers) {
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
      data_ptr[j * batch_size + i] = (image_buf_[j] - mean_ptr[j])/std_ptr[j];
    }
  }
  state.CopyToDevice();
}

SlidingWindowIterator::SlidingWindowIterator(const int window_size, const int stride):
  window_size_(window_size), stride_(stride), num_windows_(0),
  center_x_(0), center_y_(0), done_(true) {}

void SlidingWindowIterator::SetImage(const string& filename) {
  image_.assign(filename.c_str());
  center_x_ = 0;
  center_y_ = 0;
  int num_modules_x = (image_.width() - window_size_ % 2) / stride_ + 1;
  int num_modules_y = (image_.height() - window_size_ % 2) / stride_ + 1;
  num_windows_ = num_modules_x * num_modules_y;
  done_ = false;
}

void SlidingWindowIterator::Reset() {
  done_ = true;
}

void SlidingWindowIterator::GetNext(float* data_ptr) {
  GetNext(data_ptr, center_x_, center_y_);
  center_x_ += stride_;
  if (center_x_  >= image_.width()) {
    center_x_ = 0;
    center_y_ += stride_;
    if (center_y_ >= image_.height()) {
      center_y_ = 0;
      done_ = true;
    }
  }
}

bool SlidingWindowIterator::Done() {
  return done_;
}

void SlidingWindowIterator::GetNext(float* data_ptr, int center_x, int center_y) {
  int left    = center_x - window_size_ / 2,
      right   = left + window_size_,
      top     = center_y - window_size_ / 2,
      bottom  = top + window_size_;
  CImg<float> img = image_.get_crop(left, top, right - 1, bottom - 1, true);
  int num_pixels = window_size_ * window_size_ * 3;
  memcpy(data_ptr, img.data(), num_pixels * sizeof(float));
}


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
