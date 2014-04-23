#include "hdf5_datahandler.h"
#include <algorithm>

SimpleHDF5DataHandler::SimpleHDF5DataHandler(const config::DatasetConfig& config):
  DataHandler(config), file_pattern_(config.file_pattern()), start_(0) {
  for (string name : config.dataset_name()) {
    dataset_names_.push_back(name);
  }
  LoadFromDisk();
}

void SimpleHDF5DataHandler::LoadFromDisk() {
  string data_file = base_dir_ + file_pattern_;
  hid_t file = H5Fopen(data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  data_.resize(dataset_names_.size());
  for (int i = 0; i < dataset_names_.size(); i++) {
    data_[i].AllocateAndReadHDF5(file, dataset_names_[i]);
    if (i == 0) dataset_size_ = data_[i].GetCols();
    if (i > 0 && dataset_size_ != data_[i].GetCols()) {
      cerr << "Error: All datasets must have the same number of rows." << endl;
      exit(1);
    }
  }
  if (randomize_gpu_) {
    SetupShuffler(dataset_size_);
    Shuffle();
  }
  H5Fclose(file);
}

void SimpleHDF5DataHandler::Shuffle() {
  float* cpu_rand_perm_indices = rand_perm_indices_.GetHostData();
  const int dataset_size = rand_perm_indices_.GetCols();
  random_shuffle(cpu_rand_perm_indices, cpu_rand_perm_indices + dataset_size);
  rand_perm_indices_.CopyToDevice();
  for (Matrix& m : data_) {
    shuffleColumns(m.GetMat(), rand_perm_indices_.GetMat());
  }
}

void SimpleHDF5DataHandler::GetBatch(vector<Layer*>& data_layers) {
  Matrix::SyncAllDevices();
  Matrix::SetDevice(gpu_id_);
  if (start_ == 0) LoadFromDisk();
  int num_data_layers = data_layers.size();
  if (num_data_layers > data_.size()) {
    cerr << "Expecting " << data_.size() << " layers. Found " << num_data_layers << endl;
    exit(1);
  }
  int batch_size = data_layers[0]->GetState().GetRows();
  
  int end = start_ + batch_size;
  if (end > dataset_size_) {
    if (randomize_gpu_) Shuffle();
    start_ = 0;
    end = batch_size;
  }
  int i = 0;
  for (Layer* l : data_layers) {
    cudamat data_slice;
    Matrix& m = data_[i];
    get_slice(m.GetMat(), &data_slice, start_, end);
    copy_transpose(&data_slice, l->GetData().GetMat());
    l->GetData().SetReady();
    i += 1;
  }
  start_ = end;
}

HDF5DataHandler::HDF5DataHandler(const config::DatasetConfig& config):
  DataHandler(config), file_pattern_(config.file_pattern()), start_(0),
  can_translate_(config.can_translate()),
  can_flip_(config.can_flip()), normalize_(config.normalize_images()),
  pixelwise_normalize_(config.pixelwise_normalize()),
  image_size_(config.image_size()) {
  for (string name : config.dataset_name()) {
    dataset_names_.push_back(name);
  }
  if (normalize_ && mean_file_.empty()) {
    cerr << "Need mean_file to normalize data." << endl;
    exit(1);
  }
  LoadMetaDataFromDisk();
  if (!mean_file_.empty()) LoadMeansFromDisk();
  SetupJitter(batch_size_);
}

void HDF5DataHandler::GetBatch(vector<Layer*>& data_layers) {
  Matrix::SyncAllDevices();
  Matrix::SetDevice(gpu_id_);
  if (start_ == 0) LoadFromDisk();
  int num_data_layers = data_layers.size();
  if (num_data_layers > data_.size()) {
    cerr << "Expecting " << data_.size() << " layers. Found " << num_data_layers << endl;
    exit(1);
  }
  int batch_size = data_layers[0]->GetState().GetRows();
  
  int end = start_ + batch_size;
  if (end > dataset_size_) {
    if (randomize_gpu_) Shuffle();
    start_ = 0;
    end = batch_size;
  }
  int i = 0;
  for (Layer* l : data_layers) {
    if (l->IsInput()) {  // Randomly translate and flip image.
      Jitter(data_[i], start_, end, l->GetState());
      l->GetState().SetReady();
    } else {
      cudamat data_slice;
      Matrix& m = data_[i];
      get_slice(m.GetMat(), &data_slice, start_, end);
      copy_transpose(&data_slice, l->GetData().GetMat());
      l->GetData().SetReady();
    }
    i += 1;
  }
  start_ = end;
}

void HDF5DataHandler::LoadMetaDataFromDisk() {
  string data_file = base_dir_ + file_pattern_;
  hid_t file = H5Fopen(data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  int num_dims;
  ReadHDF5Shape(file, dataset_names_[0], &num_dims, &dataset_size_);
  H5Fclose(file);
}


// Loads data from disk to gpu.
void HDF5DataHandler::LoadFromDisk() {
  string data_file = base_dir_ + file_pattern_;
  hid_t file = H5Fopen(data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  data_.resize(dataset_names_.size());
  for (int i = 0; i < dataset_names_.size(); i++) {
    data_[i].AllocateAndReadHDF5(file, dataset_names_[i]);
    if (dataset_size_ != data_[i].GetCols()) {
      cerr << "Error: All datasets must have the same number of rows." << endl;
      exit(1);
    }
  }
  if (normalize_) {
    add_col_mult(data_[0].GetMat(), mean_.GetMat(), data_[0].GetMat(), -1);
    div_by_col_vec(data_[0].GetMat(), std_.GetMat(), data_[0].GetMat());
  }
  if (randomize_gpu_) {
    SetupShuffler(dataset_size_);
    Shuffle();
  }
  H5Fclose(file);
}

void HDF5DataHandler::LoadMeansFromDisk() {
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
  } else {
    mean_.AllocateAndReadHDF5(file, "mean");
    std_.AllocateAndReadHDF5(file, "std");
  }
  H5Fclose(file);
}

void HDF5DataHandler::Shuffle() {
  float* cpu_rand_perm_indices = rand_perm_indices_.GetHostData();
  const int dataset_size = rand_perm_indices_.GetCols();
  random_shuffle(cpu_rand_perm_indices, cpu_rand_perm_indices + dataset_size);
  rand_perm_indices_.CopyToDevice();
  for (Matrix& m : data_) {
    shuffleColumns(m.GetMat(), rand_perm_indices_.GetMat());
  }
}

void HDF5DataHandler::SetupJitter(int batch_size) {
  width_offset_.AllocateGPUMemory(1, batch_size);
  height_offset_.AllocateGPUMemory(1, batch_size);
  flip_.AllocateGPUMemory(1, batch_size);
  flip_.Set(0);
}

void HDF5DataHandler::SetJitterVariables(int max_offset) {
  if (can_translate_) {  // Random jitter.
    width_offset_.FillWithRand();
    height_offset_.FillWithRand();
  } else {  // Take center patch.
    width_offset_.Set(0.5);
    height_offset_.Set(0.5);
  }
  if (can_flip_) {
    flip_.FillWithRand();
  }
  cudamat *wo = width_offset_.GetMat(), *ho = height_offset_.GetMat();
  mult_by_scalar(wo, max_offset + 1, wo);  // Rounded down.
  mult_by_scalar(ho, max_offset + 1, ho);
}

void HDF5DataHandler::Jitter(Matrix& source, int start, int end, Matrix& dest) {
  cudamat data_slice;
  get_slice(source.GetMat(), &data_slice, start, end);

  int err_code = 0;
  int num_colors = source.GetRows() / (image_size_ * image_size_);
  int patch_size = (int)sqrt(dest.GetCols() / num_colors);
  int max_offset = image_size_ - patch_size;

  if (max_offset > 0 || can_flip_) {
    SetJitterVariables(max_offset);
    cudamat *wo = width_offset_.GetMat(), *ho = height_offset_.GetMat(),
            *f = flip_.GetMat(), *dest_mat = dest.GetMat();

    //cout << "Jittering " << max_offset << endl;
    // Extract shifted images.
    err_code = extract_patches(&data_slice, dest_mat, wo, ho, f, image_size_,
                               image_size_, patch_size, patch_size);
    if (err_code != 0) {
      cout << "Error extracting patches " << GetStringError(err_code) << endl;
      exit(1);
    }
  } else {
    copy_transpose(&data_slice, dest.GetMat());
  }
}

HDF5MultiplePositionDataHandler::HDF5MultiplePositionDataHandler(
    const config::DatasetConfig& config) : HDF5DataHandler(config), pos_id_(0) {
  num_positions_ = can_flip_ ? 10 : 5;
  real_dataset_size_ = dataset_size_;
  dataset_size_ *= num_positions_;
}

void HDF5MultiplePositionDataHandler::SetJitterVariables(int max_offset) {
  int wo = 0, ho = 0, flip = can_flip_ ? (pos_id_ % 2): 0;
  int id = can_flip_ ? (pos_id_ >> 1) : pos_id_;
  if (id == 0) {
    wo = max_offset/2;
    ho = max_offset/2;
  } else {
    wo = (id == 1 || id == 3) ? 0 : max_offset;
    ho = (id == 1 || id == 2) ? 0 : max_offset;
  }
  width_offset_.Set(wo);
  height_offset_.Set(ho);
  flip_.Set(flip);
}


void HDF5MultiplePositionDataHandler::GetBatch(vector<Layer*>& data_layers) {
  Matrix::SyncAllDevices();
  Matrix::SetDevice(gpu_id_);
  int num_data_layers = data_layers.size();
  if (num_data_layers > data_.size()) {
    cerr << "Expecting " << data_.size() << " layers. Found " << num_data_layers << endl;
    exit(1);
  }
  int batch_size = data_layers[0]->GetState().GetRows();
  if (start_ + batch_size > real_dataset_size_) {
    if (randomize_gpu_) Shuffle();
    start_ = 0;
  }
  int i = 0;
  for (Layer* l : data_layers) {
    if (l->IsInput()) {  // Randomly translate and flip image.
      Jitter(data_[i], start_, start_ + batch_size, l->GetState());
    } else {
      cudamat data_slice;
      Matrix& m = data_[i];
      get_slice(m.GetMat(), &data_slice, start_, start_ + batch_size);
      copy_transpose(&data_slice, l->GetData().GetMat());
    }
    i++;
  }
  pos_id_++;
  if (pos_id_ == num_positions_) {
    pos_id_ = 0;
    start_ += batch_size;
  }
}

ImageNetCLSDataHandler::ImageNetCLSDataHandler(const config::DatasetConfig& config):
  HDF5DataHandler(config), it_(NULL), chunk_size_(config.chunk_size()),
  max_reuse_count_(config.max_reuse_count()), reuse_counter_(0),
  preload_thread_(NULL), first_time_(true),
  use_multithreading_(config.pipeline_loads()) {
  string data_file = base_dir_ + file_pattern_;
  it_ = randomize_cpu_ ? 
          new HDF5RandomMultiAccessor(data_file, dataset_names_,
              config.random_access_chunk_size()) :
          new HDF5MultiIterator(data_file, dataset_names_);
  dataset_size_ = it_->GetDatasetSize();
  int max_dataset_size = config.max_dataset_size();
  if (max_dataset_size > 0 && max_dataset_size < dataset_size_) {
    dataset_size_ = max_dataset_size;
  }
  LoadFromDisk();
}

ImageNetCLSDataHandler::~ImageNetCLSDataHandler() {
  WaitForPreload();
  delete it_;
  for (void* ptr : buf_) if (ptr != NULL) delete reinterpret_cast<char*>(ptr);
  if (preload_thread_ != NULL) delete preload_thread_;
}

void ImageNetCLSDataHandler::LoadFromDisk() {
  data_.resize(dataset_names_.size());
  for (int i = 0; i < data_.size(); i++) {
    data_[i].AllocateGPUMemory(it_->GetDims(i), chunk_size_);
  }
  buf_.push_back(new unsigned char[it_->GetDims(0)]);
  buf_.push_back(new unsigned int[it_->GetDims(1)]);
  if (randomize_gpu_) SetupShuffler(chunk_size_);
}

void ImageNetCLSDataHandler::GetBatch(vector<Layer*>& data_layers) {
  Matrix::SyncAllDevices();
  Matrix::SetDevice(gpu_id_);
  int batch_size = data_layers[0]->GetState().GetRows();
  if (first_time_) {
    first_time_ = false;
    if (use_multithreading_) StartPreload();
    GetChunk();
    if (randomize_gpu_) Shuffle();
  }
  int num_data_layers = data_layers.size();
  if (num_data_layers > data_.size()) {
    cerr << "Expecting " << data_.size() << " layers. Found " << num_data_layers << endl;
    exit(1);
  }
  
  int end = start_ + batch_size;
  if (end > chunk_size_) {
    if (reuse_counter_ < max_reuse_count_) {
      reuse_counter_++;
    } else {
      reuse_counter_ = 0;
      GetChunk();  // Loads the next chunk in data_.
    }
    if (randomize_gpu_) Shuffle();
    start_ = 0;
    end = batch_size;
  }
  int i = 0;
  for (Layer* l : data_layers) {
    if (l->IsInput()) {  // Randomly translate and flip image.
      Jitter(data_[i], start_, end, l->GetState());
    } else {
      cudamat data_slice;
      Matrix& m = data_[i];
      get_slice(m.GetMat(), &data_slice, start_, end);
      copy_transpose(&data_slice, l->GetData().GetMat());
    }
    i += 1;
  }
  start_ = end;
}

void ImageNetCLSDataHandler::GetChunk() {
  if (use_multithreading_) {
    WaitForPreload();
  } else {
    DiskAccess();
  }

  for (int i = 0; i < data_.size(); i++) data_[i].CopyToDevice();

  // Normalize.
  add_col_mult(data_[0].GetMat(), mean_.GetMat(), data_[0].GetMat(), -1);
  div_by_col_vec(data_[0].GetMat(), std_.GetMat(), data_[0].GetMat());

  if (use_multithreading_) {  // Start loading the next chunk in a new thread. 
    StartPreload();
  }
}

void ImageNetCLSDataHandler::StartPreload() {
  preload_thread_ = new thread(&ImageNetCLSDataHandler::DiskAccess, this);
}

// Move chunk_size_ images from the hdf5 file to a matrix in main memory.
void ImageNetCLSDataHandler::DiskAccess() {
  float *data_ptr = data_[0].GetHostData(),
        *label_ptr = data_[1].GetHostData();
  const int ndims = it_->GetDims(0),
            ndims_labels = it_->GetDims(1);

  // The iterator manages chunks in a cache so it's ok to do serial access.
  unsigned char* data_buf = reinterpret_cast<unsigned char*> (buf_[0]);
  unsigned int* label_buf = reinterpret_cast<unsigned int*> (buf_[1]);
  for (int i = 0; i < chunk_size_; i++) {
    it_->GetNext(buf_);

    // Convert from uint8, uint32.. etc to float.
    // Unfortunately we have to do this because the GPU doesn't understand less
    // than 32 bit values. We could consider packing 24-bit things into 32 and
    // them spreading them out on the GPU just before doing computation with
    // them, but it would be painful.
    for (int j = 0; j < ndims; j++) {
      *(data_ptr++) = static_cast<float>(data_buf[j]);
    }
    for (int j = 0; j < ndims_labels; j++) {
      *(label_ptr++) = static_cast<float>(label_buf[j]);
    }
  }
}

void ImageNetCLSDataHandler::WaitForPreload() {
  if (preload_thread_ != NULL) {
    preload_thread_->join();
    delete preload_thread_;
    preload_thread_ = NULL;
  }
}

void ImageNetCLSDataHandler::Seek(int location) {
  WaitForPreload();
  it_->Seek(location);
  start_ = 0; 
}

ImageNetCLSMultiplePosDataHandler::ImageNetCLSMultiplePosDataHandler(
    const config::DatasetConfig& config) :
ImageNetCLSDataHandler(config), pos_id_(0) {
  num_positions_ = can_flip_ ? 10 : 5;
}

void ImageNetCLSMultiplePosDataHandler::SetJitterVariables(int max_offset) {
  int wo = 0, ho = 0, flip = can_flip_ ? (pos_id_ % 2): 0;
  int id = can_flip_ ? (pos_id_ >> 1) : pos_id_;
  if (id == 0) {
    wo = max_offset/2;
    ho = max_offset/2;
  } else {
    wo = (id == 1 || id == 3) ? 0 : max_offset;
    ho = (id == 1 || id == 2) ? 0 : max_offset;
  }
  width_offset_.Set(wo);
  height_offset_.Set(ho);
  flip_.Set(flip);
}

void ImageNetCLSMultiplePosDataHandler::GetBatch(vector<Layer*>& data_layers) {
  Matrix::SyncAllDevices();
  Matrix::SetDevice(gpu_id_);
  int batch_size = data_layers[0]->GetState().GetRows();
  if (first_time_) {
    first_time_ = false;
    if (use_multithreading_) StartPreload();
    GetChunk();
    if (randomize_gpu_) Shuffle();
  }
  int num_data_layers = data_layers.size();
  if (num_data_layers > data_.size()) {
    cerr << "Expecting " << data_.size() << " layers. Found " << num_data_layers << endl;
    exit(1);
  }
  
  int end = start_ + batch_size;
  if (end > chunk_size_ && pos_id_ == 0) {
    if (reuse_counter_ < max_reuse_count_) {
      reuse_counter_++;
    } else {
      reuse_counter_ = 0;
      GetChunk();  // Loads the next chunk in data_.
    }
    if (randomize_gpu_) Shuffle();
    start_ = 0;
    end = batch_size;
  }
  int i = 0;
  for (Layer* l : data_layers) {
    if (l->IsInput()) {  // Randomly translate and flip image.
      Jitter(data_[i], start_, end, l->GetState());
    } else {
      cudamat data_slice;
      Matrix& m = data_[i];
      get_slice(m.GetMat(), &data_slice, start_, end);
      copy_transpose(&data_slice, l->GetData().GetMat());
    }
    i += 1;
  }
  pos_id_++;
  if (pos_id_ == num_positions_) {
    pos_id_ = 0;
    start_ = end;
  }
}
