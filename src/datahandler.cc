#include "datahandler.h"
#include "image_iterators.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

DataHandler::DataHandler(const config::DatasetConfig& config) :
  preload_thread_(NULL),
  batch_size_(config.batch_size()),
  chunk_size_(config.chunk_size()),
  max_reuse_count_(config.max_reuse_count()),
  reuse_counter_(0),
  random_access_chunk_size_(config.random_access_chunk_size()),
  dataset_size_(-1),
  start_(0),
  restart_(true),
  nothing_on_gpu_(true),
  fits_on_gpu_(false),
  pipeline_loads_(config.pipeline_loads()),
  randomize_cpu_(config.randomize_cpu()),
  randomize_gpu_(config.randomize_gpu()) {

  // Create data streams.
  for (const config::DataStreamConfig& dsc:config.data_config()) {
    const string& layer_name = dsc.layer_name();
    layer_names_.push_back(layer_name);
    data_it_[layer_name] = DataIterator::ChooseDataIterator(dsc);
    int dataset_size = data_it_[layer_name]->GetDataSetSize();
    if (dataset_size_ == -1) {
      dataset_size_ = dataset_size;
    } else {
      if (dataset_size != dataset_size_) {
        cerr << "All data streams must have the same size." << endl;
        exit(1);
      }
    }
  }
  if (chunk_size_ <= 0 || chunk_size_ > dataset_size_) {
    chunk_size_ = dataset_size_;
    fits_on_gpu_ = true;
  }
  // GPU shuffler.
  if (randomize_gpu_) {
    SetupShuffler();
  }
  if (randomize_cpu_) {
    distribution_ = new uniform_int_distribution<int>(0, dataset_size_ - 1);
  }
  Seek(0);
  for (const string& layer_name : layer_names_) {
    DataIterator* it = data_it_[layer_name];
    int num_dims = it->GetDims();
    Matrix::SetDevice(it->GetGPUId());
    Matrix& data = data_[layer_name];
    data.AllocateGPUMemory(num_dims, chunk_size_);
  }
}

DataHandler::~DataHandler() {
  for (auto it : data_it_) {
    delete it.second;
  }
}

void DataHandler::SetupShuffler() {
  rand_perm_indices_.AllocateGPUMemory(1, chunk_size_);
  float* cpu_rand_perm_indices = rand_perm_indices_.GetHostData();
  for (int i = 0; i < chunk_size_; i++) {
    cpu_rand_perm_indices[i] = i;
  }
  rand_perm_indices_.CopyToDevice();
}

void DataHandler::Seek(int row) {
  Sync();
  start_ = row;
  reuse_counter_ = 0;
  restart_ = true;
  for (auto it : data_it_) {
    it.second->Seek(row);
  }
}

void DataHandler::Sync() {
  if (pipeline_loads_) WaitForPreload();
}

void DataHandler::ShuffleIndices() {
  float* cpu_rand_perm_indices = rand_perm_indices_.GetHostData();
  const int dataset_size = rand_perm_indices_.GetCols();
  random_shuffle(cpu_rand_perm_indices, cpu_rand_perm_indices + dataset_size);
  rand_perm_indices_.CopyToDevice();
}

void DataHandler::GetBatch(vector<Layer*>& data_layers) {
  int end = start_ + batch_size_;
  if (end > chunk_size_ || restart_) {
    if (reuse_counter_ < max_reuse_count_ && !restart_) {
      reuse_counter_++;
    } else if (nothing_on_gpu_ || !fits_on_gpu_) {
      if (restart_ && pipeline_loads_) StartPreload();
      nothing_on_gpu_ = false;
      reuse_counter_ = 0;
      PipelinedDiskAccess();
    }
    restart_ = false;
    if (randomize_gpu_) {
      ShuffleIndices();
      for (Layer* l : data_layers) {
        const string& layer_name = l->GetName();
        Matrix& data = data_[layer_name];
        shuffleColumns(data.GetMat(), rand_perm_indices_.GetMat());
      }
    }
    start_ = 0;
    end = batch_size_;
  }
  for (Layer* l : data_layers) {
    const string& layer_name = l->GetName();
    DataIterator* it = data_it_[layer_name];
    if (it == NULL) continue;
    Matrix::SetDevice(it->GetGPUId());
    Matrix data_slice;
    data_[layer_name].GetSlice(data_slice, start_, end);
    Matrix& dest = l->IsInput() ? l->GetState() : l->GetData();
    
    // Add noise, if asked for, and copy to dest.
    it->AddNoise(data_slice, dest);
    dest.SetReady();
  }
  start_ = end;
}

void DataHandler::PipelinedDiskAccess() {
  if (pipeline_loads_) {
    WaitForPreload();
  } else {
    DiskAccess();
  }

  for (const string& layer_name: layer_names_) {
    Matrix& data = data_[layer_name];
    DataIterator* it = data_it_[layer_name];
    data.CopyToDevice();
    it->Preprocess(data);  // Does centering, normalization etc.
  }

  if (pipeline_loads_) {
    StartPreload();
  }
}

void DataHandler::StartPreload() {
  preload_thread_ = new thread(&DataHandler::DiskAccess, this);
}

void DataHandler::WaitForPreload() {
  if (preload_thread_ != NULL) {
    preload_thread_->join();
    delete preload_thread_;
    preload_thread_ = NULL;
  }
}

void DataHandler::DiskAccess() {
  vector<int> random_rows;
  if (randomize_cpu_) {
    for (int i = 0; i < chunk_size_; i += random_access_chunk_size_) {
      random_rows.push_back((*distribution_)(generator_));
    }
  }
  for (const string& layer_name: layer_names_) {
    DataIterator* it = data_it_[layer_name];
    Matrix& data = data_[layer_name];
    if (randomize_cpu_) {
      LoadChunk(*it, data, random_rows);
    } else {
      LoadChunk(*it, data);
    }
  }
}

void DataHandler::LoadChunk(DataIterator& it, Matrix& mat) {
  float* data_ptr = mat.GetHostData();
  int num_dims = it.GetDims();
  for (int i = 0; i < chunk_size_; i++) {
    it.GetNext(data_ptr);
    data_ptr += num_dims;
  }
}

void DataHandler::LoadChunk(DataIterator& it, Matrix& mat, vector<int>& random_rows) {
  float* data_ptr = mat.GetHostData();
  int num_dims = it.GetDims();
  int j = 0;
  for (int i = 0; i < chunk_size_; i++) {
    if (i % random_access_chunk_size_ == 0) {
      it.Seek(random_rows[j++]);
    } 
    it.GetNext(data_ptr);
    data_ptr += num_dims;
  }
}

DataIterator* DataIterator::ChooseDataIterator(const config::DataStreamConfig& config) {
  DataIterator* it = NULL;
  hid_t file, dataset, datatype;
  int size;
  switch (config.data_type()) {
    case config::DataStreamConfig::DUMMY:
      it = new DummyDataIterator(config);
      break;
    case config::DataStreamConfig::HDF5:
      file = H5Fopen(config.file_pattern().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      dataset = H5Dopen(file, config.dataset_name().c_str(), H5P_DEFAULT);
      datatype = H5Dget_type(dataset);
      size = H5Tget_size(datatype);
      if (H5Tget_class(datatype) == H5T_INTEGER) { // integer
        if (H5Tget_sign(datatype) == H5T_SGN_2) {  // signed integers.
          switch (size) {
            case 1: it = new HDF5DataIterator<char>(config); break;
            case 4: it = new HDF5DataIterator<int>(config); break;
            case 8: it = new HDF5DataIterator<long>(config); break;
          }
        } else {  // unsigned integers.
          switch (size) {
            case 1: it = new HDF5DataIterator<unsigned char>(config); break;
            case 4: it = new HDF5DataIterator<unsigned int>(config); break;
            case 8: it = new HDF5DataIterator<unsigned long>(config); break;
          }
        }
      } else {  // floating-point.
        switch (size) {
          case 4: it = new HDF5DataIterator<float>(config); break;
          case 8: it = new HDF5DataIterator<double>(config); break;
        }
      }
      H5Tclose(datatype);
      H5Dclose(dataset);
      H5Fclose(file);
      break;
    case config::DataStreamConfig::IMAGE_RAW:
      it = new ImageDataIterator(config);
      break;
    case config::DataStreamConfig::SLIDING_WINDOW:
      it = new SlidingWindowDataIterator(config);
      break;
    case config::DataStreamConfig::TXT:
      it = new TextDataIterator(config);
      break;
    default:
      cerr << "Unknown data type " << config.data_type() << endl;
      exit(1);
  }
  return it;
}

DataIterator::DataIterator(const config::DataStreamConfig& config):
  num_dims_(0), dataset_size_(0), row_(0),
  file_pattern_(config.file_pattern()),
  num_colors_(config.num_colors()),
  gpu_id_(config.gpu_id()),
  translate_(config.can_translate()),
  flip_(config.can_flip()),
  normalize_(config.normalize() || config.pixelwise_normalize()),
  pixelwise_normalize_(config.pixelwise_normalize()),
  add_pca_noise_(config.pca_noise_stddev() > 0),
  pca_noise_stddev_(config.pca_noise_stddev()) { 

}

void DataIterator::LoadMeans(const string& data_file) {
  hid_t file = H5Fopen(data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (pixelwise_normalize_) {
    hid_t file = H5Fopen(data_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    Matrix pixel_mean, pixel_std;
    pixel_mean.AllocateAndReadHDF5(file, "pixel_mean");
    pixel_std.AllocateAndReadHDF5(file, "pixel_std");
    pixel_mean.Reshape(1, -1);
    pixel_std.Reshape(1, -1);
    int num_channels = pixel_mean.GetCols();
    int num_pixels = num_dims_ / num_channels;
    mean_.AllocateGPUMemory(num_pixels, num_channels);
    std_.AllocateGPUMemory(num_pixels, num_channels);

    add_row_vec(mean_.GetMat(), pixel_mean.GetMat(), mean_.GetMat());
    add_row_vec(std_.GetMat(), pixel_std.GetMat(), std_.GetMat());

    mean_.Reshape(-1, 1);
    std_.Reshape(-1, 1);
    if (add_pca_noise_) {
      eig_values_.AllocateAndReadHDF5(file, "S");
      eig_vectors_.AllocateAndReadHDF5(file, "V");
    }
  }
  else {
    mean_.AllocateAndReadHDF5(file, "mean");
    std_.AllocateAndReadHDF5(file, "std");
  }
  H5Fclose(file);
}

int DataIterator::GetDims() const {
  return num_dims_;
}

int DataIterator::GetDataSetSize() const {
  return dataset_size_;
}

void DataIterator::Seek(int row) {
  row_ = row;
}

// m is on the GPU, stored so that different cases are far apart.
void DataIterator::Preprocess(Matrix& m) {
  if (normalize_) {
    cudamat* mat = m.GetMat();
    add_col_mult(mat, mean_.GetMat(), mat, -1);
    div_by_col_vec(mat, std_.GetMat(), mat);
  }
}

void DataIterator::AddPCANoise(Matrix& m) {
  if (pca_noise1_.GetRows() != m.GetRows()) {
    pca_noise1_.AllocateGPUMemory(m.GetRows(), eig_vectors_.GetRows());
    pca_noise2_.AllocateGPUMemory(m.GetRows(), eig_vectors_.GetCols());
  }
  pca_noise1_.FillWithRandn();
  cudamat* rand_mat = pca_noise1_.GetMat();
  cudamat* pca_noise_mat = pca_noise2_.GetMat();
  mult_by_row_vec(rand_mat, eig_values_.GetMat(), rand_mat);
  dot(rand_mat, eig_vectors_.GetMat(), pca_noise_mat, 0, 1);
  add_to_each_pixel(m.GetMat(), pca_noise_mat, m.GetMat(), pca_noise_stddev_);
}


void DataIterator::AddNoise(Matrix& input, Matrix& output) {
  if (flip_ || translate_ || (output.GetCols() <  input.GetRows())) {
    Jitter(input, output);
  } else {
    copy_transpose(input.GetMat(), output.GetMat());
  }
  if (add_pca_noise_) {
    AddPCANoise(output);
  }
}

void DataIterator::SetJitterVariables(int max_offset) {
  if (translate_) {  // Random jitter.
    width_offset_.FillWithRand();
    height_offset_.FillWithRand();
  } else {  // Take center patch.
    width_offset_.Set(0.5);
    height_offset_.Set(0.5);
  }
  if (flip_) {
    flip_bit_.FillWithRand();  // flip if > 0.5.
  } else {
    flip_bit_.Set(0);
  }
  cudamat *wo = width_offset_.GetMat(), *ho = height_offset_.GetMat();
  mult_by_scalar(wo, max_offset + 1, wo);  // Rounded down.
  mult_by_scalar(ho, max_offset + 1, ho);
}

void DataIterator::Jitter(Matrix& source, Matrix& dest) {
  int patch_size = (int)sqrt(dest.GetCols() / num_colors_);
  int image_size = (int)sqrt(source.GetRows() / num_colors_);
  int max_offset = image_size - patch_size;

  if (max_offset > 0 || flip_) {
    if (width_offset_.GetCols() != source.GetCols()) {
      int batch_size = source.GetCols();
      width_offset_.AllocateGPUMemory(1, batch_size);
      height_offset_.AllocateGPUMemory(1, batch_size);
      flip_bit_.AllocateGPUMemory(1, batch_size);
    }
    SetJitterVariables(max_offset);
    cudamat *wo = width_offset_.GetMat(), *ho = height_offset_.GetMat(),
            *f = flip_bit_.GetMat(), *src_mat = source.GetMat(),
            *dest_mat = dest.GetMat();

    // Extract shifted images.
    int err_code = extract_patches(src_mat, dest_mat, wo, ho, f, image_size,
                                   image_size, patch_size, patch_size);
    if (err_code != 0) {
      cerr << "Error extracting patches " << GetStringError(err_code) << endl;
      exit(1);
    }
  } else {
    copy_transpose(source.GetMat(), dest.GetMat());
  }
}
DummyDataIterator::DummyDataIterator(const config::DataStreamConfig& config):
  DataIterator(config) {
    num_dims_ = 100;
    dataset_size_ = 100000;
  if (normalize_) {
    LoadMeans(config.mean_file());
  }
}

void DummyDataIterator::GetNext(float* data_out) {
  for (int i = 0; i < num_dims_; i++) {
    data_out[i] = rand();
  }
}

template <typename T>
HDF5DataIterator<T>::HDF5DataIterator(const config::DataStreamConfig& config):
  DataIterator(config) {
  
  // Open the hdf5 file and dataset.
  file_ = H5Fopen(file_pattern_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  size_t cache_size = 1024 * 1024 * 1024;
  dapl_id_ = H5Pcreate(H5P_DATASET_ACCESS);
  H5Pset_chunk_cache(dapl_id_, H5D_CHUNK_CACHE_NSLOTS_DEFAULT, cache_size, H5D_CHUNK_CACHE_W0_DEFAULT);
  dataset_ = H5Dopen(file_, config.dataset_name().c_str(), dapl_id_);
  hid_t dataspace = H5Dget_space(dataset_);
  hsize_t dims_out[2];
  const int ndims = H5Sget_simple_extent_ndims(dataspace);
  H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
  dataset_size_ = dims_out[0];
  num_dims_ = (ndims == 1) ? 1: dims_out[1];
  start_[0] = 0;
  start_[1] = 0;
  count_[0] = 1;
  count_[1] = num_dims_;
  H5Sclose(dataspace);
  m_dataspace_ = H5Screate_simple(2, count_, NULL);
  type_ = H5Dget_type(dataset_);

  buf_ = new T[num_dims_];

  /*
   * Check if gzip compression is available and can be used for both
   * compression and decompression.
   */
  htri_t avail = H5Zfilter_avail(H5Z_FILTER_DEFLATE);
  if (!avail) {
    printf ("gzip filter not available.\n");
  }
  unsigned int filter_info;
  H5Zget_filter_info (H5Z_FILTER_DEFLATE, &filter_info);
  if ( !(filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED) ||
      !(filter_info & H5Z_FILTER_CONFIG_DECODE_ENABLED) ) {
    printf ("gzip filter not available for encoding and decoding.\n");
  }
  if (normalize_) {
    LoadMeans(config.mean_file());
  }
}

template <typename T>
HDF5DataIterator<T>::~HDF5DataIterator() {
  delete buf_;
  H5Tclose(type_);
  H5Sclose(m_dataspace_);
  H5Pclose(dapl_id_);
  H5Dclose(dataset_);
  H5Fclose(file_);
}

template <typename T>
void HDF5DataIterator<T>::GetNext(float* data_ptr, const int row) {
  start_[0] = row;
  hid_t f_dataspace = H5Dget_space(dataset_);
  H5Sselect_hyperslab(f_dataspace, H5S_SELECT_SET, start_, NULL, count_, NULL);
  H5Dread(dataset_, type_, m_dataspace_, f_dataspace, H5P_DEFAULT, buf_);
  H5Sclose(f_dataspace);

  // Copy and type-cast from buf_ to data_ptr.
  for (int i = 0; i < num_dims_; i++) {
    data_ptr[i] = static_cast<float>(buf_[i]);
  }
}

template <typename T>
void HDF5DataIterator<T>::GetNext(float* data_ptr) {
  GetNext(data_ptr, row_);
  row_++;
  if (row_ == dataset_size_) row_ = 0;
}

ImageDataIterator::ImageDataIterator(const config::DataStreamConfig& config):
  DataIterator(config),
  raw_image_size_(config.raw_image_size()),
  image_size_(config.image_size()) {
  it_ = new RawImageFileIterator<unsigned char>(
      file_pattern_, image_size_, raw_image_size_, flip_, translate_);

  dataset_size_ = it_->GetDataSetSize();
  num_dims_ = image_size_ * image_size_ * num_colors_;
  buf_ = new unsigned char[num_dims_];
  if (normalize_) {
    LoadMeans(config.mean_file());
  }
}

ImageDataIterator::~ImageDataIterator() {
  delete buf_;
  delete it_;
}

void ImageDataIterator::Seek(int row) {
  it_->Seek(row);
}

void ImageDataIterator::GetNext(float* data_out) {
  it_->GetNext(buf_);
  for (int i = 0; i < image_size_ * image_size_ * num_colors_; i++) {
    data_out[i] = static_cast<float>(buf_[i]);
  }
}

SlidingWindowDataIterator::SlidingWindowDataIterator(
    const config::DataStreamConfig& config):
  DataIterator(config), stride_(config.stride()),
  raw_image_size_(config.raw_image_size()),
  image_size_(config.image_size()),
  file_id_(0) {
  num_dims_ = image_size_ * image_size_ * num_colors_;
  it_ = new SlidingWindowIterator<unsigned char>(image_size_, stride_);
  buf_ = new unsigned char[num_dims_];
  ReadLines(file_pattern_, file_names_);
  dataset_size_ = file_names_.size() * it_->GetNumWindows();
  if (normalize_) {
    LoadMeans(config.mean_file());
  }
}

SlidingWindowDataIterator::~SlidingWindowDataIterator() {
  delete buf_;
  delete it_;
}

void SlidingWindowDataIterator::Seek(int row) {
  file_id_ = row;
}

void SlidingWindowDataIterator::GetNext(float* data_out) {
  if (it_->Done()) {
    it_->SetImage(file_names_[file_id_++]);
    if (file_id_ == file_names_.size()) file_id_ = 0;
  }
  it_->GetNext(buf_);
  for (int i = 0; i < image_size_ * image_size_ * num_colors_; i++) {
    data_out[i] = static_cast<float>(buf_[i]);
  }
}

TextDataIterator::TextDataIterator(const config::DataStreamConfig& config):
  DataIterator(config) {
  ifstream f(file_pattern_, ios::in);
  string line;
  getline(f, line);
  istringstream iss(line);
  vector<string> tokens{istream_iterator<string>{iss},
                        istream_iterator<string>{}};
  num_dims_ = tokens.size();
  dataset_size_ = 1;
  while (getline(f, line)) dataset_size_++;
  data_ = new float[num_dims_ * dataset_size_];
  f.close();
  ifstream g(file_pattern_, ios::in);
  for (int i = 0; i < dataset_size_; i++) {
    g >> data_[i];
  }
  g.close();
  if (normalize_) {
    LoadMeans(config.mean_file());
  }
}

TextDataIterator::~TextDataIterator() {
  delete data_;
}

void TextDataIterator::GetNext(float* data_out) {
  memcpy(data_out, data_ + num_dims_ * row_, sizeof(float) * num_dims_);
  row_++;
  if (row_ == dataset_size_) row_ = 0;
}

DataWriter::DataWriter(const string& output_file, const int dataset_size) :
  output_file_(output_file), dataset_size_(dataset_size), num_streams_(0) {
  file_ = H5Fcreate(output_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                    H5P_DEFAULT);
}

DataWriter::~DataWriter(){
  //for(hid_t d : dataspace_handle_) H5Sclose(d);
  //for(hid_t d : dataset_handle_) H5Dclose(d);
  H5Fclose(file_);
}

void DataWriter::AddStream(const string& name, const int numdims) {
  hsize_t dimsf[2];
  dimsf[0] = dataset_size_;
  dimsf[1] = numdims;
  cout << "Adding Dataspace " << name << " of size " << dimsf[0]
       << " " << dimsf[1] << endl;
  numdims_.push_back(numdims);
  current_row_.push_back(0);
  dataspace_handle_.push_back(H5Screate_simple(2, dimsf, NULL));
  dataset_handle_.push_back(H5Dcreate(file_, name.c_str(), H5T_NATIVE_FLOAT,
                            dataspace_handle_[num_streams_], H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT));
  num_streams_++;
}

// This may not necessarily write to disk, but hold it in a buffer.
void DataWriter::Write(Matrix& mat, const int data_id, const int rows) {
  mat.CopyToHost();
  hsize_t dimsf[2], start[2];
  dimsf[0] = rows;
  dimsf[1] = numdims_[data_id];
  start[0] = current_row_[data_id];
  start[1] = 0;
  hid_t mem_dataspace = H5Screate_simple(2, dimsf, NULL);
  H5Sselect_none(dataspace_handle_[data_id]);
  H5Sselect_hyperslab(dataspace_handle_[data_id], H5S_SELECT_SET, start, NULL,
                      dimsf, NULL);
  H5Dwrite(dataset_handle_[data_id], H5T_NATIVE_FLOAT, mem_dataspace,
           dataspace_handle_[data_id], H5P_DEFAULT, mat.GetHostData());
  H5Sclose(mem_dataspace);
  current_row_[data_id] += rows;
}

AveragedDataWriter::AveragedDataWriter(
    const string& output_file, const int dataset_size, const int avg_after,
    const int max_batchsize):
  DataWriter(output_file, dataset_size), avg_after_(avg_after),
  max_batchsize_(max_batchsize) {
  } 

AveragedDataWriter::~AveragedDataWriter() {
  //for(Matrix* buf: buf_) delete buf;
}

void AveragedDataWriter::AddStream(const string& name, const int numdims) {
  DataWriter::AddStream(name, numdims);
  Matrix* mat = new Matrix();
  mat->AllocateGPUMemory(numdims, max_batchsize_);
  buf_.push_back(mat);
  mat->Set(0);
  counter_.push_back(0);
}

void AveragedDataWriter::Write(Matrix& mat, const int data_id, const int rows) {
  Matrix* buf = buf_[data_id];
  buf->Add(mat);
  if(++counter_[data_id] == avg_after_) {
    divide_by_scalar(buf->GetMat(), avg_after_, buf->GetMat());
    DataWriter::Write(*buf, data_id, rows);
    buf->Set(0);
    counter_[data_id] = 0;
  }
}

SequentialAveragedDataWriter::SequentialAveragedDataWriter(
    const string& output_file, const int dataset_size, const int avg_after) :
  DataWriter(output_file, dataset_size), avg_after_(avg_after),
  dataset_size_(dataset_size), consumed_(0), num_rows_written_(0) {
  }

SequentialAveragedDataWriter::~SequentialAveragedDataWriter() {
  //for(Matrix* buf: buf_) delete buf;
}

void SequentialAveragedDataWriter::AddStream(const string& name, const int numdims) {
  DataWriter::AddStream(name, numdims);
  Matrix* mat = new Matrix();
  mat->AllocateGPUMemory(numdims, 1);
  buf_.push_back(mat);
  mat->Set(0);
}

void SequentialAveragedDataWriter::Write(Matrix& mat, const int data_id, const int rows) {
  Matrix* buf = buf_[data_id];
  int numcases = mat.GetCols();
  int end = 0, start = 0;

  while(start < numcases && num_rows_written_ < dataset_size_) {
    cudamat slice;
    end = start + avg_after_ - consumed_;
    if (end > numcases) end = numcases;
    int avg_over = end - start;
    Matrix ones;
    Matrix::GetOnes(avg_over, 1, ones);
    get_slice(mat.GetMat(), &slice, start, end);
    dot(&slice, ones.GetMat(), buf->GetMat(), 1, 1.0 / avg_after_);
    consumed_ += avg_over;
    if (consumed_ == avg_after_) {
      DataWriter::Write(*buf, data_id, 1);
      num_rows_written_++;
      buf->Set(0);
      consumed_ = 0;
    }
    start = end;
  }
}
