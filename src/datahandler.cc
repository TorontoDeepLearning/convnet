#include "datahandler.h"
#include "hdf5_datahandler.h"
#include "raw_image_datahandler.h"
#include <algorithm>
#include <iostream>

DataHandler::DataHandler(const config::DatasetConfig& config) :
  batch_size_(config.batch_size()),
  chunk_size_(config.chunk_size()),
  max_reuse_count_(config.max_reuse_count()),
  random_access_chunk_size_(config.random_access_chunk_size()),
  dataset_size_(0),
  row_(0),
  pipeline_loads_(config.pipeline_loads()),
  randomize_cpu_(config.randomize_cpu()),
  randomize_gpu_(config.randomize_gpu()) {

  for (const config::DataStreamConfig& dsc:config.data_config()) {
    it_[dsc.layer_name()] = ChooseDataIterator(dsc);
  }

}

void DataHandler::SetupShuffler(int dataset_size) {
}

void DataHandler::GetBatch(vector<Layer*>& data_layers) {

  for (Layer* l : data_layers) {
    const string& layer_name = l->GetLayerName();
    DataIterator* it = data_it_[layer_name];
    it.

  }

}

HDF5Iterator::HDF5Iterator(const string& file_name, const string& dataset_name) {
  // Open the hdf5 file and dataset.
  file_ = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  size_t cache_size = 1024 * 1024 * 1024;
  dapl_id_ = H5Pcreate(H5P_DATASET_ACCESS);
  H5Pset_chunk_cache(dapl_id_, H5D_CHUNK_CACHE_NSLOTS_DEFAULT, cache_size, H5D_CHUNK_CACHE_W0_DEFAULT);
  dataset_ = H5Dopen(file_, dataset_name.c_str(), dapl_id_);

  hid_t datatype = H5Dget_type(dataset_);
  atomic_size_ = H5Tget_size(datatype);
  is_int_type_ = H5Tget_class(datatype) == H5T_INTEGER;
  is_signed_type_ = is_int_type_ && (H5Tget_sign(datatype) == H5T_SGN_2);
  H5Tclose(datatype);

  hid_t dataspace = H5Dget_space(dataset_);
  hsize_t dims_out[2];
  H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
  dataset_size_ = dims_out[0];
  num_dims_ = dims_out[1];
  row_ = 0;
  start_[0] = 0;
  start_[1] = 0;
  count_[0] = 1;
  count_[1] = num_dims_;
  H5Sclose(dataspace);
  m_dataspace_ = H5Screate_simple(2, count_, NULL);
  type_ = H5Dget_type(dataset_);
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
}

HDF5Iterator::~HDF5Iterator() {
  H5Tclose(type_);
  H5Sclose(m_dataspace_);
  H5Pclose(dapl_id_);
  H5Dclose(dataset_);
  H5Fclose(file_);
}

void HDF5Iterator::GetNext(void* data_ptr, const int row) {
  start_[0] = row;
  hid_t f_dataspace = H5Dget_space(dataset_);
  H5Sselect_hyperslab(f_dataspace, H5S_SELECT_SET, start_, NULL, count_, NULL);
  H5Dread(dataset_, type_, m_dataspace_, f_dataspace, H5P_DEFAULT, data_ptr);
  H5Sclose(f_dataspace);
}

void HDF5Iterator::GetNext(void* data_ptr) {
  GetNext(data_ptr, row_);
  row_++;
  if (row_ == dataset_size_) row_ = 0;
}

HDF5MultiIterator::HDF5MultiIterator(const string& file_name,
                                     const vector<string>& dataset_names):
  num_it_(dataset_names.size()), row_(0) {
  for (const string& dataset_name: dataset_names) {
    it_.push_back(new HDF5Iterator(file_name, dataset_name));
  }
  dataset_size_ = it_[0]->GetDatasetSize();
  for(const HDF5Iterator* it: it_) {
    if (dataset_size_ != it->GetDatasetSize()) {
      cerr << "All datasets running in parallel must have same size." << endl;
      exit(1);
    }
  }
}

HDF5MultiIterator::~HDF5MultiIterator() {
  for (HDF5Iterator* it: it_) delete it;
}

void HDF5MultiIterator::GetNext(vector<void*>& data_ptr, int row) {
  if (data_ptr.size() != num_it_) {
    cerr << "Expecting " << num_it_ << " pointers. Received "
         << data_ptr.size();
    exit(1);
  }
  for (int i = 0; i < num_it_; i++) it_[i]->GetNext(data_ptr[i], row);
}

void HDF5MultiIterator::GetNext(vector<void*>& data_ptr) {
  GetNext(data_ptr, row_);
  row_++;
  if (row_ == dataset_size_) row_ = 0;
}

HDF5RandomAccessor::HDF5RandomAccessor(
    const string& file_name, const string& dataset_name, int chunk_size):
  HDF5Iterator(file_name, dataset_name),
  distribution_(new uniform_int_distribution<int>(0, GetDatasetSize() - 1)),
  chunk_size_(chunk_size), ind_(0) {}

HDF5RandomAccessor::~HDF5RandomAccessor() {
  delete distribution_;
}

void HDF5RandomAccessor::GetNext(void* data_ptr) {
  if (ind_ == chunk_size_) {
    ind_ = 0;
    int random_row = (*distribution_)(generator_);
    Seek(random_row);  // Seek to a random place.
  }
  HDF5Iterator::GetNext(data_ptr);
  ind_++;
}

HDF5RandomMultiAccessor::HDF5RandomMultiAccessor(
    const string& file_name, const vector<string>& dataset_names, int chunk_size):
  HDF5MultiIterator(file_name, dataset_names),
  distribution_(new uniform_int_distribution<int>(0, GetDatasetSize() - 1)),
  chunk_size_(chunk_size), ind_(0) {}

HDF5RandomMultiAccessor::~HDF5RandomMultiAccessor() {
  delete distribution_;
}

void HDF5RandomMultiAccessor::GetNext(vector<void*>& data_ptr) {
  if (ind_ == chunk_size_) {
    ind_ = 0;
    int random_row = (*distribution_)(generator_);
    Seek(random_row);  // Seek to a random place.
  }
  HDF5MultiIterator::GetNext(data_ptr);
  ind_++;
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
