#ifndef DATAHANDLER_H_
#define DATAHANDLER_H_
#include "layer.h"
#include "image_iterators.h"
#include <random>
#include <thread>
class DataIterator;
class DataHandler {
 public:
  DataHandler(const config::DatasetConfig& config);
  virtual ~DataHandler();

  void GetBatch(vector<Layer*>& data_layers);
  int GetBatchSize() const { return batch_size_; }
  int GetDataSetSize() const { return dataset_size_; }
  void Seek(int row);
  void Preprocess(Matrix& input, Matrix& output);
  void Sync();

 protected:
  void SetupShuffler();
  void ShuffleIndices();
  void LoadChunk(DataIterator& it, Matrix& mat);
  void LoadChunk(DataIterator& it, Matrix& mat, vector<int>& random_rows);

  void PipelinedDiskAccess();
  void DiskAccess();
  void StartPreload();
  void WaitForPreload();

  default_random_engine generator_;
  uniform_int_distribution<int> * distribution_;
  map<string, DataIterator*> data_it_;
  map<string, Matrix> data_;
  vector<string> layer_names_;
  thread* preload_thread_;
  Matrix rand_perm_indices_;
  int batch_size_, chunk_size_, max_reuse_count_, reuse_counter_,
      random_access_chunk_size_, dataset_size_, start_;
  bool restart_, nothing_on_gpu_, fits_on_gpu_;
  const bool pipeline_loads_, randomize_cpu_, randomize_gpu_;
};

class DataIterator {
 public:
  DataIterator(const config::DataStreamConfig& config);
  virtual ~DataIterator() {};
  virtual void GetNext(float* data_out) = 0;
  virtual void Seek(int row);
  void Preprocess(Matrix& m);
  void AddNoise(Matrix& input, Matrix& output);
  int GetDims() const;
  int GetDataSetSize() const;
  void AddPCANoise(Matrix& m);
  void SetJitterVariables(int max_offset);
  void Jitter(Matrix& source, Matrix& dest);
  static DataIterator* ChooseDataIterator(const config::DataStreamConfig& config);

 protected:
  void LoadMeans(const string& data_file);

  int num_dims_, dataset_size_, row_;
  Matrix mean_, std_, pca_noise1_, pca_noise2_, eig_values_, eig_vectors_,
         width_offset_, height_offset_, flip_bit_;
  const string file_pattern_;
  const int num_colors_, gpu_id_;
  const bool translate_, flip_, normalize_, pixelwise_normalize_, add_pca_noise_;
  const float pca_noise_stddev_; 
};

class DummyDataIterator : public DataIterator {
 public:
  DummyDataIterator(const config::DataStreamConfig& config);
  void GetNext(float* data_out);
};

template <typename T>
class HDF5DataIterator : public DataIterator {
 public:
  HDF5DataIterator(const config::DataStreamConfig& config);
  ~HDF5DataIterator();
  void GetNext(float* data_out);
  void GetNext(float* data_out, const int row);

 protected:
  hid_t file_, dataset_, dapl_id_, m_dataspace_, type_;
  hsize_t start_[2], count_[2];
  T* buf_;
};

class ImageDataIterator : public DataIterator {
 public:
  ImageDataIterator(const config::DataStreamConfig& config);
  ~ImageDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Seek(int row);
 
 protected:
  RawImageFileIterator<unsigned char> *it_;
  unsigned char* buf_;
  const int raw_image_size_, image_size_;
};

class SlidingWindowDataIterator : public DataIterator {
 public:
  SlidingWindowDataIterator(const config::DataStreamConfig& config);
  ~SlidingWindowDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Seek(int row);
 
 protected:
  SlidingWindowIterator<unsigned char> *it_;
  unsigned char* buf_;
  vector<string> file_names_;
  const int stride_, raw_image_size_, image_size_;
  int file_id_;
};

class TextDataIterator : public DataIterator {
 public:
  TextDataIterator(const config::DataStreamConfig& config);
  ~TextDataIterator();
  virtual void GetNext(float* data_out);

 protected:
  float* data_;
};

class DataWriter {
 public:
  DataWriter(const string& output_file, const int dataset_size);
  ~DataWriter();
  virtual void AddStream(const string& name, const int numdims);
  virtual void Write(Matrix& mat, const int data_id, const int rows);

 private:
  const string output_file_;
  const int dataset_size_;
  vector<int> numdims_;
  vector<hid_t> dataset_handle_, dataspace_handle_;
  vector<int> current_row_;
  hid_t file_;
  int num_streams_;
};

class AveragedDataWriter : public DataWriter {
 public:
  AveragedDataWriter(const string& output_file, const int dataset_size,
                     const int avg_after, int max_batchsize);
  ~AveragedDataWriter();
  virtual void AddStream(const string& name, const int numdims);
  virtual void Write(Matrix& mat, const int data_id, const int rows);
 private:
  const int avg_after_, max_batchsize_;
  vector<Matrix*> buf_;
  vector<int> counter_;
};

class SequentialAveragedDataWriter : public DataWriter {
 public:
  SequentialAveragedDataWriter(const string& output_file, const int dataset_size,
                               const int avg_after);
  ~SequentialAveragedDataWriter();
  virtual void AddStream(const string& name, const int numdims);
  virtual void Write(Matrix& mat, const int data_id, const int rows);

 private:
  const int avg_after_, dataset_size_;
  vector<Matrix*> buf_;
  int consumed_, num_rows_written_;
};

#endif

