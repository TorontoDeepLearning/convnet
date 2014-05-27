#ifndef HDF5_DATAHANDLER_H_
#define HDF5_DATAHANDLER_H_
#include "datahandler.h"
#include <thread>
// Handles data stored in HDF5 file format.
class SimpleHDF5DataHandler : public DataHandler {
 public:
  SimpleHDF5DataHandler(const config::DatasetConfig& config);
  virtual void GetBatch(vector<Layer*>& data_layers);

 protected:
  virtual void LoadFromDisk();
  virtual void LoadMetaFromDisk();
  void Shuffle();
  vector<Matrix> data_;
  vector<string> dataset_names_;
  const string file_pattern_;
  int start_;
};

class BigSimpleHDF5DataHandler : public SimpleHDF5DataHandler {
 public:
  BigSimpleHDF5DataHandler(const config::DatasetConfig& config);
  virtual ~BigSimpleHDF5DataHandler();
  virtual void GetBatch(vector<Layer*>& data_layers);
  virtual void Seek(int location);

 protected:
  void GetChunk();
  virtual void LoadFromDisk();
  void StartPreload();
  virtual void DiskAccess();
  void WaitForPreload();

  HDF5MultiIterator *it_;
  int chunk_size_, max_reuse_count_;
  int reuse_counter_;
  vector<void*> buf_;
  thread* preload_thread_;
  bool first_time_;
  const bool use_multithreading_;
};

class HDF5DataHandler : public DataHandler {
 public:
  HDF5DataHandler(const config::DatasetConfig& config);
  virtual ~HDF5DataHandler() {}
  virtual void GetBatch(vector<Layer*>& data_layers);

 protected:
  virtual void LoadFromDisk();
  virtual void LoadMetaDataFromDisk();
  virtual void LoadMeansFromDisk();
  void Shuffle();
  void Jitter(Matrix& source, int start, int end, Matrix& dest);
  void AddPCANoise(Matrix& m);
  void SetupPCANoise(int batch_size, int num_colors);
  void SetupJitter(int batch_size);
  virtual void SetJitterVariables(int max_offset);
  vector<Matrix> data_;
  Matrix mean_, std_, eig_values_, eig_vectors_,
         width_offset_, height_offset_, flip_, pca_noise1_, pca_noise2_;
  const string file_pattern_;
  int start_, num_dims_;
  vector<string> dataset_names_;
  const bool can_translate_, can_flip_, normalize_, pixelwise_normalize_, add_pca_noise_;
  const int image_size_;
  const float pca_noise_stddev_;
};

class HDF5MultiplePositionDataHandler: public HDF5DataHandler {
 public:
  HDF5MultiplePositionDataHandler(const config::DatasetConfig& config);
  virtual void GetBatch(vector<Layer*>& data_layers);

 protected:
  virtual void SetJitterVariables(int max_offset);
  int pos_id_;
  int real_dataset_size_;
};

class ImageNetCLSDataHandler : public HDF5DataHandler {
 public:
  ImageNetCLSDataHandler(const config::DatasetConfig& config);
  virtual ~ImageNetCLSDataHandler();
  virtual void GetBatch(vector<Layer*>& data_layers);
  virtual void Seek(int location);
  virtual void Sync();

 protected:
  void GetChunk();
  virtual void LoadFromDisk();
  void StartPreload();
  virtual void DiskAccess();
  void WaitForPreload();

  HDF5MultiIterator *it_;
  const int chunk_size_, max_reuse_count_;
  int reuse_counter_;
  vector<void*> buf_;
  thread* preload_thread_;
  bool first_time_;
  const bool use_multithreading_;
};

class ImageNetCLSMultiplePosDataHandler : public ImageNetCLSDataHandler {
 public:
  ImageNetCLSMultiplePosDataHandler(const config::DatasetConfig& config);
  virtual void GetBatch(vector<Layer*>& data_layers);

 protected:
  virtual void SetJitterVariables(int max_offset);
  int pos_id_;
};
#endif
