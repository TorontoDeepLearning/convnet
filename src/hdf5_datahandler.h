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
  void Shuffle();
  vector<Matrix> data_;
  vector<string> dataset_names_;
  const string file_pattern_;
  int start_;
};



class HDF5DataHandler : public DataHandler {
 public:
  HDF5DataHandler(const config::DatasetConfig& config);
  virtual void GetBatch(vector<Layer*>& data_layers);

 protected:
  virtual void LoadFromDisk();
  virtual void LoadMetaDataFromDisk();
  virtual void LoadMeansFromDisk();
  void Shuffle();
  void Jitter(Matrix& source, int start, int end, Matrix& dest);
  void SetupJitter(int batch_size);
  virtual void SetJitterVariables(int max_offset);
  vector<Matrix> data_;
  Matrix mean_, std_, width_offset_, height_offset_, flip_;
  const string file_pattern_;
  int start_;
  vector<string> dataset_names_;
  const bool can_translate_, can_flip_, normalize_, pixelwise_normalize_;
  const int image_size_;
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
  ~ImageNetCLSDataHandler();
  virtual void GetBatch(vector<Layer*>& data_layers);
  virtual void Seek(int location);

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
