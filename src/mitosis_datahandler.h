#ifndef MITOSIS_DATAHANDLER_H_
#define MITOSIS_DATAHANDLER_H_
#include "hdf5_datahandler.h"
#include <mutex>

class PosNegHDF5DataHandler: public HDF5DataHandler {
 public:
  PosNegHDF5DataHandler(const config::DatasetConfig& config);
  ~PosNegHDF5DataHandler();
  virtual void GetBatch(vector<Layer*>& data_layers);

 protected:
  virtual void LoadMetaDataFromDisk();
  virtual void LoadFromDisk(int data_id);
  virtual void LoadFromDisk(int data_id, bool first_time);
  void SetupShuffler(int data_id, int dataset_size);
  void Shuffle(int data_id);
  void DiskAccess(int data_id);
  void WaitForPreload(int data_id);
  void StartPreload(int data_id);
  
  Matrix rand_perm_indices_[2];
  int pos_start_, neg_start_;
  const float pos_frac_;
  HDF5Iterator* it_[2];
  unsigned char* buf_[2];
  thread* preload_thread_[2];
  const bool use_multithreading_;
  bool fits_on_gpu_[2];
  mutex disk_access_mutex_;
};
#endif
