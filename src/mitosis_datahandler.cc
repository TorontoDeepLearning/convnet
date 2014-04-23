#include "mitosis_datahandler.h"

PosNegHDF5DataHandler::PosNegHDF5DataHandler(const config::DatasetConfig& config):
  HDF5DataHandler(config), pos_start_(0), neg_start_(0),
  pos_frac_(config.pos_frac()),
  use_multithreading_(config.pipeline_loads()) {

  string data_file = base_dir_ + file_pattern_;
  cout << data_file << endl;
  data_.resize(dataset_names_.size());
  int ndims, numcases;
  dataset_size_ = 0;
  for (int i = 0; i < dataset_names_.size(); i++) {
    ReadHDF5ShapeFromFile(data_file, dataset_names_[i], &ndims, &numcases);
    cout << "Rows: " << ndims << " Cols " << numcases << endl;
    dataset_size_ += numcases;
    fits_on_gpu_[i] = numcases <= config.chunk_size();
    if (!fits_on_gpu_[i]) numcases = config.chunk_size();

    it_[i] = (randomize_cpu_ && !fits_on_gpu_[i]) ?
      new HDF5RandomAccessor(data_file, dataset_names_[i], config.random_access_chunk_size()) :
      new HDF5Iterator(data_file, dataset_names_[i]);
    
    buf_[i] = new unsigned char[ndims];
    for (int j = 0; j < ndims; j++) buf_[i][j] = 0;
    cout << "Data " << i << " numcases " << numcases << endl;
    data_[i].AllocateGPUMemory(ndims, numcases, "data buffer");
    SetupShuffler(i, numcases);
    preload_thread_[i] = NULL;
    if (use_multithreading_) StartPreload(i);
    LoadFromDisk(i, true);
  }
}

PosNegHDF5DataHandler::~PosNegHDF5DataHandler() {
  WaitForPreload(0);
  WaitForPreload(1);
  delete buf_[0];
  delete buf_[1];
  delete it_[0];
  delete it_[1];
}

void PosNegHDF5DataHandler::LoadMetaDataFromDisk() {}

void PosNegHDF5DataHandler::GetBatch(vector<Layer*>& data_layers) {
  Matrix::SyncAllDevices();
  Matrix::SetDevice(gpu_id_);
  int batch_size = data_layers[0]->GetState().GetRows();
 
  int num_pos = (int)(pos_frac_ * batch_size);
  int num_neg = batch_size - num_pos;
  if (pos_start_ + num_pos > data_[0].GetCols()) {
    LoadFromDisk(0);
    pos_start_ = 0;
  }
  if (neg_start_ + num_neg > data_[1].GetCols()) {
    LoadFromDisk(1);
    neg_start_ = 0;
  }

  Matrix temp;
  Matrix::GetTemp(data_[0].GetRows(), batch_size, temp);
  
  Matrix temp_pos_slice, temp_neg_slice, pos_slice, neg_slice;
  temp.GetSlice(temp_pos_slice, 0, num_pos);
  temp.GetSlice(temp_neg_slice, num_pos, batch_size);
  data_[0].GetSlice(pos_slice, pos_start_, pos_start_ + num_pos);
  data_[1].GetSlice(neg_slice, neg_start_, neg_start_ + num_neg);
  temp_pos_slice.Set(pos_slice);
  temp_neg_slice.Set(neg_slice);
  
  Matrix& state = data_layers[0]->GetState();
  Jitter(temp, 0, batch_size, state);
  state.SetReady();

  Matrix& labels = data_layers[1]->GetData();
  Matrix labels_pos, labels_neg;
  labels.Reshape(1, -1);
  labels.GetSlice(labels_pos, 0, num_pos);
  labels.GetSlice(labels_neg, num_pos, batch_size);
  labels.Reshape(-1, 1);
  labels_pos.Set(1);
  labels_neg.Set(0);
  labels.SetReady();

  pos_start_ += num_pos;
  neg_start_ += num_neg;
}


void PosNegHDF5DataHandler::LoadFromDisk(int data_id) {
  LoadFromDisk(data_id, false);
}

void PosNegHDF5DataHandler::LoadFromDisk(int data_id, bool first_time) {
  if (use_multithreading_) {
    WaitForPreload(data_id);
  } else {
    DiskAccess(data_id);
  }

  if (first_time || !fits_on_gpu_[data_id]) {
    data_[data_id].CopyToDevice();
    // Normalize.
    if (normalize_) {
      cudamat* mat = data_[data_id].GetMat();
      add_col_mult(mat, mean_.GetMat(), mat, -1);
      div_by_col_vec(mat, std_.GetMat(), mat);
    }
  }
  if (randomize_gpu_) Shuffle(data_id);

  if (use_multithreading_ && !fits_on_gpu_[data_id]) {  // Start loading the next chunk in a new thread. 
    StartPreload(data_id);
  }
}


// Loads data from disk to gpu.
void PosNegHDF5DataHandler::DiskAccess(int data_id) {
  unique_lock<mutex> lck(disk_access_mutex_);
  HDF5Iterator* it = it_[data_id];
  Matrix& mat = data_[data_id];
  float* data_ptr = mat.GetHostData();
  unsigned char* buf = buf_[data_id];

  for (int i = 0; i < mat.GetCols(); i++) {
    it->GetNext(buf);
    for (int j = 0; j < mat.GetRows(); j++) {
      *(data_ptr++) = static_cast<float>(buf[j]);
    }
  }
}

void PosNegHDF5DataHandler::Shuffle(int data_id) {
  Matrix& rand_perm_indices = rand_perm_indices_[data_id];
  float* cpu_rand_perm_indices = rand_perm_indices.GetHostData();
  const int dataset_size = rand_perm_indices.GetCols();
  random_shuffle(cpu_rand_perm_indices, cpu_rand_perm_indices + dataset_size);
  rand_perm_indices.CopyToDevice();
  shuffleColumns(data_[data_id].GetMat(), rand_perm_indices.GetMat());
}

void PosNegHDF5DataHandler::SetupShuffler(int data_id, int dataset_size) {
  rand_perm_indices_[data_id].AllocateGPUMemory(1, dataset_size);
  float* indices = rand_perm_indices_[data_id].GetHostData();
  for (int i = 0; i < dataset_size; i++) indices[i] = i;
  rand_perm_indices_[data_id].CopyToDevice();
}

void PosNegHDF5DataHandler::WaitForPreload(int data_id) {
  if (preload_thread_[data_id] != NULL) {
    preload_thread_[data_id]->join();
    delete preload_thread_[data_id];
    preload_thread_[data_id] = NULL;
  }
}

void PosNegHDF5DataHandler::StartPreload(int data_id) {
  preload_thread_[data_id] = new thread(&PosNegHDF5DataHandler::DiskAccess, this, data_id);
}


