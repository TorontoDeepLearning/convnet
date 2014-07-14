#include "datawriter.h"

DataWriter::DataWriter(const config::FeatureExtractorConfig config) :
  dataset_size_(0) {
  file_ = H5Fcreate(config.output_file().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                    H5P_DEFAULT);
  for (const config::FeatureStreamConfig& feature : config.feature()) {
    stream& s = streams_[feature.layer()];
    s.average_batches = feature.average_batches();
    s.average_online = feature.average_online();
    s.counter = 0;
    s.current_row = 0;
  }
}

DataWriter::~DataWriter(){
  for(auto it : streams_) {
    H5Sclose(it.second.dataspace);
    H5Dclose(it.second.dataset);
  }
  H5Fclose(file_);
}

void DataWriter::SetNumDims(const string& name, const int num_dims) {
  stream& s = streams_[name];
  s.num_dims = num_dims;
  hsize_t dimsf[2];
  dimsf[0] = dataset_size_/s.average_batches;
  dimsf[1] = num_dims;
  cout << "Adding Dataspace " << name << " of size " << dimsf[0]
       << " " << dimsf[1] << endl;
  s.dataspace = H5Screate_simple(2, dimsf, NULL);
  s.dataset = H5Dcreate(file_, name.c_str(), H5T_NATIVE_FLOAT, s.dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

// This may not necessarily write to disk, but hold it in a buffer.
void DataWriter::WriteHDF5(Matrix& m, const string& dataset, int numcases) {
  Matrix m_t;
  Matrix::GetTemp(m.GetCols(), m.GetRows(), m_t);
  copy_transpose_big_matrix(m.GetMat(), m_t.GetMat());
  m_t.CopyToHost();
  stream& s = streams_[dataset];

  hsize_t dimsf[2], start[2];
  dimsf[0] = numcases;
  dimsf[1] = s.num_dims;
  start[0] = s.current_row;
  start[1] = 0;
  hid_t mem_dataspace = H5Screate_simple(2, dimsf, NULL);
  H5Sselect_none(s.dataspace);
  H5Sselect_hyperslab(s.dataspace, H5S_SELECT_SET, start, NULL, dimsf, NULL);
  H5Dwrite(s.dataset, H5T_NATIVE_FLOAT, mem_dataspace, s.dataspace,
           H5P_DEFAULT, m_t.GetHostData());
  H5Sclose(mem_dataspace);
  s.current_row += numcases;
}

void DataWriter::Write(vector<Layer*>& layers, int numcases) {
  for (Layer* l: layers) {
    Matrix& m = l->GetState();
    const string& dataset = l->GetName();
    stream& s = streams_[dataset];
    if(s.average_batches == 1) {
      WriteHDF5(m, dataset, numcases);
    } else {
      if (s.buf.GetNumEls() == 0) {
        s.buf.AllocateGPUMemory(m.GetRows(), m.GetCols());
        s.buf.Set(0);
      }
      s.buf.Add(m);
      if(++s.counter == s.average_batches) {
        divide_by_scalar(s.buf.GetMat(), s.average_batches, s.buf.GetMat());
        WriteHDF5(s.buf, dataset, numcases);
        s.buf.Set(0);
        s.counter = 0;
      }
    }
  }
}
/*
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
*/
