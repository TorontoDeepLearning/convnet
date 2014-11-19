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
    s.consumed = 0;
  }
}

DataWriter::~DataWriter(){
  for(auto& it : streams_) {
    H5Dclose(it.second.dataset);
  }
  H5Fclose(file_);
}

void DataWriter::SetDataSetSize(int dataset_size) {
  dataset_size_ = dataset_size;
}

void DataWriter::SetNumDims(const string& name, const int num_dims) {
  stream& s = streams_[name];
  s.num_dims = num_dims;
  hsize_t dimsf[2];
  dimsf[0] = dataset_size_/(s.average_online * s.average_batches);
  dimsf[1] = num_dims;
  cout << "Adding Dataspace " << name << " of size " << dimsf[0]
       << " " << dimsf[1] << endl;
  hid_t dataspace = H5Screate_simple(2, dimsf, NULL);
  s.dataset = H5Dcreate(file_, name.c_str(), H5T_NATIVE_FLOAT, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (s.dataset < 0) {
    cerr << "Could not crete dataset for writing." << endl;
    exit(1);
  }
  H5Sclose(dataspace);
}

// This may not necessarily write to disk, but hold it in a buffer.
void DataWriter::WriteHDF5(Matrix& m, const string& dataset, int numcases, bool transpose) {
  float* data;
  Matrix m_t;
  if (transpose) {
    Matrix::GetTemp(m.GetCols(), m.GetRows(), m_t);
    m.CopyTransposeBig(m_t);
    m_t.CopyToHost();
    data = m_t.GetHostData();
  } else {
    m.CopyToHost();
    data = m.GetHostData();
  }
  stream& s = streams_[dataset];

  hsize_t dimsf[2], start[2];
  dimsf[0] = numcases;
  dimsf[1] = s.num_dims;
  start[0] = s.current_row;
  start[1] = 0;
  hid_t f_dataspace   = H5Dget_space(s.dataset);
  if (f_dataspace < 0) {
    cerr << "Error in datawriter: Could not get file dataspace." << endl;
    exit(1);
  }
  hid_t mem_dataspace = H5Screate_simple(2, dimsf, NULL);
  if (mem_dataspace < 0) {
    cerr << "Error in datawriter: Could not create memory dataspace." << endl;
    exit(1);
  }
  if (H5Sselect_hyperslab(f_dataspace, H5S_SELECT_SET, start, NULL, dimsf, NULL) < 0) {
    cerr << "Error in datawriter: Could not select hyperslab for writing." << endl;
    exit(1);
  }
  if (H5Dwrite(s.dataset, H5T_NATIVE_FLOAT, mem_dataspace, f_dataspace, H5P_DEFAULT, data) < 0) {
    cerr << "Error in datawriter: Could not write to hdf5 file." << endl;
    exit(1);
  }
  if (H5Sclose(mem_dataspace) < 0) {
    cerr << "Error in datawriter: Could not close memory dataspace" << endl;
    exit(1);
  }
  if (H5Sclose(f_dataspace) < 0) {
    cerr << "Error in datawriter: Could not close file dataspace" << endl;
    exit(1);
  }
  s.current_row += numcases;
}

void DataWriter::WriteHDF5SeqBuf(Matrix& m, const string& dataset, int numcases) {
  stream& s = streams_[dataset];
  if (s.average_online == 1) {
    WriteHDF5(m, dataset, numcases, true);
  } else {
    Matrix m_t;
    Matrix::GetTemp(m.GetCols(), m.GetRows(), m_t);
    m.CopyTransposeBig(m_t);
    
    Matrix& buf = s.seq_buf;
    if (buf.GetNumEls() == 0) {
      buf.AllocateGPUMemory(m_t.GetRows(), 1);
      buf.Set(0);
    }
    int numcases = m_t.GetCols();
    int end = 0, start = 0;

    while(start < numcases && s.current_row * s.average_online < dataset_size_) {
      Matrix slice;
      end = start + s.average_online - s.consumed;
      if (end > numcases) end = numcases;
      m_t.GetSlice(slice, start, end);
      slice.SumCols(buf, 1, 1.0 / s.average_online);
      s.consumed += end - start;
      if (s.consumed == s.average_online) {
        WriteHDF5(buf, dataset, 1, false);
        buf.Set(0);
        s.consumed = 0;
      }
      start = end;
    }
  }
}

void DataWriter::Write(vector<Layer*>& layers, int numcases) {
  for (Layer* l: layers) {
    Matrix& m = l->GetState();
    const string& dataset = l->GetName();
    stream& s = streams_[dataset];
    if(s.average_batches == 1) {
      WriteHDF5SeqBuf(m, dataset, numcases);
    } else {
      Matrix& buf = s.buf;
      if (buf.GetNumEls() == 0) {
        buf.AllocateGPUMemory(m.GetRows(), m.GetCols());
        buf.Set(0);
      }
      buf.Add(m);
      if(++s.counter == s.average_batches) {
        buf.Divide(s.average_batches);
        WriteHDF5SeqBuf(buf, dataset, numcases);
        buf.Set(0);
        s.counter = 0;
      }
    }
  }
}
