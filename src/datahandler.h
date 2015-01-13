#ifndef DATAHANDLER_H_
#define DATAHANDLER_H_
#include "layer.h"
#include "image_iterators.h"
#include "video_iterators.h"
#include <random>
#include <thread>
class DataIterator;

/** Makes data accessible to the model.
 * Provides a GetBatch() method that is used by the model to fetch
 * mini-batches.
 * Handles multiple streams of data.
 */ 
class DataHandler {
 public:
  DataHandler(const config::DatasetConfig& config);
  virtual ~DataHandler();

  void GetBatch(std::vector<Layer*>& data_layers);
  int GetBatchSize() const { return batch_size_; }
  int GetDataSetSize() const { return dataset_size_; }
  int GetMultiplicity() const { return multiplicity_; }
  void Seek(int row);
  void Sync();
  void SetFOV(const int size, const int stride, const int pad1,
              const int pad2, const int patch_size, const int num_fov_x,
              const int num_fov_y);
  void AllocateMemory();
  int GetDims(const std::string& layer_name) const;
  int GetImageSizeY(const std::string& layer_name) const;
  int GetImageSizeX(const std::string& layer_name) const;

 protected:
  void SetupShuffler();
  void ShuffleIndices();
  void LoadChunk(DataIterator& it, Matrix& mat);
  void LoadChunk(DataIterator& it, Matrix& mat, std::vector<int>& random_rows);
  void LoadChunkParallel(DataIterator& it, Matrix& mat);
  void LoadChunkParallel(DataIterator& it, Matrix& mat, std::vector<int>& random_rows);

  void PipelinedDiskAccess();
  void DiskAccess();
  void StartPreload();
  void WaitForPreload();

  std::default_random_engine generator_;
  std::map<std::string, DataIterator*> data_it_;
  std::map<std::string, Matrix> data_;
  std::vector<std::string> layer_names_;
  std::thread* preload_thread_;
  Matrix rand_perm_indices_;
  int batch_size_, chunk_size_, max_reuse_count_, reuse_counter_,
      random_access_chunk_size_, dataset_size_, start_, multiplicity_counter_,
      random_indices_ind_;
  bool restart_, nothing_on_gpu_, fits_on_gpu_;
  const bool pipeline_loads_, randomize_cpu_, randomize_gpu_;
  const int multiplicity_;
  std::vector<int> random_indices_;
};

/** Base class for implementing data iterators.
 * Each data iterator handles a single stream of data.
 * All derived classes must implement the GetNext() and Get() methods
 * and override the Seek() method appripriately.
 */ 
class DataIterator {
 public:
  DataIterator(const config::DataStreamConfig& config);
  virtual ~DataIterator() {};
  virtual void GetNext(float* data_out) = 0;
  virtual void Get(float* data_out, const int row) const = 0;
  virtual void Get(float* data_out, const int row_start, const int row_end) const;
  virtual void Seek(int row);
  virtual int Tell() const;
  virtual void Prep(const int chunk_size);
  virtual void Preprocess(Matrix& m);
  virtual void AddNoise(Matrix& input, Matrix& output);
  virtual void SetMaxDataSetSize(int max_dataset_size);
  virtual void SetFOV(const int size, const int stride, const int pad1,
                      const int pad2, const int patch_size,
                      const int num_fov_x, const int num_fov_y);
  virtual void SetNoiseSource(DataIterator* it);

  int GetDims() const;
  int GetImageSizeY() const { return gpu_image_size_y_; }
  int GetImageSizeX() const { return gpu_image_size_x_; }
  int GetDataSetSize() const;
  int GetGPUId() const { return gpu_id_;}
  void AddPCANoise(Matrix& m);
  void SampleNoise(int batch_size, int dest_num_dims, int multiplicity_id);
  bool DoParallelDiskAccess() const { return parallel_disk_access_; }
  bool NeedsNoiseFromLayer() const { return !noise_layer_name_.empty(); }
  const std::string& GetNoiseLayerName() const { return noise_layer_name_; }
  Matrix& GetWidthOffset() { return width_offset_;}
  Matrix& GetHeightOffset() { return height_offset_;}
  Matrix& GetFlipBit() { return flip_bit_;}
  int GetDestDims() const { return dest_num_dims_; }
  static DataIterator* ChooseDataIterator(const config::DataStreamConfig& config);

 protected:
  void LoadMeans(const std::string& data_file);

  int num_dims_, dataset_size_, row_, dest_num_dims_;
  Matrix mean_, std_, pca_noise1_, pca_noise2_, eig_values_, eig_vectors_,
         width_offset_, height_offset_, flip_bit_;
  const std::string file_pattern_, noise_layer_name_;
  const int image_size_y_, image_size_x_, gpu_image_size_y_, gpu_image_size_x_,
            num_colors_, gpu_id_;
  const bool translate_, flip_, normalize_, pixelwise_normalize_,
             add_pca_noise_, parallel_disk_access_, normalize_local_;
  const float pca_noise_stddev_;
  DataIterator* noise_source_;
};

/** A dummy iterator.
 * Returns random numbers.
 */ 
class DummyDataIterator : public DataIterator {
 public:
  DummyDataIterator(const config::DataStreamConfig& config);
  void GetNext(float* data_out);
  void Get(float* data_out, const int row) const;
};

/** An iterator over a dataset in an HDF5 file.
 * T specifies the type of data being iterated over.*/
template <typename T>
class HDF5DataIterator : public DataIterator {
 public:
  HDF5DataIterator(const config::DataStreamConfig& config);
  ~HDF5DataIterator();
  void GetNext(float* data_out);
  void Get(float* data_out, const int row) const;
  virtual void Get(float* data_out, const int row_start, const int row_end) const;

 protected:
  hid_t file_, dataset_, dapl_id_, m_dataspace_, type_;
  hsize_t start_[2], count_[2];
  T* buf_;
};

/** An iterator over images stored as individual files.*/
class ImageDataIterator : public DataIterator {
 public:
  ImageDataIterator(const config::DataStreamConfig& config);
  ~ImageDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;
  virtual void Seek(int row);
  virtual int Tell() const;
  virtual void Prep(const int chunk_size);
  virtual void SetMaxDataSetSize(int max_dataset_size);
  void RectifyBBox(box& b, int width, int height, int row) const;

 protected:
  RawImageFileIterator<unsigned char> *it_;
  unsigned char* buf_;
  const int raw_image_size_y_, raw_image_size_x_;
};

class CropDataIterator : public DataIterator {
 public:
  CropDataIterator(const config::DataStreamConfig& config);
  ~CropDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;
  virtual void Seek(int row);
  virtual int Tell() const;

 protected:
  CropIterator<unsigned char> *it_;
  std::vector<std::string> file_names_;
  std::vector<std::vector<box>> crops_;
  unsigned char* buf_;
  const int image_size_;
  int file_id_;
};

/** An iterator over sliding windows of an image dataset.*/
class SlidingWindowDataIterator : public DataIterator {
 public:
  SlidingWindowDataIterator(const config::DataStreamConfig& config);
  ~SlidingWindowDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;
  virtual void Seek(int row);
  virtual int Tell() const;
  virtual void SetMaxDataSetSize(int max_dataset_size);
 
 protected:
  SlidingWindowIterator<unsigned char> *it_;
  unsigned char* buf_;
  std::vector<std::string> file_names_;
  const int stride_, raw_image_size_, image_size_;
  int file_id_;
};

/** An iterator over data stored in a text file.
 * Each data vector on a new line.
 * Whitespace separated.
 */
class TextDataIterator : public DataIterator {
 public:
  TextDataIterator(const config::DataStreamConfig& config);
  ~TextDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;

 protected:
  float* data_;
};

/** An iterator over bounding boxes.*/
class BoundingBoxIterator : public DataIterator {
 public:
  BoundingBoxIterator(const config::DataStreamConfig& config);

  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;
  virtual void SetFOV(const int size, const int stride, const int pad1,
                      const int pad2, const int patch_size,
                      const int num_fov_x, const int num_fov_y);
  virtual void AddNoise(Matrix& input, Matrix& output);
  virtual void SetNoiseSource(DataIterator* it);

  static float VisibleFraction(const box& b, const box& fov);
  static float Intersection(const box& b1, const box& b2);
  static float Area(const box& b);

 protected:
  std::vector<std::vector<box>> data_;
  std::vector<int> img_width_, img_height_;
  int patch_size_;
  Matrix fovs_;
  std::vector<box> fov_box_;
  ImageDataIterator* jitter_source_;
};

/** An iterator over a sequences of a dataset.
 * T specifies the underlying DataIterator.*/
class SequenceDataIterator : public DataIterator {
 public:
  SequenceDataIterator(const config::DataStreamConfig& config);
  virtual ~SequenceDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;
  virtual void Prep(const int chunk_size);
  virtual void SetMaxDataSetSize(int max_dataset_size);
  virtual void Preprocess(Matrix& m);

 protected:
  void SetupRowMapping(const std::vector<int>& num_frames);
  
  DataIterator* it_;
  int seq_length_, frame_size_;
  std::vector<int> row_mapping_;
  const bool pick_first_;
};

/** An iterator over images stored as individual files.*/
class VideoDataIterator : public DataIterator {
 public:
  VideoDataIterator(const config::DataStreamConfig& config);
  ~VideoDataIterator();
  virtual void GetNext(float* data_out);
  virtual void Get(float* data_out, const int row) const;
  virtual void SetMaxDataSetSize(int max_dataset_size);

 protected:
  RawVideoFileIterator<unsigned char> *it_;
  unsigned char* buf_;
};


#endif
