#ifndef RAW_IMAGE_DATAHANDLER_
#define RAW_IMAGE_DATAHANDLER_
#include "datahandler.h"

template<typename T>
class RawImageFileIterator {
 public:
  RawImageFileIterator(const string& filelist, const int image_size,
                       const int raw_image_size, const bool flip,
                       const bool translate);

  // Memory must already be allocated : num_dims * num_dims * 3.
  void GetNext(T* data_ptr);
  void GetNext(T* data_ptr, const int row, const int position);

  void Seek(int row) { row_ = row; }
  int GetDatasetSize() const { return dataset_size_;}

 private:
  void GetCoordinates(int width, int height, int position, int* left, int* top, bool* flip);

  int dataset_size_, row_, image_id_, position_;
  vector<string> filenames_;
  CImg<T> image_;
  const int image_size_, num_positions_, raw_image_size_;
  CImgDisplay* disp_;
};


// Handles image data.
template<typename T>
class RawImageDataHandler : public DataHandler {
 public:
  RawImageDataHandler(const config::DatasetConfig& config);
  ~RawImageDataHandler();
  virtual void GetBatch(vector<Layer*>& data_layers);

 private:
  void LoadMeansFromDisk();

  const int image_size_;
  RawImageFileIterator<T> *it_;
  Matrix mean_, std_;
  T* image_buf_;
  const bool pixelwise_normalize_;
};

class SlidingWindowIterator {
 public:
  SlidingWindowIterator(const int window_size, const int stride);
  void SetImage(const string& filename);
  int GetNumWindows() { return num_windows_;}
  void GetNext(float* data_ptr);
  void GetNext(float* data_ptr, int left, int top);
  bool Done();
  void Reset();

 private:
  const int window_size_, stride_;
  int num_windows_, center_x_, center_y_;
  CImg<float> image_;
  bool done_;

};

class SlidingWindowDataHandler : public DataHandler {
 public:
  SlidingWindowDataHandler(const config::DatasetConfig& config);
  ~SlidingWindowDataHandler();
  virtual void GetBatch(vector<Layer*>& data_layers);

 private:
  void LoadMeansFromDisk();
  SlidingWindowIterator* it_;
  float* buf_;
  Matrix mean_, std_;
  int image_id_;
  vector<string> image_file_names_;
};
#endif
