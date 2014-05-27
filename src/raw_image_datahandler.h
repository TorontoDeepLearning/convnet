#ifndef RAW_IMAGE_DATAHANDLER_
#define RAW_IMAGE_DATAHANDLER_
#include "datahandler.h"
#include "image_iterators.h"

// Handles raw image data.
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
