#ifndef IMAGE_ITERATORS_
#define IMAGE_ITERATORS_
#define cimg_use_jpeg
#define cimg_use_lapack
#include "CImg/CImg.h"
#include <string>
#include <iostream>
#include <vector>
#include <random>
using namespace cimg_library;
using namespace std;

/** An iterator over list of image files.
 * Takes a list of image files and iterates over them.
 * Images are cropped, jittered, flip etc.
 */ 
template<typename T>
class RawImageFileIterator {
 public:
  RawImageFileIterator(const string& filelist, const int image_size,
                       const int raw_image_size, const bool flip,
                       const bool translate, const bool random_jitter,
                       const int max_angle=45, const float min_scale=0.9);
  ~RawImageFileIterator();

  // Memory must already be allocated : num_dims * num_dims * 3.
  void GetNext(T* data_ptr);
  void GetNext(T* data_ptr, const int row, const int position);
  void Get(T* data_ptr, const int row, const int position) const;

  void Seek(int row) { row_ = row; }
  int Tell() const { return row_; }
  int GetDataSetSize() const { return dataset_size_;}
  void SampleNoiseDistributions(const int chunk_size);
  void SetMaxDataSetSize(int max_dataset_size);

 private:
  void Resize(CImg<T>& image) const;
  void AddRandomJitter(CImg<T>& image, int row) const;
  void ExtractRGB(CImg<T>& image, T* data_ptr, int position) const;
  void GetCoordinates(int width, int height, int position, int* left, int* top, bool* flip) const;

  default_random_engine generator_;
  uniform_real_distribution<float> * distribution_;
  vector<float> angles_, trans_x_, trans_y_, scale_;
  int dataset_size_, row_, image_id_, position_;
  vector<string> filenames_;
  CImg<T> image_;
  const int image_size_, num_positions_, raw_image_size_;
  CImgDisplay* disp_;
  const bool random_jitter_;
  const int max_angle_;
  const float min_scale_;
};

/** An iterator over sliding windows of an image.*/
template<typename T>
class SlidingWindowIterator {
 public:
  SlidingWindowIterator(const int window_size, const int stride);
  void SetImage(const string& filename);
  int GetNumWindows() { return num_windows_;}
  void GetNext(T* data_ptr);
  void GetNext(T* data_ptr, int left, int top);
  bool Done();
  void Reset();

 private:
  const int window_size_, stride_;
  int num_windows_, center_x_, center_y_;
  CImg<T> image_;
  bool done_;
};


#endif
