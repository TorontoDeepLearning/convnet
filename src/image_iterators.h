#ifndef IMAGE_ITERATORS_
#define IMAGE_ITERATORS_
#define cimg_use_jpeg
#define cimg_use_lapack
#include <string>
#include <iostream>
#include <vector>
#include "CImg.h"
using namespace cimg_library;
using namespace std;

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
  int GetDataSetSize() const { return dataset_size_;}

 private:
  void GetCoordinates(int width, int height, int position, int* left, int* top, bool* flip);

  int dataset_size_, row_, image_id_, position_;
  vector<string> filenames_;
  CImg<T> image_;
  const int image_size_, num_positions_, raw_image_size_;
  CImgDisplay* disp_;
};

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
