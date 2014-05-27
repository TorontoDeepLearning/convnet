#include "image_iterators.h"
#include <fstream>

template <typename T>
RawImageFileIterator<T>::RawImageFileIterator(
    const string& filelist, const int image_size, const int raw_image_size,
    const bool flip, const bool translate) :
    row_(0), image_id_(-1), position_(0), image_size_(image_size),
    num_positions_((flip ? 2 : 1) * (translate ? 5 : 1)),
    raw_image_size_(raw_image_size) {

  ifstream f(filelist, ios::in);
  if (!f.is_open()) {
    cerr << "Could not open data file : " << filelist << endl;
    exit(1);
  }
  while (!f.eof()) {
    string str;
    f >> str;
    if (!f.eof()) filenames_.push_back(str);
  }
  f.close();

  dataset_size_ = filenames_.size();

  disp_ = new CImgDisplay();
}

template <typename T>
void RawImageFileIterator<T>::GetNext(T* data_ptr) {
  GetNext(data_ptr, row_, position_);
  position_++;
  if (position_ == num_positions_) {
    position_ = 0;
    row_++;
    if (row_ == dataset_size_) row_ = 0;
  }
}

template <typename T>
void RawImageFileIterator<T>::GetCoordinates(
    int width, int height, int position, int* left, int* top, bool* flip) {
  *flip = position >= 5;
  position %= 5;
  int x_slack = width - image_size_;
  int y_slack = height - image_size_;
  switch(position) {
    case 0 :  // Center. 
            *left = x_slack / 2;
            *top = y_slack / 2;
            break;
    case 1 :  // Top left.
            *left = 0;
            *top = 0;
            break;
    case 2 :  // Top right.
            *left = x_slack;
            *top = 0;
            break;
    case 3 :  // Bottom right.
            *left = x_slack;
            *top = y_slack;
            break;
    case 4 :  // Bottom left.
            *left = 0;
            *top = y_slack;
            break;
  }
}

template <typename T>
void RawImageFileIterator<T>::GetNext(T* data_ptr, const int row, const int position) {
  if (image_id_ != row) {  // Load a new image from disk.
    image_id_ = row;
    string& filename = filenames_[row];
    image_.assign(filename.c_str());
    
    // Resize it so that the shorter side is image_size_.
    int width = image_.width(), height = image_.height();
    int new_width, new_height;
    if (width > height) {
      new_height = raw_image_size_;
      new_width = (width * raw_image_size_) / height;
    } else {
      new_width = raw_image_size_;
      new_height = (height * raw_image_size_) / width;
    }
    image_.resize(new_width, new_height, 1, -100, 3);
  }
  int width = image_.width(), height = image_.height();
  int left = 0, top = 0;
  bool flip = false;
  GetCoordinates(width, height, position, &left, &top, &flip);

  CImg<T> img = image_.get_crop(
      left, top, left + image_size_ - 1, top + image_size_ - 1, true);

  if (flip) img.mirror('x');
  //img.display(*disp_);

  int num_image_colors = img.spectrum();
  int num_pixels = image_size_ * image_size_;
  if (num_image_colors >= 3) {  // Image has 3 channels.
    memcpy(data_ptr, img.data(), 3 * num_pixels * sizeof(T));
  } else if (num_image_colors == 1) {  // Image has 1 channel.
    for (int i = 0; i < 3; i++) {
      memcpy(data_ptr + i * num_pixels, img.data(), num_pixels * sizeof(T));
    }
  } else {
    cerr << "Image has " << num_image_colors << "colors." << endl;
    exit(1);
  }
  /*
  for (int i = 0; i < 10; i++) {
    cout << data_ptr[i] << " ";
  }
  cout << endl;
  */
}
template class RawImageFileIterator<float>;
template class RawImageFileIterator<unsigned char>;


SlidingWindowIterator::SlidingWindowIterator(const int window_size, const int stride):
  window_size_(window_size), stride_(stride), num_windows_(0),
  center_x_(0), center_y_(0), done_(true) {}

void SlidingWindowIterator::SetImage(const string& filename) {
  image_.assign(filename.c_str());
  center_x_ = 0;
  center_y_ = 0;
  int num_modules_x = (image_.width() - window_size_ % 2) / stride_ + 1;
  int num_modules_y = (image_.height() - window_size_ % 2) / stride_ + 1;
  num_windows_ = num_modules_x * num_modules_y;
  done_ = false;
}

void SlidingWindowIterator::Reset() {
  done_ = true;
}

void SlidingWindowIterator::GetNext(float* data_ptr) {
  GetNext(data_ptr, center_x_, center_y_);
  center_x_ += stride_;
  if (center_x_  >= image_.width()) {
    center_x_ = 0;
    center_y_ += stride_;
    if (center_y_ >= image_.height()) {
      center_y_ = 0;
      done_ = true;
    }
  }
}

bool SlidingWindowIterator::Done() {
  return done_;
}

void SlidingWindowIterator::GetNext(float* data_ptr, int center_x, int center_y) {
  int left    = center_x - window_size_ / 2,
      right   = left + window_size_,
      top     = center_y - window_size_ / 2,
      bottom  = top + window_size_;
  CImg<float> img = image_.get_crop(left, top, right - 1, bottom - 1, true);
  int num_pixels = window_size_ * window_size_ * 3;
  memcpy(data_ptr, img.data(), num_pixels * sizeof(float));
}


