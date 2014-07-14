#include "image_iterators.h"
#include <fstream>
#define INTERPOLATION_TYPE 3
#define PI 3.14159265
// 3 -> LINEAR
// 5 -> BICUBIC
template <typename T>
RawImageFileIterator<T>::RawImageFileIterator(
    const string& filelist, const int image_size, const int raw_image_size,
    const bool flip, const bool translate, const bool random_jitter,
    const int max_angle, const float min_scale) :
    row_(0), image_id_(-1), position_(0), image_size_(image_size),
    num_positions_((flip ? 2 : 1) * (translate ? 5 : 1)),
    raw_image_size_(raw_image_size), random_jitter_(random_jitter),
    max_angle_(max_angle), min_scale_(min_scale) {

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
  distribution_ = random_jitter_ ? new uniform_real_distribution<float>(0, 1) : NULL;
}

template<typename T>
RawImageFileIterator<T>::~RawImageFileIterator() {
  if (random_jitter_) {
    delete distribution_;
  }
  delete disp_;
}

template<typename T>
void RawImageFileIterator<T>::SetMaxDataSetSize(int max_dataset_size) {
  if (max_dataset_size > 0 && dataset_size_ > max_dataset_size) {
    dataset_size_ = max_dataset_size;
  }
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
    int width, int height, int position, int* left, int* top, bool* flip) const {
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
void RawImageFileIterator<T>::Resize(CImg<T>& image) const {
  int width = image.width(), height = image.height();
  if (width != raw_image_size_ || height != raw_image_size_) {
    int new_width, new_height;
    if (width > height) {
      new_height = raw_image_size_;
      new_width = (width * raw_image_size_) / height;
    } else {
      new_width = raw_image_size_;
      new_height = (height * raw_image_size_) / width;
    }
    image.resize(new_width, new_height, 1, -100, INTERPOLATION_TYPE);
  }
}

template<typename T>
void RawImageFileIterator<T>::SampleNoiseDistributions(const int chunk_size) {
  if (!random_jitter_) return;
  angles_.resize(chunk_size);
  trans_x_.resize(chunk_size);
  trans_y_.resize(chunk_size);
  scale_.resize(chunk_size);
  for (int i = 0; i < chunk_size; i++) {
    angles_[i] = 2 * max_angle_ * ((*distribution_)(generator_) - 0.5);
    trans_x_[i] = (*distribution_)(generator_);
    trans_y_[i] = (*distribution_)(generator_);
    scale_[i] = min_scale_ + (1 - min_scale_) * (*distribution_)(generator_);
  }
}

template<typename T>
void RawImageFileIterator<T>::AddRandomJitter(CImg<T>& image, int row) const {
  // Add Random rotation, scale, translation.

  int ind = row % angles_.size();
  float angle = angles_[ind];
  float trans_x = trans_x_[ind];
  float trans_y = trans_y_[ind];
  float scale = scale_[ind];

  // Translation.
  int width = image.width(), height = image.height();
  int size = (int)(scale * ((width < height) ? width : height));
  int left = (int)((width - size) * trans_x);
  int top = (int)((height - size) * trans_y);
  image.crop(left, top, left + size - 1, top + size - 1, true);

  // Resize (so that after rotation, we can crop out the central raw_image_size_ * raw_image_size_ image).
  int rot_adjusted_size = (int)(raw_image_size_ * (sin(fabs(angle)*PI/180) + cos(fabs(angle)*PI/180)));
  image.resize(rot_adjusted_size, rot_adjusted_size, 1, -100, INTERPOLATION_TYPE);

  // Rotation.
  image.rotate(angle, 1, 0);
  
  // Crop out the border created by rotation.
  left = image.width() / 2 - raw_image_size_/2;
  top = image.height() / 2 - raw_image_size_/2;
  image.crop(left, top, left + raw_image_size_-1, top + raw_image_size_-1, true);
}

template <typename T>
void RawImageFileIterator<T>::Get(T* data_ptr, const int row, const int position) const {
  const string& filename = filenames_[row];
  CImg<T> image(filename.c_str());
  if (random_jitter_) {
    AddRandomJitter(image, row);
  }
  Resize(image);
  ExtractRGB(image, data_ptr, position);
}

template <typename T>
void RawImageFileIterator<T>::GetNext(T* data_ptr, const int row, const int position) {
  if (image_id_ != row) {  // Load a new image from disk.
    image_id_ = row;
    string& filename = filenames_[row];
    image_.assign(filename.c_str());
    
    if (random_jitter_) {
      AddRandomJitter(image_, row);
    }
    // Resize it so that the shorter side is raw_image_size_.
    Resize(image_);
  }
  ExtractRGB(image_, data_ptr, position);
}

template<typename T>
void RawImageFileIterator<T>::ExtractRGB(CImg<T>& image, T* data_ptr, int position) const {
  int width = image.width(), height = image.height();
  int left = 0, top = 0;
  bool flip = false;
  GetCoordinates(width, height, position, &left, &top, &flip);

  CImg<T> img = image.get_crop(
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
}
template class RawImageFileIterator<float>;
template class RawImageFileIterator<unsigned char>;


template <typename T>
SlidingWindowIterator<T>::SlidingWindowIterator(const int window_size, const int stride):
  window_size_(window_size), stride_(stride), num_windows_(0),
  center_x_(0), center_y_(0), done_(true) {}

template <typename T>
void SlidingWindowIterator<T>::SetImage(const string& filename) {
  image_.assign(filename.c_str());
  center_x_ = 0;
  center_y_ = 0;
  int num_modules_x = (image_.width() - window_size_ % 2) / stride_ + 1;
  int num_modules_y = (image_.height() - window_size_ % 2) / stride_ + 1;
  num_windows_ = num_modules_x * num_modules_y;
  done_ = false;
}

template <typename T>
void SlidingWindowIterator<T>::Reset() {
  done_ = true;
}

template <typename T>
void SlidingWindowIterator<T>::GetNext(T* data_ptr) {
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

template <typename T>
bool SlidingWindowIterator<T>::Done() {
  return done_;
}

template <typename T>
void SlidingWindowIterator<T>::GetNext(T* data_ptr, int center_x, int center_y) {
  int left    = center_x - window_size_ / 2,
      right   = left + window_size_,
      top     = center_y - window_size_ / 2,
      bottom  = top + window_size_;
  CImg<T> img = image_.get_crop(left, top, right - 1, bottom - 1, true);
  int num_pixels = window_size_ * window_size_ * 3;
  memcpy(data_ptr, img.data(), num_pixels * sizeof(float));
}

template class SlidingWindowIterator<float>;
template class SlidingWindowIterator<unsigned char>;

