#include "video_iterators.h"

#include <fstream>
#include <sstream>
#include <iterator>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#define PI 3.14159265

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif
#ifndef MAX
#define MAX(x,y) ((x > y) ? x : y)
#endif

using namespace cv;
using namespace std;

inline void resizeOCV(Mat &img, unsigned int width, unsigned int height) {
  Mat out1, out2;
  
  // Seems to give sharper-looking resized image by doing it in 2 steps.
  resize(img, out1, Size(width, img.rows), 0, 0, INTER_LINEAR);
  resize(out1, out2, Size(width, height), 0, 0, INTER_LINEAR);
  
  img = out2;
}

inline unsigned int spectrumOCV(Mat &img) {
  return 1 + (img.type() >> CV_CN_SHIFT);
}

template<typename T>
void getData(Mat &image, T* data_ptr) {
  int num_image_colors = spectrumOCV(image);
  int num_pixels = image.cols * image.rows;
  if (num_image_colors >= 3) {  // Image has 3 channels.
    // Convert from opencv Mat to format: "rr..gg..bb".
    unsigned int base1 =     num_pixels;
    unsigned int base2 = 2 * num_pixels;
    for (int j=0, posr=0; j < image.rows; ++j, posr+=image.cols) {
      unsigned int offset0 =         posr;
      unsigned int offset1 = base1 + posr;
      unsigned int offset2 = base2 + posr;
      char *imgr = image.ptr<char>(j);
      for (int k=0, posc=0; k < image.cols; ++k, posc+=3) {
        data_ptr[offset0 + k] = imgr[posc+2];
        data_ptr[offset1 + k] = imgr[posc+1];
        data_ptr[offset2 + k] = imgr[posc  ];
      }
    }
  } else if (num_image_colors == 1) { // Image has 1 channel.
    for (int i=0; i < 3; ++i) {
      memcpy(data_ptr + i * num_pixels, image.data, num_pixels * sizeof(T));
    }
  } else {
    cerr << "Image has " << num_image_colors << "colors." << endl;
    exit(1);
  }
}

template <typename T>
RawVideoFileIterator<T>::RawVideoFileIterator(
  const vector<string>& filelist, const int image_size_y,
  const int image_size_x, const string& num_frames_file) :
  dataset_size_(0),
  video_id_(0),
  image_id_(0),
  current_num_frames_(0),
  filenames_(filelist),
  image_size_y_(image_size_y),
  image_size_x_(image_size_x),
  num_videos_(filelist.size()) {

  cout << "Num videos " << num_videos_ << endl;
  bool read_videos = false;
  bool write_num_frames = false;
  if (num_frames_file.empty()) {
    read_videos = true;
  } else {
    ifstream f(num_frames_file);
    read_videos = !f.good();
    f.close();
    write_num_frames = true;
  }

  if (read_videos) {
    VideoCapture vid;
    ofstream out_f;
    if (write_num_frames) {
      out_f.open(num_frames_file);
      cout << "Creating boundary file." << endl;
    } else {
      cout << "No boundary file specified." << endl;
    }
    cout << "Opening video files to read number of frames .." << endl;
    int i = 0;
    for (const string& fname : filenames_) {
      vid.open(fname);
      int num_frames = vid.get(CAP_PROP_FRAME_COUNT);
      cout << "\r" << fname << "\t" << ++i << "\t" << num_frames << " frames";
      dataset_size_ += num_frames;
      num_frames_.push_back(num_frames);
      if (write_num_frames) out_f << num_frames << "\n";
      vid.release();
    }
    cout << endl;
    if (write_num_frames) out_f.close();
  } else {
    ifstream f(num_frames_file);
    int num_frames;
    while (!f.eof()) {
      f >> num_frames;
      if (!f.eof()) {
        num_frames_.push_back(num_frames);
        dataset_size_ += num_frames;
      }
    }
    f.close();
  }
  cout << " Num frames " << dataset_size_ << endl;
}

template<typename T>
RawVideoFileIterator<T>::~RawVideoFileIterator() {
}

template<typename T>
void RawVideoFileIterator<T>::SetMaxDataSetSize(int max_dataset_size) {
  if (max_dataset_size != 0 && max_dataset_size < dataset_size_) dataset_size_ = max_dataset_size;
}

template<typename T>
int RawVideoFileIterator<T>::GetDataSetSize() const {
  return dataset_size_;
}

template<typename T>
void RawVideoFileIterator<T>::GetNext(T* data_ptr) {
  if (image_id_ == current_num_frames_) {
    video_.open(filenames_[video_id_]);
    current_num_frames_ = num_frames_[video_id_];
    video_id_++;
    if (video_id_ == num_videos_) video_id_ = 0;
    image_id_ = 0;
  }
  Mat image;
  bool success = video_.read(image);
  if (success && !image.empty() && image.rows > 0 && image.cols > 0) {
    resizeOCV(image, image_size_x_, image_size_y_);
    image.copyTo(image_);
  } else {
    cerr << "Could not read frame_id " << image_id_
         << " video id " << video_id_
         << " file " << filenames_[(video_id_-1) % num_videos_]
         << " num frames " << current_num_frames_ << endl;
    if (!success) cerr << "Unsuccessful read" << endl;
    if (image.empty())  cerr << "Got empty image" << endl;
    if (image.rows <= 0)  cerr << "Got num rows " << image.rows << endl;
    if (image.cols <= 0)  cerr << "Got num cols " << image.cols << endl;

    image_.copyTo(image);
  }
  getData(image, data_ptr);
  image_id_++;
}

template class RawVideoFileIterator<float>;
template class RawVideoFileIterator<unsigned char>;
