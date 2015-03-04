#ifndef VIDEO_ITERATORS_
#define VIDEO_ITERATORS_

#include <string>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/videoio/videoio.hpp>

template<typename T>
class RawVideoFileIterator {
public:
  RawVideoFileIterator(const std::vector<std::string> &filelist, const int image_size_y,
                       const int image_size_x, const std::string& num_frames_file);
  virtual ~RawVideoFileIterator();

  // Memory must already be allocated : image_size_y * image_size_x * num_colors.
  void GetNext(T* data_ptr);

  void SetMaxDataSetSize(int max_dataset_size);
  int GetDataSetSize() const;

private:
  int dataset_size_, video_id_, image_id_, current_num_frames_;
  std::vector<std::string> filenames_;
  std::vector<int> num_frames_;
  const int image_size_y_, image_size_x_, num_videos_;
  cv::VideoCapture video_;
  cv::Mat image_;
  bool flip_;
};
#endif
