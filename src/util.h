#ifndef UTIL_H_
#define UTIL_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef __USE_GNU
#define __USE_GNU
#endif

#ifdef USE_MPI
#include "mpi.h"
#endif
#include <string>
#define cimg_use_jpeg
#define cimg_use_lapack
#include "CImg/CImg.h"
#include <stdio.h>
#include <google/protobuf/text_format.h>
#include "convnet_config.pb.h"
#include "hdf5.h"
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <ucontext.h>
#include <unistd.h>

#define MPITAG_WEIGHTGRAD 11
#define MPITAG_TRAINERROR 12

template<class T> void ReadPbtxt(const std::string& pbtxt_file, T& model);
template<class T> void WritePbtxt(const std::string& pbtxt_file, const T& model);

void WriteModelBinary(const std::string& output_file, const config::Model& model);
void ReadModelBinary(const std::string& input_file, config::Model& model);
void WriteHDF5CPU(hid_t file, float* mat, int rows, int cols, const std::string& name);
void WriteHDF5CPU(hid_t file, std::vector<float>& mat, int rows, int cols, const std::string& name);
void ReadHDF5IntAttr(hid_t file, const std::string& name, int* val);
void WriteHDF5IntAttr(hid_t file, const std::string& name, const int* val);

// mat must already be allocated. Use ReadHDF5Shape to figure out the shape, if needed.
void ReadHDF5CPU(hid_t file, float* mat, int size, const std::string& name);
void ReadHDF5Shape(hid_t file, const std::string& name, int* rows, int* cols);
void ReadHDF5ShapeFromFile(const std::string& file_name, const std::string& dataset_name, int* rows, int* cols);

void readFileList(const std::string& list_name, std::vector<std::string>& filenames);
void split(const std::string &s, std::vector<std::string> &elems, char delim);
void ParseBoardIds(const std::string& board, std::vector<int>& boards);
void WaitForEnter();
int Bound(int val, int lb, int ub);
std::string GetTimeStamp();
void TimestampModelFile(const std::string& src_file, const std::string& dest_file, const std::string& timestamp);

bool ReadLines(const std::string& filename, std::vector<std::string>& lines);
void DrawRectange(cimg_library::CImg<float>& img, int xmin, int ymin, int xmax, int ymax, const float* color, int thickness);

// Outputs a std::string that describes the err_code.
std::string GetStringError(int err_code);

// a+=b;
void AddVectors(std::vector<float>& a, std::vector<float>& b);

//
// ImageDisplayer
//

class ImageDisplayer {
 public:
  ImageDisplayer();
  ImageDisplayer(int width, int height, int num_colors, bool show_separate, const std::string& name);
  
  void SetTitle(const std::string& title) {title_ = title;}
  void DisplayImage(float* data, int spacing, int image_id);
  void CreateImage(const float* data, int num_images, int image_id, cimg_library::CImg<float>& img);
  void DisplayWeights(float* data, int size, int num_filters, int display_size, bool yuv = false);
  void DisplayLocalization(float* data, float* preds, float* gt, int num_images);
  void SetFOV(int size, int stride, int pad1, int pad2, int patch_size, int num_fov_x, int num_fov_y);
  

  static void YUVToRGB(const float* yuv, float* rgb, int spacing);
  static void RGBToYUV(const float* rgb, float* yuv, int spacing);

 private:
  
  cimg_library::CImgDisplay disp_;
  int width_, height_, num_colors_;
  bool show_separate_;
  std::string title_;

  float fov_size_, fov_stride_, fov_pad1_, fov_pad2_;
  int num_fov_x_, num_fov_y_;
};


#endif
