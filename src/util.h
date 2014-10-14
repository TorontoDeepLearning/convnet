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
#include <signal.h>
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

using namespace cimg_library;
using namespace std;

// Outputs a string that describes the err_code.
string GetStringError(int err_code);

template<class T> void ReadPbtxt(const string& pbtxt_file, T& model);

void WriteModelBinary(const string& output_file, const config::Model& model);
void ReadModelBinary(const string& input_file, config::Model& model);
void WriteHDF5CPU(hid_t file, float* mat, int rows, int cols, const string& name);
void ReadHDF5IntAttr(hid_t file, const string& name, int* val);
void WriteHDF5IntAttr(hid_t file, const string& name, const int* val);

// mat must already be allocated. Use ReadHDF5Shape to figure out the shape, if needed.
void ReadHDF5CPU(hid_t file, float* mat, int size, const string& name);
void ReadHDF5Shape(hid_t file, const string& name, int* rows, int* cols);
void ReadHDF5ShapeFromFile(const string& file_name, const string& dataset_name, int* rows, int* cols);

void readFileList(const string& list_name, vector<string>& filenames);
void ParseBoardIds(const string& board, vector<int>& boards);
void WaitForEnter();
int Bound(int val, int lb, int ub);
string GetTimeStamp();
void TimestampModelFile(const string& src_file, const string& dest_file, const string& timestamp);

bool ReadLines(const string& filename, vector<string>& lines);
void DrawRectange(CImg<float>& img, int xmin, int ymin, int xmax, int ymax, const float* color, int thickness);


class ImageDisplayer {
 public:
  ImageDisplayer();
  ImageDisplayer(int width, int height, int num_colors, bool show_separate, const string& name);
  
  void SetTitle(const string& title) {title_ = title;}
  void DisplayImage(float* data, int spacing, int image_id);
  void CreateImage(const float* data, int num_images, int image_id, CImg<float>& img);
  void DisplayWeights(float* data, int size, int num_filters, int display_size, bool yuv = false);
  void DisplayLocalization(float* data, float* preds, float* gt, int num_images);
  void SetFOV(int size, int stride, int pad1, int pad2, int patch_size, int num_fov_x, int num_fov_y);
  

  static void YUVToRGB(const float* yuv, float* rgb, int spacing);
  static void RGBToYUV(const float* rgb, float* yuv, int spacing);

 private:
  
  CImgDisplay disp_;
  int width_, height_, num_colors_;
  bool show_separate_;
  string title_;

  float fov_size_, fov_stride_, fov_pad1_, fov_pad2_;
  int num_fov_x_, num_fov_y_;
};


#endif
