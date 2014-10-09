#include "util.h"
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <limits>

using namespace std;

void ParseBoardIds(const string& board, vector<int>& boards) {
  for (auto b:board)  {
    string currBoard;
    currBoard.push_back(b);
    boards.push_back(atoi(currBoard.c_str()));
  }
}

void WaitForEnter() {
  cout << "Press ENTER to continue...";
  cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

int Bound(int val, int lb, int ub) {
  val = val > ub ? ub : val;
  val = val < lb ? lb : val;
  return val;
}

void TimestampModelFile(const string& src_file, const string& dest_file, const string& timestamp) {
  ifstream src(src_file, ios::binary);
  ofstream dst(dest_file, ios::binary);
  if (!dst) {
    cerr << "Error: Could not write to " << dest_file << endl;
    exit(1);
  } else {
    cout << "Timestamped model : " << dest_file << endl;
  }
  dst << src.rdbuf();
  dst << endl << "timestamp : \"" << timestamp << "\"" << endl;
  dst.close();
  src.close();
}

// Year-Month-Day-Hour-Minute-Second
string GetTimeStamp() {
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  char timestr[30];
  strftime(timestr, sizeof(timestr), "%Y%m%d%H%M%S", timeinfo);
  stringstream ss;
  ss << timestr;
  return ss.str();
}

template <class T>
void ReadPbtxt(const string& pbtxt_file, T& model) {
  stringstream ss;
  ifstream file(pbtxt_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &model)) {
    cerr << "Could not read text proto buffer : " << pbtxt_file << endl;
    exit(1);
  }
}

template void ReadPbtxt<config::Model>(const string&, config::Model&);
template void ReadPbtxt<config::FeatureExtractorConfig>(const string&, config::FeatureExtractorConfig&);
template void ReadPbtxt<config::DatasetConfig>(const string&, config::DatasetConfig&);
template void ReadPbtxt<config::DataStreamConfig>(const string&, config::DataStreamConfig&);
template void ReadPbtxt<config::Layer>(const string&, config::Layer&);

/*
void ReadModelText(const string& model_file, config::Model& model) {
  stringstream ss;
  ifstream file(model_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &model)) {
    cerr << "Could not read text proto buffer : " << model_file << endl;
    exit(1);
  }
}
void ReadFeatureExtractorConfig(const string& config_file, config::FeatureExtractorConfig& config) {
  stringstream ss;
  ifstream file(config_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &config)) {
    cerr << "Could not read text proto buffer : " << config_file << endl;
    exit(1);
  }
}
void ReadDataConfig(const string& data_config_file, config::DatasetConfig& data_config) {
  stringstream ss;
  ifstream file(data_config_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &data_config)) {
    cerr << "Could not read text proto buffer : " << data_config_file << endl;
    exit(1);
  }
}

void ReadDataStreamConfig(const string& data_config_file, config::DataStreamConfig& data_config) {
  stringstream ss;
  ifstream file(data_config_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &data_config)) {
    cerr << "Could not read text proto buffer : " << data_config_file << endl;
    exit(1);
  }
}

void ReadLayerConfig(const string& layer_config_file, config::Layer& layer_config) {
  stringstream ss;
  ifstream file(layer_config_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), &layer_config)) {
    cerr << "Could not read text proto buffer : " << layer_config_file << endl;
    exit(1);
  }
}
*/
void WriteModelBinary(const string& output_file, const config::Model& model) {
  ofstream out(output_file.c_str());
  model.SerializeToOstream(&out);
  out.close();
}

void ReadModelBinary(const string& input_file, config::Model& model) {
  ifstream in(input_file.c_str());
  model.ParseFromIstream(&in);
  in.close();
}


void WriteHDF5CPU(hid_t file, float* mat, int rows, int cols, const string& name) {
  hid_t dataset, dataspace;
  hsize_t dimsf[2];
  dimsf[0] = rows;
  dimsf[1] = cols;
  dataspace = H5Screate_simple(2, dimsf, NULL);
  dataset = H5Dcreate(file, name.c_str(), H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat);
  H5Sclose(dataspace);
  H5Dclose(dataset);
}

void ReadHDF5ShapeFromFile(const string& file_name, const string& dataset_name, int* rows, int* cols) {
  hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  ReadHDF5Shape(file, dataset_name, rows, cols);
  H5Fclose(file);
}

void ReadHDF5Shape(hid_t file, const string& name, int* rows, int* cols) {
  hid_t dataset, dataspace;
  dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  dataspace = H5Dget_space(dataset);
  const int ndims = H5Sget_simple_extent_ndims(dataspace);
  hsize_t dims_out[2];
  H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
  *cols = dims_out[0];
  *rows = (ndims == 1) ? 1 :dims_out[1];
  H5Sclose(dataspace);
  H5Dclose(dataset);
}

void WriteHDF5IntAttr(hid_t file, const string& name, const int* val) {
  hid_t aid, attr;
  aid  = H5Screate(H5S_SCALAR);
  attr = H5Acreate2(file, name.c_str(), H5T_NATIVE_INT, aid, H5P_DEFAULT,
                    H5P_DEFAULT);
  H5Awrite(attr, H5T_NATIVE_INT, val);
  H5Sclose(aid);
  H5Aclose(attr);
}

void ReadHDF5IntAttr(hid_t file, const string& name, int* val) {
  hid_t attr = H5Aopen(file, name.c_str(), H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_INT, val);
  H5Aclose(attr);
}

void ReadHDF5CPU(hid_t file, float* mat, int size, const string& name) {
  hid_t dataset, dataspace;
  dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  dataspace = H5Dget_space(dataset);
  const int ndims = H5Sget_simple_extent_ndims(dataspace);
  hsize_t dims_out[2];
  H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
  int rows = (ndims == 1) ? 1 : dims_out[1];
  int datasize = dims_out[0] * rows;
  if (size != datasize) {
    cerr << "Dimension mismatch: Expected "
         << size << " Got " << rows << "-" << dims_out[0] << endl;
    exit(1);
  }
  H5Dread(dataset, H5T_NATIVE_FLOAT, dataspace, dataspace, H5P_DEFAULT, mat);
  H5Sclose(dataspace);
  H5Dclose(dataset);
}

bool ReadLines(const string& filename, vector<string>& lines) {
  ifstream f(filename, ios::in);
  if (!f.is_open()) {
    cerr << "Could not open data file : " << filename << endl;
    return false;
  }

  while (!f.eof()) {
    string str;
    f >> str;
    if (!f.eof()) lines.push_back(str);
  }
  f.close();
  return true;
}

string GetStringError(int err_code) {
  if (err_code == -1)
    return "Incompatible matrix dimensions.";
  if (err_code == -2)
    return "CUBLAS error.";
  if (err_code == -3)
    return "CUDA error ";
  if (err_code == -4)
    return "Operation not supported on views.";
  if (err_code == -5)
    return "Operation not supported on transposed matrices.";
  if (err_code == -6)
    return "";
  if (err_code == -7)
    return "Incompatible transposedness.";
  if (err_code == -8)
    return "Matrix is not in device memory.";
  if (err_code == -9)
    return "Operation not supported.";
  return "Some error";
}


void DrawRectange(CImg<float>& img, int xmin, int ymin, int xmax, int ymax, const float* color, int thickness) {
  for (int i = 0; i < thickness; i++) {
    img.draw_rectangle(xmin-i, ymin-i, xmax+i, ymax+i, color, 1.0, ~0U);
  }
}

ImageDisplayer::ImageDisplayer(int width, int height, int num_colors, bool show_separate, const string& title) :
  width_(width), height_(height), num_colors_(num_colors),
  show_separate_(show_separate), title_(title) {
  disp_.set_title(title_.c_str());
}

ImageDisplayer::ImageDisplayer() :
  width_(0), height_(0), num_colors_(3), show_separate_(false), title_("") {
}

void ImageDisplayer::DisplayImage(float* data, int num_images, int image_id) {
  CImg<float> img;
  CreateImage(data, num_images, image_id, img);
  disp_.set_title(title_.c_str());
  img.display(disp_);
}

void ImageDisplayer::CreateImage(const float* data, int num_images, int image_id, CImg<float>& img) {
  int num_colors_width = (int)sqrt(num_colors_);
  int num_colors_height = (num_colors_ + num_colors_width - 1) / num_colors_width;
  int display_width = show_separate_ ? width_ * num_colors_width: width_;
  int display_height = show_separate_ ? height_ * num_colors_height: height_;
  int display_colors = show_separate_ ? 1 : num_colors_;

  img.assign(display_width, display_height, 1, display_colors);
  img.fill(0);
  float val;
  for (int k = 0; k < num_colors_; k++) {
    for (int i = 0; i < height_; i++) {
      for (int j = 0; j < width_; j++) {
        val = data[image_id + num_images * (j + width_ * (i + k * height_))];
        if (show_separate_) {
          img(j + (k % num_colors_width) * width_, i + (k / num_colors_width) * height_, 0, 0) = val;
        } else {
          img(j, i, 0, k) = val;
        }
      }
    }
  }
}

void ImageDisplayer::YUVToRGB(const float* yuv, float* rgb, int spacing) {
  const float *Y = yuv, *U = &yuv[spacing], *V = &yuv[2 * spacing];
  float *R = rgb, *G = &rgb[spacing], *B = &rgb[2 * spacing];
  for (int i = 0; i < spacing; i++) {
    float y = Y[i], u = U[i], v = V[i];
    R[i] = y               + 1.28033 * v; 
    G[i] = y - 0.21482 * u - 0.38059 * v;
    B[i] = y + 2.12798 * u;
  }
}

void ImageDisplayer::RGBToYUV(const float* rgb, float* yuv, int spacing) {
  const float *R = rgb, *G = &rgb[spacing], *B = &rgb[2 * spacing];
  float *Y = yuv, *U = &yuv[spacing], *V = &yuv[2 * spacing];
  for (int i = 0; i < spacing; i++) {
    float r = R[i], g = G[i], b = B[i];
    Y[i] =  r * 0.21260  + g * 0.71520 + b * 0.07220; 
    U[i] = -r * 0.09991  - g * 0.33609 + b * 0.43600; 
    V[i] =  r * 0.61500  - g * 0.55861 - b * 0.05639; 
  }
}

void ImageDisplayer::DisplayWeights(float* data, int size, int num_filters, int display_size, bool yuv) {
  int num_filters_w = int(sqrt(num_filters));
  int num_filters_h = num_filters / num_filters_w +  (((num_filters % num_filters_w) > 0) ? 1 : 0);
  int data_pos, row, col;
  CImg<float> img(size * num_filters_w, size * num_filters_h, 1, 3);
  img.fill(0);
  float norm = 0;
  if (yuv) YUVToRGB(data, data, num_filters * size * size);
  for (int f = 0; f < num_filters; f++) {
    norm = 0;
    for (int i = 0; i < size * size * 3; i++) {
      norm += data[i * num_filters + f] * data[i * num_filters + f];
    }
    norm = sqrt(norm);
    for (int i = 0; i < size * size * 3; i++) {
      data[i * num_filters + f] /= norm;
    }
  }
  for (int f = 0; f < num_filters; f++) {
    for (int k = 0; k < 3; k++) {
      for (int h = 0; h < size; h++) {
        for (int w = 0; w < size; w++) {
          data_pos = f + num_filters * (w + size * (h + size * k));
          col = w + size * (f % num_filters_w);
          row = h + size * (f / num_filters_w);
          img(col, row, 0, k) = data[data_pos];
        }
      }
    }
  }
  const unsigned char color[] = {0, 0, 0};
  img.resize(display_size, display_size);
  for (int i = 0; i < num_filters_w; i++) {
    int pos = (i * img.width()) / num_filters_w;
    img.draw_line(pos, 0, pos, img.height(), color);
  }
  for (int i = 0; i < num_filters_h; i++) {
    int pos = (i * img.height()) / num_filters_h;
    img.draw_line(0, pos, img.width(), pos, color);
  }
  img.display(disp_);
}

void ImageDisplayer::SetFOV(int size, int stride, int pad1, int pad2,
                            int patch_size, int num_fov_x, int num_fov_y) {
  fov_size_ = (float)size / patch_size;
  fov_stride_ = (float)stride / patch_size;
  fov_pad1_ = (float)pad1 / patch_size;
  fov_pad2_ = (float)pad2 / patch_size;
  num_fov_x_ = num_fov_x;
  num_fov_y_ = num_fov_y;
}

void ImageDisplayer::DisplayLocalization(float* data, float* preds, float* gt, int num_images) {
  int image_id = 0;

  int num_fovs = num_fov_y_ * num_fov_x_;
  
  CImg<float> img;
  CreateImage(data, num_images, image_id, img);
  const int image_size = 250;
  img.resize(image_size, image_size);

  CImg<float> img2 = CImg<float>(img);

  const float green[] = {0, 1, 0};
  const float blue[] = {0, 0, 1};

  float fov_x, fov_y;
  gt += image_id;
  preds += image_id;

  for (int i = 0; i < num_fovs; i++) {
    int r = i / num_fov_y_, c = i % num_fov_y_;
    fov_x = -fov_pad1_ + c * fov_stride_;
    fov_y = -fov_pad1_ + r * fov_stride_;
    float xmin_gt = gt[i * num_images];
    float ymin_gt = gt[(i+num_fovs) * num_images];
    float xmax_gt = gt[(i+num_fovs*2) * num_images];
    float ymax_gt = gt[(i+num_fovs*3) * num_images];
    int xmin_gt2 = (int)((xmin_gt + fov_x) * image_size);
    int ymin_gt2 = (int)((ymin_gt + fov_y) * image_size);
    int xmax_gt2 = (int)((xmax_gt + fov_x) * image_size);
    int ymax_gt2 = (int)((ymax_gt + fov_y) * image_size);

    float xmin_preds = preds[i * num_images];
    float ymin_preds = preds[(i+num_fovs) * num_images];
    float xmax_preds = preds[(i+num_fovs*2) * num_images];
    float ymax_preds = preds[(i+num_fovs*3) * num_images];
    int xmin_preds2 = (int)((xmin_preds + fov_x) * image_size);
    int ymin_preds2 = (int)((ymin_preds + fov_y) * image_size);
    int xmax_preds2 = (int)((xmax_preds + fov_x) * image_size);
    int ymax_preds2 = (int)((ymax_preds + fov_y) * image_size);

    DrawRectange(img, xmin_gt2, ymin_gt2, xmax_gt2, ymax_gt2, green, 3);
    DrawRectange(img2, xmin_preds2, ymin_preds2, xmax_preds2, ymax_preds2, blue, 3);
  }

  CImgList<float> img_list(img, img2);
  img_list.display(disp_);

}
