#include "util.h"
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <cxxabi.h>

using namespace std;

void BackTraceHandler(int sig) {
  void *array[10];
  size_t size;

  size = backtrace(array, 10);

  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}
void crit_err_hdlr(int sig_num, siginfo_t * info, void * ucontext) {
  sig_ucontext_t * uc = (sig_ucontext_t *)ucontext;
  void * caller_address;
#if defined(__i386__) // gcc specific
  caller_address = (void *) uc->uc_mcontext.eip; // EIP: x86 specific
#elif defined(__x86_64__) // gcc specific
  caller_address = (void *) uc->uc_mcontext.rip; // RIP: x86_64 specific
#else
#error Unsupported architecture. // TODO: Add support for other arch.
#endif

  std::cerr << "signal " << sig_num 
    << " (" << strsignal(sig_num) << "), address is " 
    << info->si_addr << " from " << caller_address 
    << std::endl << std::endl;

  void * array[50];
  int size = backtrace(array, 50);

  array[1] = caller_address;

  char ** messages = backtrace_symbols(array, size);    

  // skip first stack frame (points here)
  for (int i = 1; i < size && messages != NULL; ++i) {
    char *mangled_name = 0, *offset_begin = 0, *offset_end = 0;

    // find parantheses and +address offset surrounding mangled name
    for (char *p = messages[i]; *p; ++p) {
      if (*p == '(') {
        mangled_name = p; 
      } else if (*p == '+') {
        offset_begin = p;
      } else if (*p == ')') {
        offset_end = p;
        break;
      }
    }
    // if the line could be processed, attempt to demangle the symbol
    if (mangled_name && offset_begin && offset_end && 
        mangled_name < offset_begin) {
      *mangled_name++ = '\0';
      *offset_begin++ = '\0';
      *offset_end++ = '\0';

      int status;
      char * real_name = abi::__cxa_demangle(mangled_name, 0, 0, &status);

      // if demangling is successful, output the demangled function name
      if (status == 0) {    
        std::cerr << "[bt]: (" << i << ") " << messages[i] << " : " 
          << real_name << "+" << offset_begin << offset_end 
          << std::endl;

      } else {  // otherwise, output the mangled function name
        std::cerr << "[bt]: (" << i << ") " << messages[i] << " : " 
          << mangled_name << "+" << offset_begin << offset_end 
          << std::endl;
      }
      free(real_name);
    }
    // otherwise, print the whole line
    else {
      std::cerr << "[bt]: (" << i << ") " << messages[i] << std::endl;
    }
  }
  std::cerr << std::endl;
  free(messages);
  exit(EXIT_FAILURE);
}

void WaitForEnter() {
  cout << "Press ENTER to continue...";
  cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

void SetupBackTraceHandler() {
  //signal(SIGSEGV, BackTraceHandler);  // Prints backtrace if seg fault happens.
  struct sigaction sigact;
  sigact.sa_sigaction = crit_err_hdlr;
  sigact.sa_flags = SA_RESTART | SA_SIGINFO;

  if (sigaction(SIGSEGV, &sigact, (struct sigaction *)NULL) != 0) {
    fprintf(stderr, "error setting signal handler for %d (%s)\n",
        SIGSEGV, strsignal(SIGSEGV));
    exit(EXIT_FAILURE);
  }
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
  dst << endl << "timestamp : \"" << timestamp << "\";" << endl;
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

void ReadModel(const string& model_file, config::Model *model) {
  string ext = model_file.substr(model_file.find_last_of('.'));
  if (ext.compare(".pb") == 0) {
    ReadModelBinary(model_file, model);
  } else {
    ReadModelText(model_file, model);
  }
}

void ReadModelText(const string& model_file, config::Model *model) {
  stringstream ss;
  ifstream file(model_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), model)) {
    cerr << "Could not read text proto buffer : " << model_file << endl;
    exit(1);
  }
}

void ReadDataConfig(const string& data_config_file, config::DatasetConfig *data_config) {
  stringstream ss;
  ifstream file(data_config_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), data_config)) {
    cerr << "Could not read text proto buffer : " << data_config_file << endl;
    exit(1);
  }
}

void ReadLayerConfig(const string& layer_config_file, config::Layer *layer_config) {
  stringstream ss;
  ifstream file(layer_config_file.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), layer_config)) {
    cerr << "Could not read text proto buffer : " << layer_config_file << endl;
    exit(1);
  }
}

void WriteModelBinary(const string& output_file, config::Model *model) {
  ofstream out(output_file.c_str());
  model->SerializeToOstream(&out);
  out.close();
}

void ReadModelBinary(const string& input_file, config::Model *model) {
  ifstream in(input_file.c_str());
  model->ParseFromIstream(&in);
  in.close();
}


void WriteHDF5CPU(hid_t file, float* mat, int rows, int cols, const string& name) {
  hid_t dataset, dataspace, datatype;
  hsize_t dimsf[2];
  dimsf[0] = rows;
  dimsf[1] = cols;
  dataspace = H5Screate_simple(2, dimsf, NULL);
  datatype = H5Tcopy(H5T_NATIVE_FLOAT);
  dataset = H5Dcreate(file, name.c_str(), datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat);
  H5Sclose(dataspace);
  H5Tclose(datatype);
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



ImageDisplayer::ImageDisplayer(int width, int height, int num_colors, bool show_separate, const string& title) :
  width(width), height(height), num_colors(num_colors), show_separate(show_separate) {

    /*
  if (show_separate) {
    main_disp = new CImgDisplay(width * num_colors, height);
    cout << "Main disp has size: " << width * num_colors << " " << height << endl;
  } else {
    main_disp = new CImgDisplay(width, height);
    cout << "Main disp has size: " << width << " " << height << endl;
  }
  */
  title_ = title;
  disp = new CImgDisplay();
  main_disp = new CImgDisplay();
  disp->set_title(title.c_str());
  main_disp->set_title(title.c_str());
}

ImageDisplayer::ImageDisplayer() :
  width(0), height(0), num_colors(3), show_separate(false) {
  main_disp = NULL;
  disp = new CImgDisplay();
}

void ImageDisplayer::DisplayImage(float* data, int num_images, int image_id) {
  int num_colors_width = (int)sqrt(num_colors);
  int num_colors_height = (num_colors + num_colors_width - 1) / num_colors_width;
  int display_width = show_separate ? width * num_colors_width: width;
  int display_height = show_separate ? height * num_colors_height: height;
  int display_colors = show_separate ? 1 : num_colors;
  CImg<float> img(display_width, display_height, 1, display_colors);
  img.fill(0);
  float val;
  for (int k = 0; k < num_colors; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        val = data[image_id + num_images * (j + width * (i + k * height))];
        if (show_separate) {
          img(j + (k % num_colors_width) * width, i + (k / num_colors_width) * height, 0, 0) = val;
        } else {
          img(j, i, 0, k) = val;
        }
      }
    }
  }
  img.resize(250, 250);
  main_disp->set_title(title_.c_str());
  img.display(*main_disp);
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
  if (main_disp == NULL) {
    cout << "Image will not be displayed " << endl;
    return;
  }
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
  /*
  float min = data[0];
  float max = min;
  float val;
  for (int i = 1; i < num_filters * size * size * 3; i++) {
    val = data[i];
    if (val < min) {
      min = val;
    }
    if (val > max) {
      max = val;
    }
  }
  //float min = -1.0;
  //float max = 1.0;
  */
  for (int f = 0; f < num_filters; f++) {
    for (int k = 0; k < 3; k++) {
      for (int h = 0; h < size; h++) {
        for (int w = 0; w < size; w++) {
          data_pos = f + num_filters * (w + size * (h + size * k));
          col = w + size * (f % num_filters_w);
          row = h + size * (f / num_filters_w);
          //img(col, row, 0, k) = (unsigned char)(255 * (data[data_pos] - min)) / (max - min);
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
  img.display(*main_disp);
}


/*
void ImageDisplayer::DisplayMathGL(mglGraph& gr) {
  const int width = gr.GetWidth();
  const int height = gr.GetHeight();
  const unsigned char* raster = gr.GetRGB();
  CImg<unsigned char> img(width, height, 1, 3);
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      for (int c = 0; c < 3; c++) {
        img(i, j, 0, c) = raster[(j * width + i) * 3 + c];
      }
    }
  }
  img.display(*disp);
  disp->set_title(title_.c_str());
}


void ImageDisplayer::DisplayWeightStats(float* data, int size) {
  mglGraph gr;
  mglData x(size, data);
  mglData xx=x.Hist(100, -1, 1);
  xx.Norm(0,1);
  gr.SetRanges(-1, 1, 0, 1);
  gr.Box();
  gr.Bars(xx);
  gr.Axis();
  //gr.WriteFrame("sample.png");  // save it
  DisplayMathGL(gr);
}
*/


HDF5Handler::HDF5Handler(const string& hdf5_file) {
  file_ = H5Fopen(hdf5_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  mat_ = NULL;
  batch_index_ = 0;
  case_index_ = 0;
  mat_size_ = 0;
  num_cases_ = 0;
  num_dims_ = 0;
}

HDF5Handler::~HDF5Handler() {
  H5Fclose(file_);
  if (mat_ != NULL) free(mat_);
}

void HDF5Handler::LoadNext(vector<float>& out) {
  if (case_index_ == num_cases_) {
    case_index_ = 0;
    stringstream ss;
    ss << "batch_" << batch_index_;
    string name = ss.str();
    ReadHDF5Shape(file_, name, &num_cases_, &num_dims_);
    int size = num_cases_ * num_dims_;
    if (mat_size_ != size) {
      mat_ = (float*) realloc(mat_, size * sizeof(float));
      mat_size_ = size;
    }
    ReadHDF5CPU(file_, mat_, mat_size_, name);
    batch_index_++;
  }
  for (int i = 0; i < num_dims_; i++) {
    int index = case_index_ + i * num_cases_;
    out.push_back(mat_[index]);
  }
  case_index_++;
}
