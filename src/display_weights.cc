#include "util.h"
#include <iostream>

int main(int argc, char** argv) {
  string name("input_yuv:hidden1_conv:weight");
  bool yuv = true;
  hid_t file = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
  int rows, cols;
  ReadHDF5Shape(file, name, &rows, &cols);

  int kernel_size = (int)sqrt(cols / 3), num_input_channels = 3, num_output_channels = rows;
  int size = kernel_size * kernel_size * num_input_channels * num_output_channels;


  cout << "Weight has shape " << rows << " * " << cols << endl;
  cout << "Output filters: " << num_output_channels << endl;
  cout << "Number of colors: " << num_input_channels << endl;
  cout << "Kernel sizes: " << kernel_size << endl;

  float* weights = (float*)malloc(size * sizeof(float));
  ReadHDF5CPU(file, weights, size, name);
  H5Fclose(file);

  int num_filters = num_output_channels;
  int num_filters_w = int(sqrt(num_filters));
  int num_filters_h = num_filters / num_filters_w +  (((num_filters % num_filters_w) > 0) ? 1 : 0);
  int width = 500;
  int height = (width * num_filters_h) / num_filters_w;

  ImageDisplayer img_display = ImageDisplayer(width, height, 3, false, name);
  img_display.DisplayWeights(weights, kernel_size, num_output_channels, 500, yuv);
  getchar();
  free(weights);
  return 0;
}
