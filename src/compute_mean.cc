#include "datahandler.h"
#include "util.h"

int main(int argc, char** argv) {
 
  string filename(argv[1]);
  string output_file(argv[2]);
  string dataset_name(argv[3]);
  HDF5Iterator it = HDF5Iterator(filename, dataset_name);
  int num_dims = it.GetDims();
  int dataset_size = it.GetDatasetSize();
  cout << "Dataset has " << dataset_size << " images of " << num_dims << " dimensions." << endl;
  long int num_images = dataset_size;

  unsigned char* data = new unsigned char[num_dims];
  double* mean = new double[num_dims];
  double* std = new double[num_dims];
  for (int i = 0; i < num_dims; i++) mean[i] = 0;
  cout << "Loading " << num_images << " images" << endl;
  for (int i = 0; i < num_images && i < 10000; i++) {
    cout << "\r " << i;
    cout.flush();
    it.GetNext(data);
    for (int j = 0; j < num_dims; j++) mean[j] = (mean[j] * i) / (i+1) + ((double)data[j]) / (i+1);
    for (int j = 0; j < num_dims; j++) std[j] = (std[j] * i) / (i+1) + (((double)data[j])*data[j]) / (i+1);
  }
  
  float* mean2 = new float[num_dims];
  float* std2 = new float[num_dims];
  for (int j = 0; j < num_dims; j++) mean2[j] = (float)mean[j];
  for (int j = 0; j < num_dims; j++) std2[j] = (float)sqrt(std[j] - mean[j] * mean[j]);
  hid_t file = H5Fcreate(output_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  WriteHDF5CPU(file, mean2, 1, num_dims, "mean");
  WriteHDF5CPU(file, std2, 1, num_dims, "std");

  if (num_dims % 3 == 0) {
    int num_loc = num_dims / 3;
    double mean_rgb[] = { 0., 0., 0.};
    double std_rgb[] = { 0., 0., 0.};
    float mean2_rgb[3];
    float std2_rgb[3];
    for (int i = 0; i < 3; i++) {
      for (int j = i * num_loc; j < (i+1) * num_loc; j++) mean_rgb[i] += mean[j];
      for (int j = i * num_loc; j < (i+1) * num_loc; j++) std_rgb[i] += std[j];
      mean_rgb[i] /= num_loc;
      std_rgb[i] /= num_loc;
      mean2_rgb[i] = (float)mean_rgb[i];
      std2_rgb[i] = (float)sqrt(std_rgb[i] - mean_rgb[i] * mean_rgb[i]);
    }
    WriteHDF5CPU(file, mean2_rgb, 1, 3, "pixel_mean");
    WriteHDF5CPU(file, std2_rgb, 1, 3, "pixel_std");
  }
  H5Fclose(file);

  delete std2;
  delete mean2;
  delete std;
  delete mean;
  delete data;

  return 0;
}
