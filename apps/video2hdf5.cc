// Writes video files into an hdf5 dataset as unsigned chars.
// Crops out central 256*256 patch after resizing the image
// so that its shorter side is 256.

#include "video_iterators.h"
#include "hdf5.h"
#include "util.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char ** argv) {
  const char *keys =
          "{ input  i |   | File containing list of images. }"
          "{ output o |   | Output hdf5 file. }"
          "{ width w  |256| Width of the resized image.}"
          "{ height h |256| Height of the resized image.}";
  CommandLineParser parser(argc, argv, keys);
  string file_list(parser.get<string>("input"));
  string output_file(parser.get<string>("output"));
  int image_size_x(parser.get<int>("width"));
  int image_size_y(parser.get<int>("height"));
  if (file_list.empty() || output_file.empty()) {
    parser.printMessage();
    return 1;
  }

  vector<string> filenames;
  readFileList(file_list, filenames);
  RawVideoFileIterator<unsigned char> it = RawVideoFileIterator<unsigned char>(
      filenames, image_size_y, image_size_x, "");
  const int dataset_size = it.GetDataSetSize();
  const int num_dims = image_size_y * image_size_x * 3;
  unsigned char* image_buf = new unsigned char[num_dims];
  hid_t file = H5Fcreate(output_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                         H5P_DEFAULT);
  hsize_t dimsf[2], start[2];


  dimsf[0] = dataset_size;
  dimsf[1] = num_dims;
  hid_t dataspace_handle = H5Screate_simple(2, dimsf, NULL);
  hid_t dataset_handle = H5Dcreate(file, "data", H5T_NATIVE_UCHAR,
                                   dataspace_handle, H5P_DEFAULT, H5P_DEFAULT,
                                   H5P_DEFAULT);

  dimsf[0] = 1;
  dimsf[1] = num_dims;
  start[1] = 0;
  hid_t mem_dataspace = H5Screate_simple(2, dimsf, NULL);
  for (int i = 0; i < dataset_size; i++) {
    cout << "\rImage " << i+1 << " / " << dataset_size;
    cout.flush();
    it.GetNext(image_buf);
    start[0] = i;
    H5Sselect_none(dataspace_handle);
    H5Sselect_hyperslab(dataspace_handle, H5S_SELECT_SET, start, NULL, dimsf,
                        NULL);
    H5Dwrite(dataset_handle, H5T_NATIVE_UCHAR, mem_dataspace, dataspace_handle,
             H5P_DEFAULT, image_buf);
  }
  cout << endl;
  H5Sclose(mem_dataspace);
  H5Fclose(file);
  delete[] image_buf;
  return 0;
}
