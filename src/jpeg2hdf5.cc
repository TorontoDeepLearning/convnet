// Writes jpeg files into an hdf5 dataset as unsigned chars.
// Crops out central 256*256 patch after resizing the image
// so that its shorter side is 256.
#include "image_iterators.h"
#include "hdf5.h"
#include <string>
#include <iostream>
#include <tclap/CmdLine.h>
using namespace std;

int main(int argc, char ** argv) {
  try {
    TCLAP::CmdLine cmd("jpeg2hdf5", ' ', "1.0");
    TCLAP::ValueArg<string> file_list_arg(
        "i", "input", "File containing list of images.", true, "", "string");
    TCLAP::ValueArg<string> output_file_arg(
        "o", "output", "Output hdf5 file.", true, "", "string");
    TCLAP::ValueArg<int> image_size_arg(
        "c", "crop", "After resize, crop a central patch of this size. Default=256.",
        false, 256, "integer");
    TCLAP::ValueArg<int> big_image_size_arg(
        "s", "resize", "Resize each image so that the shorter side is equal to this size. Default=256.",
        false, 256, "integer");
    
    cmd.add(file_list_arg);
    cmd.add(output_file_arg);
    cmd.add(image_size_arg);
    cmd.add(big_image_size_arg);

    cmd.parse(argc, argv);

    const string& file_list = file_list_arg.getValue();
    const string& output_file = output_file_arg.getValue();
    const int image_size = image_size_arg.getValue();
    const int big_image_size = big_image_size_arg.getValue();
    if (image_size > big_image_size) {
     cerr << "Crop size cannot be bigger than the resized image." << endl;
     exit(1);
    }

    RawImageFileIterator<unsigned char> it = RawImageFileIterator<unsigned char>(
        file_list, image_size, big_image_size, false, false, false);
    const int dataset_size = it.GetDataSetSize();
    const int num_dims = image_size * image_size * 3;
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
    delete image_buf;
  } catch (TCLAP::ArgException &e)  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
