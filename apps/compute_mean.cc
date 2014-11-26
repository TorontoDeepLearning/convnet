#include "layer.h"
#include "datahandler.h"
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;
int main(int argc, char** argv) {
  const char *keys =
          "{ board b || GPU board(s): 0 or 012, etc. }"
          "{ data d || Data pbtxt file }"
          "{ layer l || Layer pbtxt file }"
          "{ output  o || Output hdf5 file }";
  CommandLineParser parser(argc, argv, keys);
  string board(parser.get<string>("board"));
  string data_config_file(parser.get<string>("data"));
  string layer_config_file(parser.get<string>("layer"));
  string output_file(parser.get<string>("output"));
  if (board.empty() || data_config_file.empty() || layer_config_file.empty() || output_file.empty()) {
    parser.printMessage();
    return 1;
  }

  vector<int> boards;
  ParseBoardIds(board, boards);
  
  Matrix::SetupCUDADevice(boards[0]);
  cout << "Using board " << boards[0] << endl;

  config::DatasetConfig data_config;
  ReadPbtxt<config::DatasetConfig>(data_config_file, data_config);

  DataHandler* dataset_ = new DataHandler(data_config);
  int batch_size = dataset_->GetBatchSize();
  int dataset_size = dataset_->GetDataSetSize();
  dataset_->AllocateMemory();
  cout << "Data set size " << dataset_size << endl;

  config::Layer layer_config;
  ReadPbtxt<config::Layer>(layer_config_file, layer_config);
  Layer *l = Layer::ChooseLayerClass(layer_config);
  int num_colors = l->GetNumChannels();
  int image_size_y = dataset_->GetImageSizeY(l->GetName());
  int image_size_x = dataset_->GetImageSizeX(l->GetName());
  l->SetSize(image_size_y, image_size_x, 1);
  l->AllocateMemory(batch_size);
  vector<Layer*> layers;
  layers.push_back(l);
  int num_batches = dataset_size / batch_size;
  cout << "Image size " << image_size_y << "x" << image_size_x << endl;

  Matrix& state = l->GetState();
  int num_dims = state.GetCols();
  Matrix mean_batch, mean_image, mean_pixel;
  Matrix var_batch, var_image, var_pixel;
  mean_batch.AllocateGPUMemory(batch_size, num_dims);
  mean_image.AllocateGPUMemory(1, num_dims);
  mean_pixel.AllocateGPUMemory(1, num_colors);
  var_batch.AllocateGPUMemory(batch_size, num_dims);
  var_image.AllocateGPUMemory(1, num_dims);
  var_pixel.AllocateGPUMemory(1, num_colors);
  Matrix pixel_cov, out_p;
  pixel_cov.AllocateGPUMemory(num_colors, num_colors);
  out_p.AllocateGPUMemory(num_colors, num_colors);
  int num_pix = num_dims / num_colors;
  
  mean_batch.Set(0);
  var_batch.Set(0);
  pixel_cov.Set(0);
  cout << "Num batches " << num_batches << endl;
  for (int k = 0; k < num_batches; k++) {
    cout << "\r" << k+1;
    cout.flush();
    dataset_->GetBatch(layers);
    mean_batch.Add(state);
    state.Reshape(-1, num_colors);
    Matrix::Dot(state, state, pixel_cov, 1, 1.0 / (batch_size * num_pix) , true, false);
    state.Reshape(batch_size, -1);
    state.Mult(state);
    var_batch.Add(state);
  }
  dataset_->Sync();
  cout << endl;
  mean_batch.Divide(num_batches);
  var_batch.Divide(num_batches);
  pixel_cov.Divide(num_batches);

  mean_batch.SumRows(mean_image, 0, 1.0 / batch_size);
  mean_image.Reshape(-1, num_colors);
  mean_image.SumRows(mean_pixel, 0, 1.0 / mean_image.GetRows());
  mean_image.Reshape(1, -1);
  Matrix::Dot(mean_pixel, mean_pixel, out_p, 0, 1, true, false);
  pixel_cov.Subtract(out_p, pixel_cov);

  mean_image.CopyToHost();
  mean_pixel.CopyToHost();

  cout << output_file << endl;
  hid_t file = H5Fcreate(output_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  WriteHDF5CPU(file, mean_image.GetHostData(), 1, num_dims, "mean");
  WriteHDF5CPU(file, mean_pixel.GetHostData(), 1, num_colors, "pixel_mean");

  mean_batch.Mult(mean_batch);
  mean_image.Mult(mean_image);
  mean_pixel.Mult(mean_pixel);
  var_batch.Subtract(mean_batch, var_batch);
  var_batch.SumRows(var_image, 0, 1.0 / batch_size);
  mean_batch.SumRows(var_image, 1, 1.0 / batch_size);
  var_image.Subtract(mean_image, var_image);
  
  var_image.Reshape(-1, num_colors);
  mean_image.Reshape(-1, num_colors);
  var_image.SumRows(var_pixel, 0, 1.0 / var_image.GetRows());
  mean_image.SumRows(var_pixel, 1, 1.0 / var_image.GetRows());
  var_pixel.Subtract(mean_pixel, var_pixel);
  var_image.Reshape(1, -1);
  mean_image.Reshape(1, -1);
  
  var_image.Sqrt();
  var_pixel.Sqrt();
  var_image.CopyToHost();
  var_pixel.CopyToHost();
  WriteHDF5CPU(file, var_image.GetHostData(), 1, num_dims, "std");
  WriteHDF5CPU(file, var_pixel.GetHostData(), 1, num_colors, "pixel_std");

  Matrix::Dot(var_pixel, var_pixel, out_p, 0, 1, true, false);
  pixel_cov.Divide(out_p);
  pixel_cov.CopyToHost();
  
  WriteHDF5CPU(file, pixel_cov.GetHostData(), num_colors, num_colors, "pixel_cov");

  H5Fclose(file);
  return 0;
}
