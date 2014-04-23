#include "datahandler.h"
#include "util.h"
#include "layer.h"
#include <chrono>

int main(int argc, char** argv) {
  
  /*
  const int board = atoi(argv[1]);
  Matrix::InitRandom(10);
  Matrix::SetupCUDADevice(board);

  string data_config_file(argv[2]);
  string layer_config_file(argv[3]);

  config::Layer layer_config;
  ReadLayerConfig(layer_config_file, &layer_config);

  DataHandler *dh = DataHandler::ChooseDataHandler(data_config_file);
  Layer input(layer_config);
  input.AllocateMemory(32, 128);
  input.GetState().Set(1);
  input.GetState().Print();
  vector<Layer*> layers(1);
  layers[0] = &input;
  dh->GetBatch(layers);
  input.GetState().Print();
  */

  string filename(argv[1]);
  vector<string> dataset_names;
  dataset_names.push_back("train_data");
  dataset_names.push_back("train_labels");
  srand(11);
  //HDF5RandomMultiAccessor it = HDF5RandomMultiAccessor(filename, dataset_names, 128);
  HDF5MultiIterator it = HDF5MultiIterator(filename, dataset_names);
  int num_dims = it.GetDims(0);
  int num_labels = it.GetDims(1);
  int dataset_size = it.GetDatasetSize();
  cout << "Dataset has " << dataset_size << " images of " << num_dims << " dimensions." << endl;
  long int num_images = 128 * 50;

  unsigned char* data = new unsigned char[num_images * num_dims];
  unsigned int* label = new unsigned int[num_images * num_labels];
  cout << "Loading " << num_images << " images" << endl;
  chrono::time_point<chrono::system_clock> start_t, end_t;
  chrono::duration<double> time_diff;
  start_t = chrono::system_clock::now();
  vector<void*> buf(2);
  for (int i = 0; i < num_images; i++) {
    cout << "\r " << i;
    cout.flush();
    buf[0] = data + i * num_dims; 
    buf[1] = label + i * num_labels; 
    it.GetNext(buf);
  }
  cout << endl;
  end_t = chrono::system_clock::now();
  time_diff = end_t - start_t;
  printf("Time %f \n", time_diff.count());
  /*
  int width = 256;
  int height = 256;
  CImgDisplay main_disp;
  CImg<unsigned char> img(256, 256, 1, 3);
  int num_images_to_show = num_images;
  for (int i = 0; i < num_images_to_show; i++) {
    for (int k = 0; k < 3; k++) {
      for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            img(j, i, 0, k) = data[j + width * (i + k * height)];
        }
      }
    }
    data += num_dims;
    img.display(main_disp);
  }
  data -= num_dims * num_images_to_show;
  */
  delete data;
  delete label;
  return 0;
}
