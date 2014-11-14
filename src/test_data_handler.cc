#include "util.h"
#include "datahandler.h"
#include "layer.h"

using namespace std;
int main(int argc, char** argv) {
  int board_id = atoi(argv[1]);
  string data_config_file(argv[2]);
  string layer_config_file(argv[3]);
  Matrix::SetupCUDADevice(board_id);
  Matrix::InitRandom(0);
  config::DatasetConfig config;
  ReadPbtxt<config::DatasetConfig>(data_config_file, config);
  DataHandler* dataset = new DataHandler(config);
  int batch_size = dataset->GetBatchSize();
  int datasetsize = dataset->GetDataSetSize();
  dataset->AllocateMemory();
  cout << "Data set size " << datasetsize << endl;

  config::Layer layer_config;
  ReadPbtxt<config::Layer>(layer_config_file, layer_config);
  Layer *l = Layer::ChooseLayerClass(layer_config);
  int image_size_y = dataset->GetImageSizeY(l->GetName());
  int image_size_x = dataset->GetImageSizeX(l->GetName());
  l->SetSize(image_size_y, image_size_x, 1);
  l->AllocateMemory(batch_size);
  vector<Layer*> layers;
  layers.push_back(l);
  int num_batches = datasetsize / batch_size;
  cout << "Image size " << image_size_y << "x" << image_size_x << endl;

  for (int k = 0; k < num_batches; k++) {
    cout << "Batch " << k << endl;
    dataset->GetBatch(layers);
    for (int i = 0; i < batch_size; i++) {
      cout << i << endl;
      l->Display(i);
      WaitForEnter();
    }
  }
  dataset->Sync();
  delete dataset;
  return 0;
}
