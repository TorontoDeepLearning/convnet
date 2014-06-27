#include "multigpu_convnet.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
  vector<int> boards;
  boards.push_back(atoi(argv[1]));
  boards.push_back(atoi(argv[2]));
  string model_file(argv[3]);
  string data_file(argv[4]);
  string output_file(argv[5]);
  vector<string> layer_names;
  for (int i = 6; i < argc; i++) {
    layer_names.push_back(string(argv[i]));
  }

  Matrix::SetupCUDADevices(boards);
  cout << "Using boards " << boards[0] << " and  " << boards[1] << endl;

  ConvNet *net = new MultiGPUConvNet(model_file);
  net->SetupDataset(data_file);
  net->AllocateMemory(true);
  cout << "Dumping outputs to " << output_file << endl;
  net->DumpOutputs(output_file, layer_names);
  delete net;
  return 0;
}
