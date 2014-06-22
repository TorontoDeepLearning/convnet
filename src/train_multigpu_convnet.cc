#include "multigpu_convnet.h"
#include <iostream>
using namespace std;


int main(int argc, char** argv) {
  vector<int> boards;
  boards.push_back(atoi(argv[1]));
  boards.push_back(atoi(argv[2]));
  string model_file(argv[3]);
  string data_file(argv[4]);

  Matrix::SetupCUDADevices(boards);
  cout << "Using boards " << boards[0] << " and  " << boards[1] << endl;

  ConvNet *net = new MultiGPUConvNet(model_file);
  if (argc > 5) {  // Use a validation set.
    string val_data_file(argv[5]);
    net->SetupDataset(data_file, val_data_file);
  } else {
    net->SetupDataset(data_file);
  }
  net->AllocateMemory(false);
  net->Train();
  delete net;
  return 0;
}
