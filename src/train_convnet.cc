#include "convnet.h"
#include <iostream>
using namespace std;


int main(int argc, char** argv) {
  SetupBackTraceHandler();  // Prints back trace in case of seg fault.
  int board = atoi(argv[1]);
  string model_file(argv[2]);
  string data_file(argv[3]);

  Matrix::SetupCUDADevice(board);
  //Matrix::SetupCUDADevices(board);
  cout << "Using board " << board << endl;

  ConvNet net = ConvNet(model_file);
  cout << "Setting up dataset" << endl;
  if (argc > 4) {  // Use a validation set.
    string val_data_file(argv[4]);
    net.SetupDataset(data_file, val_data_file);
  } else {
    net.SetupDataset(data_file);
  }
  cout << "Calling train" << endl;
  net.Train();
  return 0;
}
