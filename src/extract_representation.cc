#include "convnet.h"
#include <iostream>
using namespace std;


int main(int argc, char** argv) {
  SetupBackTraceHandler();  // Prints back trace in case of seg fault.
  int board = atoi(argv[1]);
  string model_file(argv[2]);
  string data_file(argv[3]);
  string output_file(argv[4]);
  vector<string> layer_names;
  for (int i = 5; i < argc; i++) {
    layer_names.push_back(string(argv[i]));
  }

  Matrix::SetupCUDADevice(board);
  cout << "Using board " << board << endl;

  ConvNet net = ConvNet(model_file);
  net.SetupDataset(data_file);
  net.AllocateMemory(true);
  cout << "Dumping outputs to " << output_file << endl;
  net.DumpOutputs(output_file, layer_names);
  return 0;
}
