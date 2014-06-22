#include "grad_check.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
  int board = atoi(argv[1]);
  string model_file(argv[2]);
  string data_file(argv[3]);

  Matrix::SetupCUDADevice(board);
  cout << "Using board " << board << endl;

  GradChecker gcheck = GradChecker(model_file);
  gcheck.SetupDataset(data_file);
  gcheck.AllocateMemory(false);
  gcheck.Run();

  return 0;
}
