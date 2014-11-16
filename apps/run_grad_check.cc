#include "grad_check.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif
  int board = atoi(argv[1]);
  string model_file(argv[2]);
  string output_file(argv[3]);

  Matrix::SetupCUDADevice(board);
  cout << "Using board " << board << endl;

  GradChecker gcheck = GradChecker(model_file);
  gcheck.SetBatchsize(128);
  gcheck.AllocateMemory(false);
  gcheck.Run(output_file);

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
