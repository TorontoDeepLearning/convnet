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
  int batchsize = argc > 4 ? atoi(argv[4]) : 128;

  Matrix::SetupCUDADevice(board);
  cout << "Using board " << board << endl;

  GradChecker gcheck = GradChecker(model_file);
  gcheck.SetBatchsize(batchsize);
  gcheck.AllocateMemory(false);
  gcheck.Run(output_file);

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
