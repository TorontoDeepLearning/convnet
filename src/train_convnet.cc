#include "multigpu_convnet.h"
#include <opencv2/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif
  const char *keys =
          "{ board b || GPU board(s): 0 or 012, etc. }"
          "{ model m || Model pbtxt file }"
          "{ train t || Training data pbtxt file }"
          "{ val   v || Validation data pbtxt file }";
  CommandLineParser parser(argc, argv, keys);
  string board(parser.get<string>("board"));
  string model_file(parser.get<string>("model"));
  string train_data_file(parser.get<string>("train"));
  string val_data_file(parser.get<string>("val"));
  if (board.empty() || model_file.empty()) {
    parser.printMessage();
    return 1;
  }

  vector<int> boards;
  ParseBoardIds(board, boards);
  bool multi_gpu = boards.size() > 1; 

  // Setup GPU boards.
  if (multi_gpu) {
    Matrix::SetupCUDADevices(boards);
  } else {
    Matrix::SetupCUDADevice(boards[0]);
  }
  for (const int &b : boards) {
    cout << "Using board " << b << endl;
  }

  ConvNet *net = multi_gpu ? new MultiGPUConvNet(model_file) :
                             new ConvNet(model_file);
  net->SetupDataset(train_data_file, val_data_file);
  net->AllocateMemory(false);
  net->Train();
  delete net;

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
