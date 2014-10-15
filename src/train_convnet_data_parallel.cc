#include "multigpu_convnet.h"
#include <opencv2/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
  int myID, num_processes_;
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myID);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes_);
#else
  myID = 0;
  num_processes_ = 1;
#endif
  cout << myID << " " << num_processes_ << endl;

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
  if (board.empty() || model_file.empty() || train_data_file.empty()) {
    parser.printMessage();
    return 1;
  }

  vector<int> boards;
  ParseBoardIds(board, boards);
  
  // Setup GPU boards.
  Matrix::SetupCUDADevice(boards[myID]);
  cout << "Using board " << boards[myID] << endl;

  ConvNet *net = new ConvNet(model_file);
  if (!val_data_file.empty()) {  // Use a validation set.
    net->SetupDataset(train_data_file, val_data_file);
  } else {
    net->SetupDataset(train_data_file);
  }
  net->AllocateMemory(false);
  net->Train();
  delete net;

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
