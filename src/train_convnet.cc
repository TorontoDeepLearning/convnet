#include "multigpu_convnet.h"
#include <iostream>
#include <tclap/CmdLine.h>
using namespace std;

int main(int argc, char** argv) {
  try {
    TCLAP::CmdLine cmd("ConvNet", ' ', "1.0");
    TCLAP::MultiArg<int> board_arg(
        "b", "board", "GPU board(s)", true, "integer");
    TCLAP::ValueArg<std::string> model_file_arg(
        "m", "model", "Model pbtxt file", true, "", "string");
    TCLAP::ValueArg<std::string> train_data_file_arg(
        "t", "train", "Training data pbtxt file", true, "", "string");
    TCLAP::ValueArg<std::string> val_data_file_arg(
        "v", "val", "Validation data pbtxt file", false, "", "string");
    
    cmd.add(board_arg);
    cmd.add(model_file_arg);
    cmd.add(train_data_file_arg);
    cmd.add(val_data_file_arg);

    cmd.parse(argc, argv);

    const string& model_file = model_file_arg.getValue();
    const string& val_data_file = val_data_file_arg.getValue();
    const string& train_data_file = train_data_file_arg.getValue();
    const vector<int>& boards = board_arg.getValue();
    
    bool multi_gpu = boards.size() > 1; 
    
    // Setup GPU boards.
    if (multi_gpu) {
      Matrix::SetupCUDADevices(boards);
    } else {
      Matrix::SetupCUDADevice(boards[0]);
    }
    for (const int &b : boards){
      cout << "Using board " << b << endl;
    }

    ConvNet *net = multi_gpu ? new MultiGPUConvNet(model_file) :
                               new ConvNet(model_file);
    if (!val_data_file.empty()) {  // Use a validation set.
      net->SetupDataset(train_data_file, val_data_file);
    } else {
      net->SetupDataset(train_data_file);
    }
    net->AllocateMemory(false);
    net->Train();
    delete net;
  } catch (TCLAP::ArgException &e)  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
  return 0;
}
