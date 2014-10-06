#include "multigpu_convnet.h"
#include <iostream>
#include <tclap/CmdLine.h>
using namespace std;

int main(int argc, char** argv) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif
  try {
    TCLAP::CmdLine cmd("ConvNet Feature Extractor", ' ', "1.0");
    TCLAP::MultiArg<int> board_arg(
        "b", "board", "GPU board(s)", true, "integer");
    TCLAP::ValueArg<std::string> model_file_arg(
        "m", "model", "Model pbtxt file", true, "", "string");
    TCLAP::ValueArg<std::string> fe_file_arg(
        "f", "feature-config", "Feature extraction pbtxt file", true, "", "string");
    
    cmd.add(board_arg);
    cmd.add(model_file_arg);
    cmd.add(fe_file_arg);

    cmd.parse(argc, argv);

    const vector<int>& boards = board_arg.getValue();
    const string& model_file = model_file_arg.getValue();
    const string& fe_file = fe_file_arg.getValue();
    
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
    net->ExtractFeatures(fe_file);
    delete net;
  } catch (TCLAP::ArgException &e)  {
    cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
  }
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
