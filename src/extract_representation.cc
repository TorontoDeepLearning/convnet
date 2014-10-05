/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "multigpu_convnet.h"

#include <opencv2/core.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    const char *keys =
            "{ board          b || GPU board(s): 0 or 012, etc. }"
            "{ model          m || Model pbtxt file }"
            "{ feature-config f || Feature extraction pbtxt file }";
    CommandLineParser parser(argc, argv, keys);
    string board(parser.get<string>("board"));
    string model_file(parser.get<string>("model"));
    string fe_file(parser.get<string>("feature-config"));
    if (board.empty() || model_file.empty() || fe_file.empty())
    {
        parser.printMessage();
        return -1;
    }

    vector<int> boards;
    for (auto b:board)
    {
        string currBoard;
        currBoard.push_back(b);
        boards.push_back(atoi(currBoard.c_str()));
    }
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
    net->ExtractFeatures(fe_file);
    delete net;

    return 0;
}
