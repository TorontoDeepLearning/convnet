#include "rgb_to_yuv_edge.h"
using namespace std;

RGBToYUVEdge::RGBToYUVEdge(const config::Edge& edge_config) :
  Edge(edge_config) {}

string RGBToYUVEdge::GetDescription() {
  stringstream ss;
  ss << name_ << " RGB to YUV " << image_size_y_ << "-" << image_size_x_;
  return ss.str();
}

void RGBToYUVEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  Matrix::ConvRGBToYUV(input, output);
}

void RGBToYUVEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                               Matrix& output, Matrix& deriv_input,
                               bool overwrite) {
  cerr << "RGBtoYUV backprop Not implemented." << endl;
  exit(1);
}
