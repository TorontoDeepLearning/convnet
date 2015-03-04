#include "downsample_edge.h"
using namespace std;

DownSampleEdge::DownSampleEdge(const config::Edge& edge_config) :
  Edge(edge_config),
  sample_factor_(edge_config.sample_factor()) {}

string DownSampleEdge::GetDescription() {
  stringstream ss;
  ss << name_ << " "
     << " Downsample factor " << sample_factor_
     << " Layer: " << image_size_y_ << "-" << image_size_x_ << "-"
     << num_input_channels_ << " : " << num_modules_y_ << "-" << num_modules_x_
     << "-" << num_output_channels_ << endl;
  return ss.str();
}

void DownSampleEdge::SetImageSize(int image_size_y, int image_size_x, int image_size_t) {
  Edge::SetImageSize(image_size_y, image_size_x, image_size_t);
  num_modules_y_ = image_size_y * sample_factor_;
  num_modules_x_ = image_size_x * sample_factor_;
  num_modules_t_ = image_size_t;
}

void DownSampleEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite, bool train) {
  Matrix::ConvDownSample(input, output, sample_factor_);
}

void DownSampleEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                                 Matrix& output, Matrix& deriv_input, bool overwrite) {
  int scale_targets = overwrite ? 0 : 1;
  Matrix::ConvUpSample(deriv_output, deriv_input, sample_factor_, scale_targets);
}
