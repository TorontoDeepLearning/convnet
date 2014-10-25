#include "upsample_edge.h"

UpSampleEdge::UpSampleEdge(const config::Edge& edge_config) :
  Edge(edge_config),
  sample_factor_(edge_config.sample_factor()) {}

string UpSampleEdge::GetDescription() {
  stringstream ss;
  ss << name_ << " "
     << " Upsample factor " << sample_factor_
     << " Layer: " << image_size_y_ << "-" << image_size_x_ << "-"
     << num_input_channels_ << " : " << num_modules_y_ << "-" << num_modules_x_
     << "-" << num_output_channels_ << endl;
  return ss.str();
}

void UpSampleEdge::SetImageSize(int image_size_y, int image_size_x) {
  Edge::SetImageSize(image_size_y, image_size_x);
  num_modules_y_ = image_size_y * sample_factor_;
  num_modules_x_ = image_size_x * sample_factor_;
}

void UpSampleEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  int scale_targets = overwrite ? 0 : 1;
  // TODO: rect
  Matrix::ConvUpSample(input, output, sample_factor_, image_size_x_, scale_targets);
}

void UpSampleEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                               Matrix& output, Matrix& deriv_input,
                               bool overwrite) {
  // TODO: rect
  Matrix::ConvDownSample(deriv_output, deriv_input, sample_factor_,
                         sample_factor_ * image_size_x_);
}
