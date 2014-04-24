#include "upsample_edge.h"
#include "cudamat_conv.cuh"

UpSampleEdge::UpSampleEdge(const config::Edge& edge_config) :
  Edge(edge_config),
  sample_factor_(edge_config.sample_factor()) {}

void UpSampleEdge::SetImageSize(int image_size) {
  Edge::SetImageSize(image_size);
  num_modules_ = image_size * sample_factor_;
}

void UpSampleEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  ComputeStart(input);
  cudamat *input_mat = input.GetMat(),
          *output_mat = output.GetMat();
  int scale_targets = overwrite ? 0 : 1;
  UpSample(input_mat, output_mat, sample_factor_, image_size_, scale_targets);
  ComputeEnd(output);
}

void UpSampleEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                               Matrix& output, Matrix& deriv_input, bool overwrite) {
  ComputeStart(deriv_output);
  cudamat *deriv_output_mat = deriv_output.GetMat(),
          *deriv_input_mat = deriv_input.GetMat();
  DownSample(deriv_output_mat, deriv_input_mat, sample_factor_,
             sample_factor_ * image_size_);
  ComputeEnd(deriv_input);
}
