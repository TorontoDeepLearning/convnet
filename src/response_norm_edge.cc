#include "response_norm_edge.h"
#include "cudamat_conv.cuh"

ResponseNormEdge::ResponseNormEdge(const config::Edge& edge_config) :
  Edge(edge_config),
  num_filters_response_norm_(0),
  blocked_(edge_config.response_norm_in_blocks()),
  add_scale_(edge_config.add_scale()),
  pow_scale_(edge_config.pow_scale()),
  frac_of_filters_response_norm_(edge_config.frac_of_filters_response_norm()){}

void ResponseNormEdge::AllocateMemory(int image_size) {
  num_modules_ = image_size;
  num_filters_response_norm_ = (int)(frac_of_filters_response_norm_ * num_input_channels_);
  // There are memory requirements for this edge but the memory size is
  // batchsize dependent. I want to make edges as batch size independent as
  // possible. Therefore, memory allocation is done in ComputeUp.
}

void ResponseNormEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  ComputeStart(input);
  cudamat* input_mat = input.GetMat();
  cudamat* output_mat = output.GetMat();
  if (denoms_.GetNumEls() != input.GetNumEls()) {
    denoms_.AllocateGPUMemory(input.GetRows(), input.GetCols());
  }
  cudamat* denoms_mat = denoms_.GetMat();
  ResponseNormCrossMap(input_mat, denoms_mat, output_mat, num_input_channels_,
                       num_filters_response_norm_, add_scale_, pow_scale_,
                       blocked_);
  ComputeEnd(output);
}

void ResponseNormEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                                   Matrix& output, Matrix& deriv_input, bool overwrite) {
  ComputeStart(deriv_output);
  cudamat* input_mat = input.GetMat();
  cudamat* output_mat = output.GetMat();

  cudamat* denoms_mat = denoms_.GetMat();
  
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();

  // Deriv w.r.t input of this edge (which is to be computed).
  cudamat* deriv_input_mat = deriv_input.GetMat();

  // OVERWRITES output_mat
  ResponseNormCrossMapUndo(deriv_output_mat, denoms_mat, input_mat, output_mat,
                           deriv_input_mat, num_input_channels_,
                           num_filters_response_norm_, add_scale_, pow_scale_,
                           blocked_);
  ComputeEnd(deriv_input);
}
