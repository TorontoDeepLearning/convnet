#include "fc_edge.h"
#include <iostream>

FCEdge::FCEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config){}

void FCEdge::AllocateMemoryBprop() {
  int input_size = image_size_ * image_size_ * num_input_channels_;
  grad_weights_.AllocateGPUMemory(num_output_channels_, input_size, GetName() + "_grad_weight");
  weight_optimizer_->AllocateMemory(num_output_channels_, input_size);

  if (!has_no_bias_) {
    grad_bias_.AllocateGPUMemory(1, num_output_channels_, GetName() + "_grad_bias");
    bias_optimizer_->AllocateMemory(1, num_output_channels_);
  }
}

void FCEdge::AllocateMemoryFprop() {
  int input_size = image_size_ * image_size_ * num_input_channels_;
  weights_.AllocateGPUMemory(num_output_channels_, input_size, GetName() + "_weight");
  if (!has_no_bias_) {
    bias_.AllocateGPUMemory(1, num_output_channels_, GetName() + "_bias");
  }
}

void FCEdge::AllocateMemory(bool fprop_only) {
  if (is_tied_) return;
  Edge::AllocateMemory(fprop_only);
  cout << name_ << " ";
  printf("Fully connected : %d-%d-%d (%d) : %d\n", image_size_, image_size_,
         num_input_channels_, image_size_ * image_size_ * num_input_channels_,
         num_output_channels_);

  AllocateMemoryFprop();
  if (!fprop_only) AllocateMemoryBprop();
}

void FCEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  cudamat *input_mat = input.GetMat(),
          *output_mat = output.GetMat(),
          *w_mat_t = is_tied_? tied_edge_->GetWeight().GetMatTranspose()
                               : weights_.GetMatTranspose();

  int scale_targets = overwrite ? 0 : 1;
  dot(input_mat, w_mat_t, output_mat, scale_targets, 1);

  if (!has_no_bias_) {
    cudamat* b_mat = is_tied_? tied_edge_->GetBias().GetMat() : bias_.GetMat();
    add_row_vec(output_mat, b_mat, output_mat);
  }
}

void FCEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                         Matrix& output, Matrix& deriv_input, bool overwrite) {
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();

  // Deriv w.r.t input of this edge (which is to be computed).
  cudamat* deriv_input_mat = deriv_input.GetMat();
  
  cudamat* w_mat = is_tied_? tied_edge_->GetWeight().GetMat() : weights_.GetMat();
  int scale_targets = overwrite ? 0 : 1;
  dot(deriv_output_mat, w_mat, deriv_input_mat, scale_targets, 1);
}

void FCEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  // Input to this edge.
  cudamat* input_mat = input.GetMat();
  
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();
  cudamat* deriv_output_t_mat = deriv_output.GetMatTranspose();
  
  cudamat* dw_mat = is_tied_ ? tied_edge_->GetGradWeight().GetMat() : grad_weights_.GetMat();
  int scale_targets = GetNumGradsReceived() > 0 ? 1 : 0;

  const int batch_size = input.GetRows();
  dot(deriv_output_t_mat, input_mat, dw_mat, scale_targets, scale_gradients_ / batch_size);

  if (!has_no_bias_) {
    cudamat* db_mat = is_tied_ ? tied_edge_->GetGradBias().GetMat() : grad_bias_.GetMat();
    Matrix ones;
    Matrix::GetOnes(1, batch_size, ones);
    dot(ones.GetMat(), deriv_output_mat, db_mat, scale_targets, scale_gradients_ / batch_size);
  }
  IncrementNumGradsReceived();
}
