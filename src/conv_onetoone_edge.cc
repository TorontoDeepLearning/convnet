#include "conv_onetoone_edge.h"
#include <iostream>

ConvOneToOneEdge::ConvOneToOneEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config) {}

void ConvOneToOneEdge::SetImageSize(int image_size) {
  Edge::SetImageSize(image_size);
  num_modules_ = image_size;
}

void ConvOneToOneEdge::AllocateMemory(bool fprop_only) {
  if (is_tied_) return;
  Edge::AllocateMemory(fprop_only);

  cout << name_ << " ";
  printf("One to one convolution: %d - %d ", num_input_channels_, num_output_channels_);
  printf("Layer: %d-%d-%d (%d) ", image_size_, image_size_, num_input_channels_,
         image_size_ * image_size_ * num_input_channels_);
 
  AllocateMemoryFprop();
  if (!fprop_only) AllocateMemoryBprop();

  cout << " Allocated weight " << weights_.GetRows() << " " << weights_.GetCols()
       << " One to One Convolutional" << endl;
}


void ConvOneToOneEdge::AllocateMemoryBprop() {
  if (!is_tied_) {
    grad_weights_.AllocateGPUMemory(num_output_channels_, num_input_channels_);
    weight_optimizer_->AllocateMemory(num_output_channels_, num_input_channels_);
  }

  if (!has_no_bias_ && !is_tied_) {
    grad_bias_.AllocateGPUMemory(1, num_output_channels_);
    bias_optimizer_->AllocateMemory(1, num_output_channels_);
  }
}

void ConvOneToOneEdge::AllocateMemoryFprop() {
  weights_.AllocateGPUMemory(num_output_channels_, num_input_channels_);
  if (!has_no_bias_) {
    bias_.AllocateGPUMemory(1, num_output_channels_);
  }
}

void ConvOneToOneEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  ComputeStart(input);
  int batch_size = input.GetRows();
  input.Reshape(-1, num_input_channels_);
  output.Reshape(-1, num_output_channels_);

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

  input.Reshape(batch_size, -1);
  output.Reshape(batch_size, -1);
  ComputeEnd(output);
}

void ConvOneToOneEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite) {
  ComputeStart(deriv_output);
  int batch_size = input.GetRows();
  deriv_output.Reshape(-1, num_output_channels_);
  deriv_input.Reshape(-1, num_input_channels_);
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();

  // Deriv w.r.t input of this edge (which is to be computed).
  cudamat* deriv_input_mat = deriv_input.GetMat();
  
  cudamat* w_mat = is_tied_? tied_edge_->GetWeight().GetMat() : weights_.GetMat();
  int scale_targets = overwrite ? 0 : 1;
  dot(deriv_output_mat, w_mat, deriv_input_mat, scale_targets, 1);
  deriv_output.Reshape(batch_size, -1);
  deriv_input.Reshape(batch_size, -1);
  ComputeEnd(deriv_input);
}

void ConvOneToOneEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  ComputeStart(deriv_output);
  int batch_size = input.GetRows();
  input.Reshape(-1, num_input_channels_);
  deriv_output.Reshape(-1, num_output_channels_);
  // Input to this edge.
  cudamat* input_mat = input.GetMat();
  
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();
  cudamat* deriv_output_t_mat = deriv_output.GetMatTranspose();
  
  cudamat* dw_mat = is_tied_ ? tied_edge_->GetGradWeight().GetMat() : grad_weights_.GetMat();
  int scale_targets = GetNumGradsReceived() > 0 ? 1 : 0;

  dot(deriv_output_t_mat, input_mat, dw_mat, scale_targets, scale_gradients_ / batch_size);

  if (!has_no_bias_) {
    cudamat* db_mat = is_tied_ ? tied_edge_->GetGradBias().GetMat() : grad_bias_.GetMat();
    Matrix ones;
    Matrix::GetOnes(1, deriv_output.GetRows(), ones);
    dot(ones.GetMat(), deriv_output_mat, db_mat, scale_targets, scale_gradients_ / batch_size);
  }
  input.Reshape(batch_size, -1);
  deriv_output.Reshape(batch_size, -1);
  IncrementNumGradsReceived();
}
