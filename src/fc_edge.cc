#include "fc_edge.h"
#include <iostream>

FCEdge::FCEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config){}

void FCEdge::AllocateMemory(int image_size) {
  EdgeWithWeight::AllocateMemory(image_size);
  cout << name_ << " ";
  printf("Fully connected : %d-%d-%d (%d) : %d\n", image_size, image_size, num_input_channels_,
         image_size * image_size * num_input_channels_, num_output_channels_);

  int weight_input_size = image_size * image_size * num_input_channels_;

  // Weights.
  weights_.AllocateGPUMemory(num_output_channels_, weight_input_size, GetName() + "_weight");

  // Matrix for storing the current gradient.
  grad_weights_.AllocateGPUMemory(weights_.GetRows(), weights_.GetCols(), GetName() + "_grad_weight");

  weight_optimizer_->AllocateMemory(weights_.GetRows(), weights_.GetCols());
  // Matrix for storing gradient history (used for implementing momentum or
  // other accelerated gradient descent schemes).
  //grad_history_weights_.AllocateGPUMemory(weights_.GetRows(), weights_.GetCols());

  bias_.AllocateGPUMemory(1, num_output_channels_, GetName() + "_bias");
  grad_bias_.AllocateGPUMemory(bias_.GetRows(), bias_.GetCols(), GetName() + "_grad_bias");
  bias_optimizer_->AllocateMemory(bias_.GetRows(), bias_.GetCols());
}

void FCEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  ComputeStart(input);
  cudamat *input_mat = input.GetMat(),
          *output_mat = output.GetMat(),
          *w_mat_t = weights_.GetMatTranspose();

  int scale_targets = overwrite ? 0 : 1;
  dot(input_mat, w_mat_t, output_mat, scale_targets, 1);

  if (!has_no_bias_) {
    add_row_vec(output_mat, bias_.GetMat(), output_mat);
  }
  ComputeEnd(output);
}

void FCEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                         Matrix& output, Matrix& deriv_input, bool overwrite) {
  ComputeStart(deriv_output);
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();

  // Deriv w.r.t input of this edge (which is to be computed).
  cudamat* deriv_input_mat = deriv_input.GetMat();
  
  cudamat* w_mat = weights_.GetMat();
  int scale_targets = overwrite ? 0 : 1;
  dot(deriv_output_mat, w_mat, deriv_input_mat, scale_targets, 1);
  ComputeEnd(deriv_input);
}

void FCEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  ComputeStart(deriv_output);
  // Input to this edge.
  cudamat* input_mat = input.GetMat();
  
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();
  cudamat* deriv_output_t_mat = deriv_output.GetMatTranspose();
  
  cudamat* dw_mat = grad_weights_.GetMat();
  cudamat* db_mat = grad_bias_.GetMat();
  int scale_targets = 0;

  const int batch_size = input.GetRows();
  dot(deriv_output_t_mat, input_mat, dw_mat, scale_targets, 1.0 / batch_size);

  Matrix ones;
  Matrix::GetOnes(1, batch_size, ones);
  dot(ones.GetMat(), deriv_output_mat, db_mat, scale_targets, 1.0 / batch_size);
}
