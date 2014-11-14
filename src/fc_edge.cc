#include "fc_edge.h"
#include <iostream>

FCEdge::FCEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config){}

size_t FCEdge::GetParameterMemoryRequirement() {
  if (is_tied_) return 0;
  size_t input_size = image_size_y_ * image_size_x_ * image_size_t_ * num_input_channels_;
  return num_output_channels_ * (input_size + (has_no_bias_ ? 0 : 1));
}

string FCEdge::GetDescription() {
  stringstream ss;
  ss << name_ << " Fully Connected :" << image_size_y_ << "-" << image_size_x_
     << "-" << num_input_channels_ << ":" << num_output_channels_;
  return ss.str();
}

void FCEdge::SetMemory(Matrix& p) {
  if (is_tied_) return;
  Edge::SetMemory(p);
  
  size_t input_size = image_size_y_ * image_size_x_ * image_size_t_ * num_input_channels_;
  p.Reshape(num_output_channels_, -1);
  p.GetSlice(weights_, 0, input_size);
  if (!has_no_bias_) {
    p.GetSlice(bias_, input_size, input_size + 1);
    bias_.Reshape(1, -1);
  }
}

void FCEdge::SetGradMemory(Matrix& p) {
  if (is_tied_) return;
  Edge::SetGradMemory(p);
  size_t input_size = image_size_y_ * image_size_x_ * image_size_t_ * num_input_channels_;
  p.Reshape(num_output_channels_, -1);
  p.GetSlice(grad_weights_, 0, input_size);

  weight_optimizer_->AllocateMemory(num_output_channels_, input_size);

  if (!has_no_bias_) {
    p.GetSlice(grad_bias_, input_size, input_size + 1);
    grad_bias_.Reshape(1, -1);
    bias_optimizer_->AllocateMemory(1, num_output_channels_);
  }
}

void FCEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::Dot(input, w, output, scale_targets, 1, false, true);

  if (!has_no_bias_) {
    Matrix& b = is_tied_? tied_edge_->GetBias() : bias_;
    output.AddRowVec(b);
  }
}

void FCEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                         Matrix& output, Matrix& deriv_input, bool overwrite) {
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::Dot(deriv_output, w, deriv_input, scale_targets, 1);
}

void FCEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  Matrix& dw = is_tied_ ? tied_edge_->GetGradWeight() : grad_weights_;
  int scale_targets = GetNumGradsReceived() > 0 ? 1 : 0;
  const int batch_size = input.GetRows();

  Matrix::Dot(deriv_output, input, dw, scale_targets, scale_gradients_ / batch_size, true, false);

  if (!has_no_bias_) {
    Matrix& db = is_tied_ ? tied_edge_->GetGradBias() : grad_bias_;
    deriv_output.SumRows(db, scale_targets, scale_gradients_ / batch_size);
  }
  IncrementNumGradsReceived();
}
