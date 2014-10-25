#include "conv_onetoone_edge.h"
#include <iostream>

ConvOneToOneEdge::ConvOneToOneEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config) {}

void ConvOneToOneEdge::SetImageSize(int image_size_y, int image_size_x) {
  Edge::SetImageSize(image_size_y, image_size_x);
  num_modules_y_ = image_size_y;
  num_modules_x_ = image_size_x;
}

size_t ConvOneToOneEdge::GetParameterMemoryRequirement() {
  return num_output_channels_ * (num_input_channels_ + (has_no_bias_ ? 0 : 1));
}

string ConvOneToOneEdge::GetDescription() {
  stringstream ss;
  ss << name_ << " "
     << " One-to-One Convolutional Kernel: "
     << num_input_channels_ << " : " << num_output_channels_
     << " Layer: " << image_size_y_ << "-" << image_size_x_ << "-"
     << num_input_channels_ << " : " << num_modules_y_ << "-" << num_modules_x_
     << "-" << num_output_channels_;
  return ss.str();
}

void ConvOneToOneEdge::SetMemory(Matrix& p) {
  if (is_tied_) return;
  Edge::SetMemory(p);

  p.Reshape(num_output_channels_, -1);
  p.GetSlice(weights_, 0, num_input_channels_);
  if (!has_no_bias_) {
    p.GetSlice(bias_, num_input_channels_, num_input_channels_ + 1);
    bias_.Reshape(1, -1);
  }
}

void ConvOneToOneEdge::SetGradMemory(Matrix& p) {
  if (is_tied_) return;
  p.Reshape(num_output_channels_, -1);
  p.GetSlice(grad_weights_, 0, num_input_channels_);
  
  weight_optimizer_->AllocateMemory(num_output_channels_, num_input_channels_);

  if (!has_no_bias_) {
    p.GetSlice(grad_bias_, num_input_channels_, num_input_channels_ + 1);
    grad_bias_.Reshape(1, -1);
    bias_optimizer_->AllocateMemory(1, num_output_channels_);
  }
}

void ConvOneToOneEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  int batch_size = input.GetRows();
  input.Reshape(-1, num_input_channels_);
  output.Reshape(-1, num_output_channels_);

  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::Dot(input, w, output, scale_targets, 1, false, true);

  if (!has_no_bias_) {
    Matrix& b = is_tied_? tied_edge_->GetBias() : bias_;
    output.AddRowVec(b);
  }

  input.Reshape(batch_size, -1);
  output.Reshape(batch_size, -1);
}

void ConvOneToOneEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite) {
  int batch_size = input.GetRows();
  deriv_output.Reshape(-1, num_output_channels_);
  deriv_input.Reshape(-1, num_input_channels_);
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::Dot(deriv_output, w, deriv_input, scale_targets, 1);
  deriv_output.Reshape(batch_size, -1);
  deriv_input.Reshape(batch_size, -1);
}

void ConvOneToOneEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  int batch_size = input.GetRows();
  input.Reshape(-1, num_input_channels_);
  deriv_output.Reshape(-1, num_output_channels_);
  
  Matrix& dw = is_tied_ ? tied_edge_->GetGradWeight() : grad_weights_;
  int scale_targets = GetNumGradsReceived() > 0 ? 1 : 0;
  Matrix::Dot(deriv_output, input, dw, scale_targets, scale_gradients_ / batch_size, true, false);

  if (!has_no_bias_) {
    Matrix& db = is_tied_ ? tied_edge_->GetGradBias() : grad_bias_;
    deriv_output.SumRows(db, scale_targets, scale_gradients_ / batch_size);
  }
  input.Reshape(batch_size, -1);
  deriv_output.Reshape(batch_size, -1);
  IncrementNumGradsReceived();
}
