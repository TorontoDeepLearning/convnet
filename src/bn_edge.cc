#include "bn_edge.h"
#include <iostream>
using namespace std;

BNEdge::BNEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config),
  bn_f_(edge_config.bn_f()),
  bn_epsilon_(edge_config.bn_epsilon()){}

void BNEdge::SetImageSize(int image_size_y, int image_size_x, int image_size_t) {
  Edge::SetImageSize(image_size_y, image_size_x, image_size_t);
  num_modules_y_ = image_size_y;
  num_modules_x_ = image_size_x;
  num_modules_t_ = image_size_t;
}

string BNEdge::GetDescription() {
  stringstream ss;
  ss << name_ << " "
     << " Batch Normalization : "
     << num_input_channels_ << " : " << num_output_channels_
     << " Layer: " << image_size_y_ << "-" << image_size_x_ << "-"
     << num_input_channels_ << " : " << num_modules_y_ << "-" << num_modules_x_
     << "-" << num_output_channels_;
  return ss.str();
}

void BNEdge::SaveParameters(hid_t file) {
  if (is_tied_) return;
  stringstream ss;
  ss << source_node_ << ":" << dest_node_ << ":" << "gamma";
  weights_.WriteHDF5(file, ss.str());
  weight_optimizer_->SaveParameters(file, ss.str());
  ss.str("");
  ss << source_node_ << ":" << dest_node_ << ":" << "beta";
  bias_.WriteHDF5(file, ss.str());
  bias_optimizer_->SaveParameters(file, ss.str());
  ss.str("");
  ss << source_node_ << ":" << dest_node_ << ":" << "mu";
  mu_.WriteHDF5(file, ss.str());
  ss.str("");
  ss << source_node_ << ":" << dest_node_ << ":" << "sigma";
  sigma_.WriteHDF5(file, ss.str());
}

void BNEdge::LoadParameters(hid_t file, const string& edge_name) {
  if (is_tied_) return;
  Matrix::SetDevice(gpu_id_);
  stringstream ss;
  ss << edge_name << ":" << "gamma";
  weights_.ReadHDF5(file, ss.str());
  if (weight_optimizer_->IsAllocated()) {
    weight_optimizer_->LoadParameters(file, ss.str());
  }
  ss.str("");
  ss << edge_name << ":" << "beta";
  bias_.ReadHDF5(file, ss.str());
  if (bias_optimizer_->IsAllocated()) {
    bias_optimizer_->LoadParameters(file, ss.str());
  }
  ss.str("");
  ss << edge_name << ":" << "mu";
  mu_.ReadHDF5(file, ss.str());
  ss.str("");
  ss << edge_name << ":" << "sigma";
  sigma_.ReadHDF5(file, ss.str());
}



size_t BNEdge::GetParameterMemoryRequirement() {
  if (is_tied_) return 0;
  return num_output_channels_ * 4;
}

void BNEdge::SetMemory(Matrix& p) {
  if (is_tied_) return;
  Edge::SetMemory(p);
  p.Reshape(num_output_channels_, -1);
  p.GetSlice(weights_, 0, 1); weights_.Reshape(1, -1);
  p.GetSlice(bias_,    1, 2); bias_.Reshape(1, -1);
  p.GetSlice(mu_,      2, 3); mu_.Reshape(1, -1);
  p.GetSlice(sigma_,   3, 4); sigma_.Reshape(1, -1);
  batch_mu_.AllocateGPUMemory(1, num_output_channels_);
  batch_sigma_.AllocateGPUMemory(1, num_output_channels_);
}

void BNEdge::SetGradMemory(Matrix& p) {
  if (is_tied_) return;
  Edge::SetGradMemory(p);
  p.Reshape(num_output_channels_, -1);
  p.GetSlice(grad_weights_, 0, 1); grad_weights_.Reshape(1, -1);
  p.GetSlice(grad_bias_,    1, 2); grad_bias_.Reshape(1, -1);
  weight_optimizer_->AllocateMemory(1, num_output_channels_);
  bias_optimizer_->AllocateMemory(1, num_output_channels_);
}

void BNEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite, bool train) {
  output.Set(input);
  int batch_size = input.GetRows();
  output.Reshape(-1, num_output_channels_);
  if (train) {
    // Subtract mean.
    output.SumRows(batch_mu_, 0, 1.0f / batch_size);
    output.AddRowVec(batch_mu_, -1);

    // Divide by std dev.
    output.SqSumAxis(batch_sigma_, 0, 1.0f / (batch_size - 1), 0);
    batch_sigma_.Add(bn_epsilon_);
    batch_sigma_.Sqrt();
    output.DivideByRowVec(batch_sigma_);

    // Update the running averages.
    mu_.Mult(bn_f_);
    mu_.Add(batch_mu_, 1-bn_f_);
    sigma_.Mult(bn_f_);
    sigma_.Add(batch_sigma_, 1-bn_f_);
  } else {
    output.AddRowVec(mu_, -1);
    output.DivideByRowVec(sigma_);
  }
  Matrix& weights = is_tied_? tied_edge_->GetWeight() : weights_;
  Matrix& bias    = is_tied_? tied_edge_->GetBias()   : bias_;
  output.MultByRowVec(weights);
  output.AddRowVec(bias, 1);
  output.Reshape(batch_size, -1);
}

void BNEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                         Matrix& output, Matrix& deriv_input, bool overwrite) {
  int batch_size = input.GetRows();
  deriv_output.Reshape(-1, num_output_channels_);
  deriv_input.Reshape(-1, num_input_channels_);
  input.Reshape(-1, num_input_channels_);
  Matrix& weights = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::BNBprop(deriv_output, input, weights, batch_mu_, batch_sigma_, deriv_input, scale_targets);
  input.Reshape(batch_size, -1);
  deriv_input.Reshape(batch_size, -1);
  deriv_output.Reshape(batch_size, -1);
}

void BNEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  int batch_size = input.GetRows();
  deriv_output.Reshape(-1, num_output_channels_);
  input.Reshape(-1, num_input_channels_);
  Matrix& dw = is_tied_ ? tied_edge_->GetGradWeight() : grad_weights_;
  Matrix& db = is_tied_ ? tied_edge_->GetGradBias() : grad_bias_;
  Matrix::BNGrad(deriv_output, input, batch_mu_, batch_sigma_, dw, db);
  dw.Divide(batch_size);
  db.Divide(batch_size);
  IncrementNumGradsReceived();
  input.Reshape(batch_size, -1);
  deriv_output.Reshape(batch_size, -1);
}
