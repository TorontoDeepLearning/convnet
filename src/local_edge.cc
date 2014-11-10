#include "local_edge.h"
#include <iostream>

LocalEdge::LocalEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config),
  conv_desc_(Edge::GetConvDesc(edge_config)) {}

void LocalEdge::SetTiedTo(Edge* e) {
  EdgeWithWeight::SetTiedTo(e);
  LocalEdge* ee = dynamic_cast<LocalEdge*> (e);
  conv_desc_ = ee->GetConvDesc();
}

void LocalEdge::DisplayWeights() {
  if (img_display_ != NULL) {
    weights_.CopyToHost();
    img_display_->DisplayWeights(weights_.GetHostData(), conv_desc_.kernel_size_y, conv_desc_.num_output_channels, 250, false);
  }
}

string LocalEdge::GetDescription() {
  stringstream ss;
  ss << name_ << " "
     << " Local Kernel: " << Edge::GetDescription(conv_desc_)
     << " Layer: " << image_size_y_ << "-" << image_size_x_ << " : "
     << num_modules_y_ << "-" << num_modules_x_ << endl;
  return ss.str();
}

void LocalEdge::SetImageSize(int image_size_y, int image_size_x) {
  Edge::SetImageSize(image_size_y, image_size_x);
  conv_desc_.num_input_channels = num_input_channels_;
  conv_desc_.num_output_channels = num_output_channels_;
  Edge::GetNumModules(conv_desc_, image_size_y, image_size_x, num_modules_y_, num_modules_x_);
}

void LocalEdge::FOV(int* size, int* sep, int* pad1, int* pad2) const {
  /*
  *size = kernel_size_ + stride_ * ((*size) - 1);
  *sep = (*sep) * stride_;
  *pad1 = (*pad1) * stride_ + padding_;
  int k = (image_size_x_ + 2*padding_ - kernel_size_) / stride_;
  int effective_right_pad = k * stride_ - (image_size_x_ + padding_ - kernel_size_);
  *pad2 = (*pad2) * stride_ + effective_right_pad;
  */
}


size_t LocalEdge::GetParameterMemoryRequirement() {
  if (is_tied_) return 0;
 
  int input_size = conv_desc_.kernel_size_x * conv_desc_.kernel_size_y
                   * conv_desc_.num_input_channels * num_modules_y_
                   * num_modules_x_;
  int bias_locs = num_modules_y_ * num_modules_x_;
  return conv_desc_.num_output_channels * (input_size + (has_no_bias_ ? 0 : bias_locs));
}

void LocalEdge::SetMemory(Matrix& p) {
  if (is_tied_) return;
  Edge::SetMemory(p);
 
  int input_size = conv_desc_.kernel_size_x * conv_desc_.kernel_size_y
                   * conv_desc_.num_input_channels * num_modules_y_
                   * num_modules_x_;
  int bias_locs = num_modules_y_ * num_modules_x_;
  p.Reshape(conv_desc_.num_output_channels, -1);
  p.GetSlice(weights_, 0, input_size);
  weights_.SetShape4D(conv_desc_.num_output_channels, conv_desc_.kernel_size_x,
                      conv_desc_.kernel_size_y, conv_desc_.num_input_channels * num_modules_y_ * num_modules_x_);
  if(!has_no_bias_) {
    p.GetSlice(bias_, input_size, input_size + bias_locs);
    bias_.Reshape(1, -1);
  }
 
  if (num_input_channels_ == 3) {
    int num_filters = conv_desc_.num_output_channels;
    int num_filters_w = int(sqrt(num_filters));
    int num_filters_h = num_filters / num_filters_w +  (((num_filters % num_filters_w) > 0) ? 1 : 0);
    int width = 250;
    int height = (width * num_filters_h) / num_filters_w;
    img_display_ = new ImageDisplayer(width, height, 3, false, "weights");
  }
}

void LocalEdge::SetGradMemory(Matrix& p) {
  if (is_tied_) return;
  Edge::SetGradMemory(p);
  int input_size = conv_desc_.kernel_size_x * conv_desc_.kernel_size_y
                   * conv_desc_.num_input_channels * num_modules_y_
                   * num_modules_x_;
  int bias_locs = num_modules_y_ * num_modules_x_;
  // Matrix for storing the current gradient.

  p.Reshape(conv_desc_.num_output_channels, -1);
  p.GetSlice(grad_weights_, 0, input_size);
  weight_optimizer_->AllocateMemory(conv_desc_.num_output_channels, input_size);

  if(!has_no_bias_) {
    p.GetSlice(grad_bias_, input_size, input_size + bias_locs);
    grad_bias_.Reshape(1, -1);
    bias_optimizer_->AllocateMemory(1, conv_desc_.num_output_channels * bias_locs);
  }
}


void LocalEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::LocalUp(input, w, output, conv_desc_, scale_targets);
  if (!has_no_bias_) {
    Matrix& b = is_tied_? tied_edge_->GetBias() : bias_;
    output.AddRowVec(b);
  }
}

void LocalEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite) {
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  int scale_targets = overwrite ? 0 : 1;
  Matrix::LocalDown(deriv_output, w, deriv_input, conv_desc_, scale_targets);
}

void LocalEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  Matrix& dw = is_tied_? tied_edge_->GetGradWeight() : grad_weights_;
  int batch_size = input.GetRows();
  int scale_targets = GetNumGradsReceived() > 0 ? 1 : 0;

  Matrix::LocalOutp(input, deriv_output, dw, conv_desc_, scale_targets,
                    scale_gradients_ / batch_size);

  if (!has_no_bias_) {
    Matrix& db = is_tied_ ? tied_edge_->GetGradBias() : grad_bias_;
    deriv_output.SumRows(db, scale_targets, scale_gradients_ / batch_size);
  }

  IncrementNumGradsReceived();
}
