#include "avgpool_edge.h"

AvgPoolEdge::AvgPoolEdge(const config::Edge& edge_config) :
  Edge(edge_config),
  conv_desc_(Edge::GetConvDesc(edge_config)) {}

void AvgPoolEdge::SetTiedTo(Edge* e) {
  Edge::SetTiedTo(e);
  AvgPoolEdge* ee = dynamic_cast<AvgPoolEdge*> (e);
  conv_desc_ = ee->GetConvDesc();
}

void AvgPoolEdge::SetImageSize(int image_size_y, int image_size_x) {
  Edge::SetImageSize(image_size_y, image_size_x);
  conv_desc_.num_input_channels = num_input_channels_;
  conv_desc_.num_output_channels = num_output_channels_;
  if (conv_desc_.kernel_size_y <= 0) conv_desc_.kernel_size_y = image_size_y;
  if (conv_desc_.kernel_size_x <= 0) conv_desc_.kernel_size_x = image_size_x;
  Edge::GetNumModules(conv_desc_, image_size_y, image_size_x, num_modules_y_, num_modules_x_);
}

string AvgPoolEdge::GetDescription() {
  stringstream ss;
  ss << name_ << " "
     << " AvgPool Kernel: " << Edge::GetDescription(conv_desc_)
     << " Layer: " << image_size_y_ << "-" << image_size_x_ << " : "
     << num_modules_y_ << "-" << num_modules_x_;
  return ss.str();
}

void AvgPoolEdge::FOV(int* size, int* sep, int* pad1, int* pad2) const {
  /*
  *size = kernel_size_ + stride_ * ((*size) - 1);
  *sep = (*sep) * stride_;
  *pad1 = (*pad1) * stride_ + padding_;
  int k = (image_size_x_ + 2*padding_ - kernel_size_) / stride_;
  int effective_right_pad = k * stride_ - (image_size_x_ + padding_ - kernel_size_);
  *pad2 = (*pad2) * stride_ + effective_right_pad;
  */
}

void AvgPoolEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  if (!overwrite) {
    cerr << " In AvgPoolEdge::ComputeUp() : Looks like some other layer is"
         << " writing to this maxpool layer's output as well."
         << " Are you sure you want to do this ? Not implemented." << endl;
    exit(1);
  }
  Matrix::ConvAvgPool(input, output, conv_desc_);
}

void AvgPoolEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                              Matrix& output, Matrix& deriv_input, bool overwrite) {
  float scale_targets = overwrite ? 0 : 1;
  Matrix::ConvAvgPoolUndo(deriv_output, deriv_input, conv_desc_, scale_targets);
}
