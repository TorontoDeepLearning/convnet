#include "response_norm_edge.h"

ResponseNormEdge::ResponseNormEdge(const config::Edge& edge_config) :
  Edge(edge_config),
  num_filters_response_norm_(0),
  blocked_(edge_config.response_norm_in_blocks()),
  add_scale_(edge_config.add_scale()),
  pow_scale_(edge_config.pow_scale()),
  frac_of_filters_response_norm_(edge_config.frac_of_filters_response_norm()){}

void ResponseNormEdge::SetTiedTo(Edge* e) {
  Edge::SetTiedTo(e);
  ResponseNormEdge* ee = dynamic_cast<ResponseNormEdge*> (e);
  blocked_ = ee->Blocked();
  add_scale_ = ee->AddScale();
  pow_scale_ = ee->PowScale();
  frac_of_filters_response_norm_ = ee->FracOfFilters();
}

string ResponseNormEdge::GetDescription() {
  stringstream ss;
  ss << name_ << " Response norm crossmap : " <<
    "num_filters = " << num_filters_response_norm_ << " "
    << image_size_y_ << "-" << image_size_x_ << "-" << num_input_channels_ << ":"
    << num_modules_y_ << "-" << num_modules_x_ << "-" << num_output_channels_;
  return ss.str();
}

void ResponseNormEdge::SetImageSize(int image_size_y, int image_size_x, int image_size_t) {
  Edge::SetImageSize(image_size_y, image_size_x, image_size_t);
  num_modules_y_ = image_size_y;
  num_modules_x_ = image_size_x;
  num_modules_t_ = image_size_t;
  num_filters_response_norm_ = (int)(frac_of_filters_response_norm_
                                     * num_input_channels_);
}

void ResponseNormEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite){
  Matrix::ConvResponseNormCrossMap(
      input, output, num_input_channels_, num_filters_response_norm_,
      add_scale_, pow_scale_, blocked_);
}

void ResponseNormEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                                   Matrix& output, Matrix& deriv_input,
                                   bool overwrite) {
  // OVERWRITES output_mat
  Matrix::ConvResponseNormCrossMapUndo(
      deriv_output, input, output, deriv_input, num_input_channels_,
      num_filters_response_norm_, add_scale_, pow_scale_, blocked_);
}
