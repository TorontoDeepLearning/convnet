#ifndef RESPONSE_NORM_EDGE_H_
#define RESPONSE_NORM_EDGE_H_
#include "edge.h"

/** Response Normalization across filters at the same location.
 */
class ResponseNormEdge : public Edge {
 public:
  ResponseNormEdge(const config::Edge& edge_config);
  virtual std::string GetDescription();
  virtual void SetTiedTo(Edge* e);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite, bool train);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void SetImageSize(int image_size_y, int image_size_x, int image_size_t);

  bool Blocked() const { return blocked_; }
  float AddScale() const { return add_scale_; }
  float PowScale() const { return pow_scale_; }
  float FracOfFilters() const { return frac_of_filters_response_norm_; }

 private:
  int num_filters_response_norm_;
  bool blocked_;
  float add_scale_, pow_scale_, frac_of_filters_response_norm_;
};
#endif
