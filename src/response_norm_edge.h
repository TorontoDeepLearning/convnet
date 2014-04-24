#ifndef RESPONSE_NORM_EDGE_H_
#define RESPONSE_NORM_EDGE_H_
#include "edge.h"

class ResponseNormEdge : public Edge {
 public:
  ResponseNormEdge(const config::Edge& edge_config);
  virtual void AllocateMemory(bool fprop_only);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual bool RequiresMemoryForDeriv() const { return true; }
  virtual void SetImageSize(int image_size);

 private:
  Matrix denoms_;
  int num_filters_response_norm_;
  const bool blocked_;
  const float add_scale_, pow_scale_, frac_of_filters_response_norm_;
};
#endif
