#ifndef UPSAMPLE_EDGE_H_
#define UPSAMPLE_EDGE_H_
#include "edge.h"

/** Implements an up-sampling edge.
 * Only integer up-sampling factors are supported.
 */ 
class UpSampleEdge : public Edge {
 public:
  UpSampleEdge(const config::Edge& edge_config);
  virtual string GetDescription();
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void SetImageSize(int image_size_y, int image_size_x);

 private:
  const int sample_factor_;
};
#endif
