#ifndef DOWNSAMPLE_EDGE_H_
#define DOWNSAMPLE_EDGE_H_
#include "edge.h"

class DownSampleEdge : public Edge {
 public:
  DownSampleEdge(const config::Edge& edge_config);
  virtual void AllocateMemory(int image_size);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual int GetNumModules() const { return image_size_ / sample_factor_; }

 private:
  const int sample_factor_;
  int image_size_;
};
#endif
