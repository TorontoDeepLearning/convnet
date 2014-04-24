#ifndef CONV_EDGE_H_
#define CONV_EDGE_H_
#include "edge_with_weight.h"

class ConvEdge : public EdgeWithWeight {
 public:
  ConvEdge(const config::Edge& edge_config);
  virtual void AllocateMemory(bool fprop_only);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void ComputeOuter(Matrix& input, Matrix& deriv_output);

  virtual void SetTiedTo(Edge* e);
  virtual void DisplayWeights();
  virtual int GetNumModules() const { return num_modules_; }
  virtual void SetImageSize(int image_size);
 
 private:
  void AllocateMemoryBprop();
  void AllocateMemoryFprop();

  Matrix grad_weights_partial_sum_;
  const int kernel_size_, stride_, padding_, partial_sum_;
  const bool shared_bias_;
};
#endif
