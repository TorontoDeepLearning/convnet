#ifndef CONV_EDGE_H_
#define CONV_EDGE_H_
#include "edge_with_weight.h"

/** Implements a convolutional edge.*/
class ConvEdge : public EdgeWithWeight {
 public:
  ConvEdge(const config::Edge& edge_config);
  virtual std::string GetDescription();
  virtual void SetMemory(Matrix& p);
  virtual void SetGradMemory(Matrix& p);
  virtual size_t GetParameterMemoryRequirement();
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite, bool train);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void ComputeOuter(Matrix& input, Matrix& deriv_output);

  virtual void SetTiedTo(Edge* e);
  virtual void DisplayWeights();
  virtual void SetImageSize(int image_size_y, int image_size_x, int image_size_t);
  virtual void FOV(int* size, int* sep, int* pad1, int* pad2) const;
 
  ConvDesc GetConvDesc() const { return conv_desc_; }
  int GetPartialSumY() const { return partial_sum_y_; }
  int GetPartialSumX() const { return partial_sum_x_; }
  bool GetSharedBias() const { return shared_bias_; }

 private:
  Matrix grad_weights_partial_sum_;
  ConvDesc conv_desc_;
  int partial_sum_y_, partial_sum_x_;
  bool shared_bias_;
};
#endif
