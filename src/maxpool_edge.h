#ifndef MAXPOOL_EDGE_H_
#define MAXPOOL_EDGE_H_
#include "edge.h"

/** Implements a Max-pool edge.*/
class MaxPoolEdge : public Edge {
 public:
  MaxPoolEdge(const config::Edge& edge_config);
  virtual std::string GetDescription();
  virtual void SetTiedTo(Edge* e);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite, bool train);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);

  virtual void SetImageSize(int image_size_y, int image_size_x, int image_size_t);
  virtual void FOV(int* size, int* sep, int* pad1, int* pad2) const;

  ConvDesc GetConvDesc() const { return conv_desc_; }

 private:
  ConvDesc conv_desc_;
};
#endif
