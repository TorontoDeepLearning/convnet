#ifndef MAXPOOL_EDGE_H_
#define MAXPOOL_EDGE_H_
#include "edge.h"

class MaxPoolEdge : public Edge {
 public:
  MaxPoolEdge(const config::Edge& edge_config);
  virtual void AllocateMemory(bool fprop_only);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);

  virtual int GetNumModules() const { return num_modules_; }
  virtual bool RequiresMemoryForDeriv() const { return true; }
  virtual void SetImageSize(int image_size);

 private:
  const int kernel_size_, stride_, padding_;
};
#endif
