#ifndef FC_EDGE_H_
#define FC_EDGE_H_
#include "edge_with_weight.h"

class FCEdge : public EdgeWithWeight {
 public:
  FCEdge(const config::Edge& edge_config);
  virtual void AllocateMemory(int image_size);
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void ComputeOuter(Matrix& input, Matrix& deriv_output);
};
#endif
