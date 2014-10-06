#ifndef FC_EDGE_H_
#define FC_EDGE_H_
#include "edge_with_weight.h"

/** Implements a fully-connected edge.*/
class FCEdge : public EdgeWithWeight {
 public:
  FCEdge(const config::Edge& edge_config);
  virtual string GetDescription();
  virtual void SetMemory(Matrix& p);
  virtual void SetGradMemory(Matrix& p);
  virtual size_t GetParameterMemoryRequirement();
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void ComputeOuter(Matrix& input, Matrix& deriv_output);
};
#endif
