#ifndef CONV_ONETOONE_EDGE_H_
#define CONV_ONETOONE_EDGE_H_
#include "edge_with_weight.h"

/** An edge with one-to-one connectivity over spatial locations.
 */ 
class ConvOneToOneEdge : public EdgeWithWeight {
 public:
  ConvOneToOneEdge(const config::Edge& edge_config);
  virtual string GetDescription();
  virtual void SetMemory(Matrix& p);
  virtual void SetGradMemory(Matrix& p);
  virtual size_t GetParameterMemoryRequirement();
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void ComputeOuter(Matrix& input, Matrix& deriv_output);

  virtual void SetImageSize(int image_size_y, int image_size_x);
};
#endif
