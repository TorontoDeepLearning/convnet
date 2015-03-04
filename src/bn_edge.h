#ifndef BN_EDGE_H_
#define BN_EDGE_H_
#include "edge_with_weight.h"

/** Implements a batch normalization edge.*/
class BNEdge : public EdgeWithWeight {
 public:
  BNEdge(const config::Edge& edge_config);
  void SetImageSize(int image_size_y, int image_size_x, int image_size_t);
  virtual std::string GetDescription();
  virtual void SaveParameters(hid_t file);
  virtual void LoadParameters(hid_t file, const std::string& edge_name);
  virtual void SetMemory(Matrix& p);
  virtual void SetGradMemory(Matrix& p);
  virtual size_t GetParameterMemoryRequirement();
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite, bool train);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
  virtual void ComputeOuter(Matrix& input, Matrix& deriv_output);
 private:
  Matrix mu_, sigma_, batch_mu_, batch_sigma_;
  const float bn_f_, bn_epsilon_;
};
#endif
