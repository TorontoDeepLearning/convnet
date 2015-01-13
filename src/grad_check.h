#ifndef GRAD_CHECK_H_
#define GRAD_CHECK_H_
#include "convnet.h"

class GradChecker : public ConvNet {
 public:
  GradChecker(const std::string& model_file);
  void Run(const std::string& output_file);
  void ComputeNumericGrad(Matrix& w, float epsilon,
                          std::vector<float>& numerical_grad, int max_params=0);
  bool GradCheck(Matrix& w, std::vector<float>& eps_values, int num_params,
                 const float* analytical_g, const std::string& output_dset,
                 hid_t& output_file);

 protected:
  virtual void GetLoss(std::vector<float>& error);
};
#endif
