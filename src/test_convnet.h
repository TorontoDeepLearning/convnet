#ifndef GRAD_CHECK_H_
#define GRAD_CHECK_H_

#include "convnet.h"

class GradChecker : public ConvNet {
 public:
  GradChecker(const string& model_file);
  void Run();
  void Run(Matrix& w, Matrix& dLbydw_analytical, float epsilon, const vector<float>& analytical_error); 

 protected:
  virtual void GetLoss(vector<float>& error);
};

#endif
