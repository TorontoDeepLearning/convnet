#ifndef MULTIGPU_CONVNET_H_
#define MULTIGPU_CONVNET_H_
#include "convnet.h"

class MultiGPUConvNet : public ConvNet {
 public:
  MultiGPUConvNet(const std::string& model_file);

 protected:
  virtual void Fprop(bool train);
  virtual void ComputeDeriv();
  virtual void GetLoss(std::vector<float>& error);
  virtual void Bprop(bool update_weights);

  bool broadcast_;
};
#endif
