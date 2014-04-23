#ifndef MULTIGPU_CONVNET_H_
#define MULTIGPU_CONVNET_H_
#include "convnet.h"
#include <mutex>
#include <condition_variable>

class MultiGPUConvNet : public ConvNet {
 public:
  MultiGPUConvNet(const string& model_file);
  virtual void Train();

 protected:
  void StartFprop();
  void WaitForFprop();
  void NotifyFpropComplete();
  void WaitForFpropStartSignal();
  virtual void Fprop(bool train);
  virtual void TrainOneBatch(vector<float>& error);

  //static mutex fprop_start_mutex_, fprop_finished_mutex_;
  //static condition_variable fprop_finish_cond_var_, fprop_start_cond_var_;
  bool ready_for_fprop_, fprop_finish_, stop_fprop_;

};
#endif
