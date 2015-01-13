#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#ifdef USE_CUDA
#include "matrix.h"
#else
#include "CPUMatrix.h"
#endif

#include "util.h"

/** Base class for all optimizers.
 */
class Optimizer {
 public:
  Optimizer(const config::Optimizer& optimizer_config);
  virtual ~Optimizer();

  // Allocate any memory needed.
  virtual void AllocateMemory(const int rows, const int cols);
  virtual bool IsAllocated() { return false; }

  // Do the optimizaion. This will update parameter.
  virtual void Optimize(Matrix& gradient, Matrix& parameter) = 0;
  virtual void ReduceLearningRate(float factor);
  virtual void NotifyStart(Matrix& parameter);

  // Load and Save gradient history so that optimization can be restarted if it
  // gets interrupted for some reason.
  virtual void LoadParameters(hid_t file, const string& prefix);
  virtual void SaveParameters(hid_t file, const string& prefix);
  
  // Shared hierarchical priors.
  void ApplySharedPriorGradient(Matrix& gradient, Matrix& parameter);

  static Optimizer* ChooseOptimizer(const config::Optimizer& config);

 protected:
  float GetDecayedEpsilon() const;
  void ApplyConstraints(Matrix& parameter);

  const config::Optimizer::Decay epsilon_decay_type_;
  float epsilon_, minimum_epsilon_, decay_factor_;
  const int epsilon_decay_timescale_, start_optimization_after_;
  const float l2_decay_, weight_norm_limit_, weight_norm_constraint_;
  const bool shared_prior_;
  Matrix shared_prior_gradient_matrix_;
  int step_;
};

/** Stochastic gradient descent.
 * Implements stochastic gradient descent with momentum.
 */
class SGDOptimizer : public Optimizer {
 public:
  SGDOptimizer(const config::Optimizer& optimizer_config);
  virtual void AllocateMemory(const int rows, const int cols);
  virtual void Optimize(Matrix& gradient, Matrix& parameter);
  virtual void LoadParameters(hid_t file, const string& prefix);
  virtual void SaveParameters(hid_t file, const string& prefix);
  virtual bool IsAllocated() { return gradient_history_.GetNumEls() > 0; }
  virtual void NotifyStart(Matrix& parameter);

 protected:
  float GetMomentum() const;

  Matrix gradient_history_;
  const float gradient_clip_;

  // Hyperparams.
  const float initial_momentum_, final_momentum_;
  const int momentum_transition_timescale_;
  const bool nesterov_momentum_;
};

class AdagradSGDOptimizer : public SGDOptimizer {
 public:
  AdagradSGDOptimizer(const config::Optimizer& optimizer_config);
  virtual void AllocateMemory(const int rows, const int cols);
  virtual void Optimize(Matrix& gradient, Matrix& parameter);
  virtual void LoadParameters(hid_t file, const string& prefix);
  virtual void SaveParameters(hid_t file, const string& prefix);
  virtual bool IsAllocated() { return adagrad_history_.GetNumEls() > 0; }

 protected:
  Matrix adagrad_history_;
  const float adagrad_delta_;
};

class RMSPropSGDOptimizer : public SGDOptimizer {
 public:
  RMSPropSGDOptimizer(const config::Optimizer& optimizer_config);
  virtual void AllocateMemory(const int rows, const int cols);
  virtual void Optimize(Matrix& gradient, Matrix& parameter);
  virtual void LoadParameters(hid_t file, const string& prefix);
  virtual void SaveParameters(hid_t file, const string& prefix);
  virtual bool IsAllocated() { return rms_history_.GetNumEls() > 0; }

 protected:
  Matrix rms_history_;
  const float factor_;
};

/** Implmenets LBFGS optimization.
 * This class is under construction.
 */
class LBFGSOptimizer : public Optimizer {
 public:
  LBFGSOptimizer(const config::Optimizer& optimizer_config);
  virtual void AllocateMemory(const int rows, const int cols);
  virtual void Optimize(Matrix& gradient, Matrix& parameter);
  virtual void LoadParameters(hid_t file, const string& prefix);
  virtual void SaveParameters(hid_t file, const string& prefix);
  virtual bool IsAllocated() { return q_.GetNumEls() > 0; }

 protected:
  Matrix q_, last_q_, last_w_;
  const int m_;
  vector<float> rho_, alpha_, beta_;
  vector<Matrix> s_, y_;
  int start_;

};
#endif
