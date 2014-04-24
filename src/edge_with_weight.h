#ifndef EDGE_WITH_WEIGHT_H_
#define EDGE_WITH_WEIGHT_H_
#include "edge.h"
#include "optimizer.h"

// All edges which have trainable parameters should inherit from this class.
class EdgeWithWeight : public Edge {
 public:
  EdgeWithWeight(const config::Edge& edge_config);
  ~EdgeWithWeight();

  virtual void Initialize();
  virtual void SaveParameters(hid_t file);
  virtual void LoadParameters(hid_t file, bool fprop_only);

  virtual float GetRMSWeight();
  virtual void ReduceLearningRate(float factor);
  virtual bool HasNoParameters() const;
  virtual int GetNumModules() const;
  virtual void DisplayWeights();
  virtual void DisplayWeightStats();

  virtual void UpdateWeights();
  Matrix& GetWeight() { return weights_;}
  Matrix& GetGradWeight() { return grad_weights_;}
  Matrix& GetBias() { return bias_;}
  Matrix& GetGradBias() { return grad_bias_;}

  float GetDecayedEpsilon(float base_epsilon) const;
  float GetMomentum() const;

  virtual void InsertPolyak();
  virtual void BackupCurrent();
  virtual void LoadCurrentOnGPU();
  virtual void LoadPolyakOnGPU();

 protected:
  Matrix weights_, grad_weights_, bias_, grad_bias_;
  Optimizer * const weight_optimizer_;
  Optimizer * const bias_optimizer_;

  vector<Matrix> polyak_weights_, polyak_bias_;
  Matrix weights_backup_, bias_backup_;
  const config::Edge::Initialization initialization_;
  const int polyak_queue_size_;
  int polyak_index_;
  bool polyak_queue_full_;

  // Hyperparams.
  const float init_wt_, init_bias_;

  const bool has_no_bias_;
  int num_grads_received_;
};
#endif
