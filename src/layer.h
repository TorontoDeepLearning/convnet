#ifndef LAYER_H_
#define LAYER_H_
#include "edge.h"
#include <set>

class Layer {
 public:
  Layer(const config::Layer& config);
  ~Layer();

  virtual void AllocateMemory(int imgsize, int batch_size);
  virtual void ApplyActivation(bool train) = 0; 
  virtual void ApplyDerivativeOfActivation() = 0;
  virtual void ComputeDeriv() = 0;
  virtual float GetLoss() = 0;
  virtual float GetLoss2();  // For computing a loss that grad check cares about.

  void ApplyDropout(bool train);
  void ApplyDropoutAtTrainTime();
  void ApplyDropoutAtTestTime();
  void ApplyDerivativeofDropout();
  void AllocateMemoryEdges(int image_size);

  // Methods for prevent race conditions when using multiple GPUs.
  void AccessStateBegin();
  void AccessStateEnd();
  void AccessDerivBegin();
  void AccessDerivEnd();

  Edge* GetIncomingEdge(int index) { return incoming_edge_[index]; }  // TODO:add check for size.
  Matrix& GetState() { return state_;}
  Matrix& GetDeriv() { return deriv_;}
  Matrix& GetData() { return data_;}

  void Display();
  void Display(int image_id);

  void AddIncoming(Edge* e);
  void AddOutgoing(Edge* e);

  const string& GetName() const { return name_; };
  int GetNumChannels() const { return num_channels_; }
  int GetSize() const { return image_size_; }
  bool IsInput() const { return is_input_; }
  bool IsOutput() const { return is_output_; }

  int GetGPUId() const { return gpu_id_; }
  void AllocateMemoryOnOtherGPUs();
  Matrix& GetOtherState(int gpu_id);
  Matrix& GetOtherDeriv(int gpu_id);
  void SyncIncomingState();
  void SyncOutgoingState();
  void SyncIncomingDeriv();
  void SyncOutgoingDeriv();

  static Layer* ChooseLayerClass(const config::Layer& layer_config);

  vector<Edge*> incoming_edge_, outgoing_edge_;
  bool has_incoming_from_same_gpu_, has_outgoing_to_same_gpu_;
  bool has_incoming_from_other_gpus_, has_outgoing_to_other_gpus_;

 protected:
  const string name_;
  const int num_channels_;
  const bool is_input_, is_output_;
  const float dropprob_;
  const bool display_, dropout_scale_up_at_train_time_, gaussian_dropout_;

  // Maximum activation after applying gaussian dropout.
  // This is needed to prevent blow ups due to sampling large values.
  const float max_act_gaussian_dropout_;

  int scale_targets_, image_size_;

  Matrix state_;  // State (activation) of the layer.
  Matrix deriv_;  // Deriv of the objective function w.r.t. the state.
  Matrix data_;   // Data (inputs or targets) associated with this layer.
  Matrix rand_gaussian_;  // We need to store random variates when doing gaussian dropout.
  map<int, Matrix> other_states_; // Layer copies on other gpus.
  map<int, Matrix> other_derivs_; // Layer copies on other gpus.

  ImageDisplayer *img_display_;
  const int gpu_id_;
  set<int> other_incoming_gpu_ids_, other_outgoing_gpu_ids_;
};

class LinearLayer : public Layer {
 public:
  LinearLayer(const config::Layer& config) : Layer(config) {};
  virtual void AllocateMemory(int imgsize, int batch_size);
  virtual void ApplyActivation(bool train);
  virtual void ApplyDerivativeOfActivation();
  virtual void ComputeDeriv();
  virtual float GetLoss();
};

class ReLULayer : public LinearLayer {
 public:
  ReLULayer(const config::Layer& config);
  virtual void ApplyActivation(bool train);
  virtual void ApplyDerivativeOfActivation();
 protected:
  const bool rectify_after_gaussian_dropout_;
};

class SoftmaxLayer : public Layer {
 public:
  SoftmaxLayer(const config::Layer& config) : Layer(config) {};
  virtual void AllocateMemory(int imgsize, int batch_size);
  virtual void ApplyActivation(bool train);
  virtual void ApplyDerivativeOfActivation();
  virtual void ComputeDeriv();
  virtual float GetLoss();
  virtual float GetLoss2();
};

class SoftmaxDistLayer : public SoftmaxLayer {
 public:
  SoftmaxDistLayer(const config::Layer& config) : SoftmaxLayer(config) {};
  virtual void AllocateMemory(int imgsize, int batch_size);
  virtual void ComputeDeriv();
  virtual float GetLoss();

 private:
  Matrix cross_entropy_;
};

class LogisticLayer : public Layer {
 public:
  LogisticLayer(const config::Layer& config) : Layer(config) {};
  virtual void AllocateMemory(int image_size, int batch_size);
  virtual void ApplyActivation(bool train);
  virtual void ApplyDerivativeOfActivation();
  virtual void ComputeDeriv();
  virtual float GetLoss();
};
#endif
