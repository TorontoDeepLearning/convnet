#ifndef LAYER_H_
#define LAYER_H_
#include "edge.h"
#include "loss_functions.h"
#include <set>

/** The base class for all layers.
 * Each layer has a state_ and deriv_.
 */ 
class Layer {
 public:
  /** Instantiate a layer from config. */ 
  Layer(const config::Layer& config);
  virtual ~Layer();

  /** Allocate memory for storing the state and derivative at this layer.
   * @param batch_size The mini-batch size.
   */ 
  virtual void AllocateMemory(int batch_size);

  /** Apply the activation function.
   * Derived classes must implement this. This method applies the activation
   * function to the state_ and overwrites it.
   */ 
  virtual void ApplyActivation() = 0;

  /** Apply the derivative of the activation.
   * Derived classes must implement this. Computes the derivative w.r.t the
   * inputs to this layer from the derivative w.r.t the outputs of this layer.
   * Applies the derivative of the activation function to deriv_ and overwrites
   * it.
   */ 
  virtual void ApplyDerivativeOfActivation() = 0;

  /** Compute derivative of loss function.
   * This is applicable only if this layer is an output layer.
   */ 
  void ComputeDeriv();

  /** Compute the performance metric that is displayed during training.
   * This is applicable only if this layer is an output layer.
   */ 
  float GetPerformanceMetric();

  /** Compute the value of the actual loss function.
   * This is applicable only if this layer is an output layer.
   */ 
  float GetLoss();

  /** Apply dropout to this layer.
   * @param train If train is true, drop units stochastically,
   * else use all the units.
   */ 
  virtual void ApplyDropout(bool train);

  /** Apply derivative of dropout.
   * This method scales the derivative to compensate for dropout.
   */ 
  virtual void ApplyDerivativeofDropout();

  // Methods for preventing race conditions when using multiple GPUs.
  void AccessStateBegin();
  void AccessStateEnd();
  void AccessDerivBegin();
  void AccessDerivEnd();

  /** Returns the incoming edge by index. */
  Edge* GetIncomingEdge(int index) { return incoming_edge_[index]; }  // TODO:add check for size.

  /** Returns a reference to the state of the layer.*/
  Matrix& GetState();
  Matrix& GetState(const string& slice);

  /** Returns a reference to the deriv at this layer.*/
  Matrix& GetDeriv();
  Matrix& GetDeriv(const string& slice);
  
  /** Returns a reference to the data at this layer.*/
  Matrix& GetData() { return data_;}

  void Display();
  void Display(int image_id);

  /** Add an incoming edge to this layer.*/
  void AddIncoming(Edge* e);

  /** Add an outgoing edge from this layer.*/
  void AddOutgoing(Edge* e);

  const string& GetName() const { return name_; }
  int GetNumChannels() const { return num_channels_; }
  int GetNumChannels(const string& slice) const;
  int GetSizeY() const { return image_size_y_; }
  int GetSizeX() const { return image_size_x_; }
  int GetSizeT() const { return image_size_t_; }
  bool IsInput() const { return is_input_; }
  bool IsOutput() const { return is_output_; }

  void SetSize(int image_size_x, int image_size_y, int image_size_t);
  int GetGPUId() const { return gpu_id_; }
  void AllocateMemoryOnOtherGPUs();
  Matrix& GetOtherState(int gpu_id);
  Matrix& GetOtherDeriv(int gpu_id);

  void AccumulateState();
  void AccumulateDeriv();
  void BroadcastState();
  void BroadcastDeriv();
  void CopyStateToGPU(int dest_gpu);
  void CopyDerivToGPU(int dest_gpu);
  void ResetStateCopies();
  void ResetDerivCopies();

  static Layer* ChooseLayerClass(const config::Layer& layer_config);

  vector<Edge*> incoming_edge_, outgoing_edge_;
  bool has_incoming_from_same_gpu_, has_outgoing_to_same_gpu_;
  bool has_incoming_from_other_gpus_, has_outgoing_to_other_gpus_;

  bool AddOrOverwriteState(const string& slice);
  bool AddOrOverwriteDeriv(const string& slice);
  void ResetAddOrOverwrite();
  bool HasTiedData() const;
  const string& GetTiedDataLayerName() const;

 protected:
  void ApplyDropoutAtTrainTime();
  void ApplyDropoutAtTestTime();
  void SetupSlices();

  const string name_;
  int num_channels_;
  bool is_input_, is_output_;
  const float dropprob_;
  const bool display_, dropout_scale_up_at_train_time_, gaussian_dropout_;

  // Maximum activation after applying gaussian dropout.
  // This is needed to prevent blow ups due to sampling large values.
  const float max_act_gaussian_dropout_;

  int scale_targets_, image_size_y_, image_size_x_, image_size_t_;

  Matrix state_;  /** State (activation) of the layer. */
  Matrix deriv_;  /** Deriv of the loss function w.r.t. the state. */
  Matrix data_;   /** Data (targets) associated with this layer. */
  Matrix dropout_noise_;  /** If we need to store random variates when doing dropout. */
  map<int, Matrix> other_states_; /** Copies of this layer's state on other gpus.*/
  map<int, Matrix> other_derivs_; /** Copies of this layer's deriv on other gpus.*/
  map<int, bool> state_copied_;
  map<int, bool> deriv_copied_;
  ImageDisplayer *img_display_;
  const int gpu_id_;
  set<int> other_incoming_gpu_ids_, other_outgoing_gpu_ids_;
  map<string, Matrix> state_slices_, deriv_slices_;
  map<string, int> slice_channels_;
  map<string, bool> add_or_overwrite_state_, add_or_overwrite_deriv_;
  bool store_dropout_noise_;
  LossFunction *loss_, *performance_;
  config::Layer::LossFunction loss_function_, performance_metric_;
  const float loss_function_weight_;
  const bool has_tied_data_;
  const string tied_data_layer_name_;
};

/** Implements a layer with a linear activation function.*/
class LinearLayer : public Layer {
 public:
  LinearLayer(const config::Layer& config);
  virtual void AllocateMemory(int batch_size);
  virtual void ApplyActivation();
  virtual void ApplyDerivativeOfActivation();
};

/** Implements a layer with a rectified linear activation function.*/
class ReLULayer : public LinearLayer {
 public:
  ReLULayer(const config::Layer& config);
  virtual void ApplyActivation();
  virtual void ApplyDerivativeOfActivation();
  virtual void ApplyDropout(bool train);
 protected:
  const bool rectify_after_gaussian_dropout_;
};

/** Implements a layer with a softmax activation function.
 * This must be an output layer. The target must be one of K choices.
 */
class SoftmaxLayer : public Layer {
 public:
  SoftmaxLayer(const config::Layer& config) : Layer(config) {};
  virtual void AllocateMemory(int batch_size);
  virtual void ApplyActivation();
  virtual void ApplyDerivativeOfActivation();
};

/** Implements a layer with a softmax activation function.
 * This must be an output layer.
 * The target must be a distribution over K choices.
 */
class SoftmaxDistLayer : public SoftmaxLayer {
 public:
  SoftmaxDistLayer(const config::Layer& config) : SoftmaxLayer(config) {};
  virtual void AllocateMemory(int batch_size);

 private:
  Matrix cross_entropy_;
};

/** Implements a layer with a logistic activation function.
 */ 
class LogisticLayer : public Layer {
 public:
  LogisticLayer(const config::Layer& config);
  virtual void AllocateMemory(int batch_size);
  virtual void ApplyActivation();
  virtual void ApplyDerivativeOfActivation();
};
#endif
