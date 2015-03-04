#include "layer.h"
#include <iostream>
#include <sstream>
#include <set>
using namespace std;

Layer* Layer::ChooseLayerClass(const config::Layer& config) {
  Layer* l = NULL;
  switch (config.activation()) {
    case config::Layer::LINEAR :
      l = new LinearLayer(config);
      break;
    case config::Layer::LOGISTIC :
      l = new LogisticLayer(config);
      break;
    case config::Layer::RECTIFIED_LINEAR :
      l = new ReLULayer(config);
      break;
    case config::Layer::SOFTMAX :
      l = new SoftmaxLayer(config);
      break;
    case config::Layer::SOFTMAX_DIST :
      l = new SoftmaxDistLayer(config);
      break;
    default:
      cerr << "Undefined layer type." << endl;
      exit(1);
  }
  return l;
}

Layer::Layer(const config::Layer& config) :
  has_incoming_from_same_gpu_(false),
  has_outgoing_to_same_gpu_(false),
  has_incoming_from_other_gpus_(false),
  has_outgoing_to_other_gpus_(false),
  name_(config.name()),
  num_channels_(config.num_channels()),
  is_input_(true),
  is_output_(true),
  dropprob_(config.dropprob()),
  display_(config.display()),
  dropout_scale_up_at_train_time_(true),
  gaussian_dropout_(config.gaussian_dropout()),
  max_act_gaussian_dropout_(config.max_act_gaussian_dropout()),
  scale_targets_(0),
  image_size_y_(config.image_size_y()),
  image_size_x_(config.image_size_x()),
  image_size_t_(config.image_size_t()),
  img_display_(NULL),
  gpu_id_(config.gpu_id()),
  store_dropout_noise_(dropprob_ > 0),
  loss_(NULL),
  performance_(NULL),
  loss_function_(config.loss_function()),
  performance_metric_(config.performance_metric()),
  loss_function_weight_(config.loss_function_weight()),
  has_tied_data_(!config.tied_data().empty()),
  tied_data_layer_name_(config.tied_data()) {

  add_or_overwrite_state_[""] = true;
  add_or_overwrite_deriv_[""] = true;
  for (const config::LayerSlice& s:config.layer_slice()) {
    slice_channels_[s.name()] = s.num_channels();
    num_channels_ += s.num_channels();
    add_or_overwrite_state_[s.name()] = true;
    add_or_overwrite_deriv_[s.name()] = true;
  }
}

Layer:: ~Layer() {
  if (img_display_ != NULL) delete img_display_;
  if (loss_ != NULL) delete loss_;
  if (performance_ != NULL) delete performance_;
}

void Layer::AddIncoming(Edge* e) {
  is_input_ = false;
  incoming_edge_.push_back(e);
  int edge_gpu_id = e->GetGPUId();
  if (edge_gpu_id != gpu_id_) {
    other_incoming_gpu_ids_.insert(edge_gpu_id);
    has_incoming_from_other_gpus_ = true;
  } else {
    has_incoming_from_same_gpu_ = true;
  }
}

void Layer::AddOutgoing(Edge* e) {
  is_output_ = false;
  outgoing_edge_.push_back(e);
  int edge_gpu_id = e->GetGPUId();
  if (edge_gpu_id != gpu_id_) {
    other_outgoing_gpu_ids_.insert(edge_gpu_id);
    has_outgoing_to_other_gpus_ = true;
  } else {
    has_outgoing_to_same_gpu_ = true;
  }
}

bool Layer::HasTiedData() const {
  return has_tied_data_;
}

const string& Layer::GetTiedDataLayerName() const {
  return tied_data_layer_name_;
}

void Layer::AllocateMemoryOnOtherGPUs() {
  set<int> other_gpu_ids = other_incoming_gpu_ids_;
  other_gpu_ids.insert(other_outgoing_gpu_ids_.begin(),
                       other_outgoing_gpu_ids_.end());

  for (int gpu_id : other_gpu_ids) {
    Matrix::SetDevice(gpu_id);
    other_states_[gpu_id].AllocateGPUMemory(state_.GetRows(), state_.GetCols(), GetName() + " other state");
    other_derivs_[gpu_id].AllocateGPUMemory(deriv_.GetRows(), deriv_.GetCols(), GetName() + " other deriv");
    state_copied_[gpu_id] = false;
    deriv_copied_[gpu_id] = false;
  }
}

Matrix& Layer::GetOtherState(int gpu_id) {
  map<int, Matrix>::iterator it;
  it = other_states_.find(gpu_id);
  if (it == other_states_.end()) {
    cerr << "Other state not found on gpu " << gpu_id << endl;
    exit(1);
  }
  return it->second;
}
Matrix& Layer::GetOtherDeriv(int gpu_id) {
  map<int, Matrix>::iterator it;
  it = other_derivs_.find(gpu_id);
  if (it == other_derivs_.end()) {
    cerr << "Other deriv not found on gpu " << gpu_id << endl;
    exit(1);
  }
  return it->second;
}

/** Add up the state from all GPUs.*/
void Layer::AccumulateState() {
  bool overwrite = !has_incoming_from_same_gpu_;
  for (int gpu_id : other_incoming_gpu_ids_) {
    Matrix& other = GetOtherState(gpu_id);
    other.WaitTillReady();  // dst->SetReady after ComputeUp.
    if (overwrite) {
      state_.Set(other);
    } else {
      state_.Add(other);
    }
    overwrite = false;
  }
}

void Layer::AccumulateDeriv() {
  bool overwrite = !has_outgoing_to_same_gpu_;
  for (int gpu_id : other_outgoing_gpu_ids_) {
    Matrix& other = GetOtherDeriv(gpu_id);
    other.WaitTillReady();  // setready after computedown.
    if (overwrite) {
      deriv_.Set(other);
    } else {
      deriv_.Add(other);
    }
    overwrite = false;
  }
}

void Layer::BroadcastState() {
  if (has_outgoing_to_other_gpus_) {
    for (int gpu_id: other_outgoing_gpu_ids_) {
      CopyStateToGPU(gpu_id);
    }
  }
}

void Layer::ResetStateCopies() {
  for (int gpu_id: other_incoming_gpu_ids_) state_copied_[gpu_id] = false;
  for (int gpu_id: other_outgoing_gpu_ids_) state_copied_[gpu_id] = false;
}

void Layer::ResetDerivCopies() {
  for (int gpu_id: other_incoming_gpu_ids_) deriv_copied_[gpu_id] = false;
  for (int gpu_id: other_outgoing_gpu_ids_) deriv_copied_[gpu_id] = false;
}

void Layer::CopyStateToGPU(int dest_gpu) {
  if (!state_copied_[dest_gpu]) {
    Matrix::SetDevice(dest_gpu);
    state_.WaitTillReady();  // wait for l->GetState().SetReady() after ApplyActivation.
    //GetOtherState(gpu_id).CopyP2PAsync(state_);
    GetOtherState(dest_gpu).Set(state_);
    state_copied_[dest_gpu] = true;
  }
}

void Layer::BroadcastDeriv() {
  if (has_incoming_from_other_gpus_) {
    for (int gpu_id: other_incoming_gpu_ids_) {
      CopyDerivToGPU(gpu_id);
    }
  }
}

void Layer::CopyDerivToGPU(int dest_gpu) {
  if (!deriv_copied_[dest_gpu]) {
    Matrix::SetDevice(dest_gpu);
    deriv_.WaitTillReady();  // wait for l->GetDeriv().SetReady() after ApplyDerivativeofActivation.
    //GetOtherDeriv(dest_gpu).CopyP2PAsync(deriv_);
    GetOtherDeriv(dest_gpu).Set(deriv_);
    deriv_copied_[dest_gpu] = true;
  }
}

void Layer::SetSize(int image_size_y, int image_size_x, int image_size_t) {
  image_size_y_ = image_size_y;
  image_size_x_ = image_size_x;
  image_size_t_ = image_size_t;
  cout << "Layer " << name_ << ": " << image_size_y << "x" << image_size_x;
  if (image_size_t > 1) cout << "x" << image_size_t;
  cout << endl;
  if (display_) {
    if (num_channels_ == 3) {
      img_display_ = new ImageDisplayer(image_size_x, image_size_y, num_channels_, false, name_);
    } else {
      img_display_ = new ImageDisplayer(image_size_x, image_size_y, num_channels_, true, name_);
    }
  }
}

void Layer::SetupSlices() {
  int start = 0, end;
  const int num_pixels = image_size_y_ * image_size_x_ * image_size_t_;
  const int batch_size = state_.GetRows();
  for (auto& kv : slice_channels_) {
    end = start + num_pixels * kv.second;
    state_.GetSlice(state_slices_[kv.first], start, end);
    deriv_.GetSlice(deriv_slices_[kv.first], start, end);
    state_slices_[kv.first].SetShape4D(batch_size, image_size_x_, image_size_y_, kv.second * image_size_t_);
    deriv_slices_[kv.first].SetShape4D(batch_size, image_size_x_, image_size_y_, kv.second * image_size_t_);
    start = end;
  }
}

void Layer::AllocateMemory(int batch_size) {
  const int num_pixels = image_size_y_ * image_size_x_ * image_size_t_;
  Matrix::SetDevice(gpu_id_);
  state_.AllocateGPUMemory(batch_size, num_pixels * num_channels_, GetName() + " state");
  deriv_.AllocateGPUMemory(batch_size, num_pixels * num_channels_, GetName() + " deriv");
  state_.SetShape4D(batch_size, image_size_x_, image_size_y_, num_channels_ * image_size_t_);
  deriv_.SetShape4D(batch_size, image_size_x_, image_size_y_, num_channels_ * image_size_t_);

  if (is_input_) store_dropout_noise_ = false;
  if (store_dropout_noise_) {
    dropout_noise_.AllocateGPUMemory(batch_size, num_pixels * num_channels_, GetName() + " dropout");
  }
  SetupSlices();

  AllocateMemoryOnOtherGPUs();
  Matrix::SetDevice(gpu_id_);
  if (is_output_) {
    loss_ = LossFunction::ChooseLossFunction(loss_function_);
    performance_ = LossFunction::ChooseLossFunction(performance_metric_);
  }
}

Matrix& Layer::GetState() {
  return state_;
}

Matrix& Layer::GetState(const string& slice) {
  if (slice.empty()) {
    return state_;
  } else {
    auto it = state_slices_.find(slice);
    if (it == state_slices_.end()) {
      cerr << "Layer " << name_ << " does not contain a slice called " << slice << endl;
      exit(1);
    }
    return it->second;
  }
}

bool Layer::AddOrOverwriteState(const string& slice) {
  auto it = add_or_overwrite_state_.find(slice);
  if (it == add_or_overwrite_state_.end()) {
    cerr << "Layer " << name_ << " does not contain a slice called " << slice << endl;
    exit(1);
  }
  bool val = it->second;
  it->second = false;
  return val;
}

bool Layer::AddOrOverwriteDeriv(const string& slice) {
  auto it = add_or_overwrite_deriv_.find(slice);
  if (it == add_or_overwrite_deriv_.end()) {
    cerr << "Layer " << name_ << " does not contain a slice called " << slice << endl;
    exit(1);
  }
  bool val = it->second;
  it->second = false;
  return val;
}

void Layer::ResetAddOrOverwrite() {
  for (auto& kv : add_or_overwrite_state_) kv.second = true;
  for (auto& kv : add_or_overwrite_deriv_) kv.second = true;
}

int Layer::GetNumChannels(const string& slice) const {
  int res = 0;
  if (slice.empty()) {
    res = num_channels_;
  } else {
    auto it = slice_channels_.find(slice);
    if (it == slice_channels_.end()) {
      cerr << "Layer " << name_ << " does not contain a slice called " << slice << endl;
      exit(1);
    } else {
      res = it->second;
    }
  }
  return res;
}

Matrix& Layer::GetDeriv() {
  return deriv_;
}

Matrix& Layer::GetDeriv(const string& slice) {
  if (slice.empty()) {
    return deriv_;
  } else {
    auto it = deriv_slices_.find(slice);
    if (it == deriv_slices_.end()) {
      cerr << "Layer " << name_ << " does not contain a slice called " << slice << endl;
      exit(1);
    }
    return it->second;
  }
}

void Layer::ApplyDropoutAtTrainTime() {
  if (dropprob_ > 0) {
    if (gaussian_dropout_) {
      dropout_noise_.FillWithRandn();
      dropout_noise_.Mult(dropprob_);
      dropout_noise_.Add(1);
      state_.Mult(dropout_noise_);
      if (max_act_gaussian_dropout_ > 0) {
        // Clip the activations so that |act| <= max_act_gaussian_dropout_
        state_.UpperBoundMod(max_act_gaussian_dropout_);
      }
    } else {  //  Standard binary dropout.
      float scale = dropout_scale_up_at_train_time_ ?
                    (1.0 / (1 - dropprob_)) : 1.0;
      if (store_dropout_noise_) {
        dropout_noise_.SampleBernoulli(1- dropprob_);
        dropout_noise_.Mult(scale);
        state_.Mult(dropout_noise_);
      } else {
        // Does the same thing as above, but doesn't remember the noise.
        // The only reason to do this is to save memory on the gpu.
        // This is ok to do for layers
        // (1) whose activation function has a slope of 0 when the activation is 0 (logistic, relu).
        // (2) which we don't want to backprop through (e.g. input layers).
        state_.Dropout(dropprob_, 0, scale);
      }
    }
  }
}

void Layer::ApplyDerivativeofDropout() {
  if (dropprob_ > 0) {
    if (gaussian_dropout_) {
      deriv_.Mult(dropout_noise_);
      // The real state must be used for backproping through the non linearity.
      // The gradient for the layer above has already been computed.
      // Undo dropout.
      state_.Divide(dropout_noise_);
    } else {
      if (store_dropout_noise_) {
        deriv_.Mult(dropout_noise_);
      } else if (dropout_scale_up_at_train_time_) {
        deriv_.Mult(1. / (1 - dropprob_));
      }
    }
  }
}

void Layer::ApplyDropoutAtTestTime() {
  if (dropprob_ > 0) {
    if (!dropout_scale_up_at_train_time_ && !gaussian_dropout_) {
      // Scale down.
      state_.Mult(1 - dropprob_);
    }
  }
}

float Layer::GetPerformanceMetric() {
  return performance_->GetLoss(state_, data_);
}

void Layer::ComputeDeriv() {
  loss_->GetLossDerivative(state_, data_, deriv_);
  if (loss_function_weight_ != 1.0) {
    deriv_.Mult(loss_function_weight_);
  }
}

float Layer::GetLoss() {
  return loss_function_weight_ * loss_->GetLoss(state_, data_);
}

void Layer::Display() {
  Display(0);
}

void Layer::Display(int image_id) {
  if (img_display_ != NULL && display_) {
    state_.CopyToHost();
    img_display_->DisplayImage(state_.GetHostData(), state_.GetRows(), image_id);
    //copy_to_host(&deriv_);
    //img_display->DisplayImage(deriv_.data_host, deriv_.size[0], image_id);
  }
}

void Layer::ApplyDropout(bool train) {
  if (train) {
    ApplyDropoutAtTrainTime();
  } else {
    ApplyDropoutAtTestTime();
  }
}

LinearLayer::LinearLayer(const config::Layer& config) : Layer(config) {
}

void LinearLayer::ApplyActivation() {
  // Do nothing.
}

void LinearLayer::ApplyDerivativeOfActivation() {
  // Do nothing.
}

void LinearLayer::AllocateMemory(int batch_size) {
  Layer::AllocateMemory(batch_size);
  const int num_pixels = image_size_y_ * image_size_x_ * image_size_t_;
  if (is_output_) data_.AllocateGPUMemory(batch_size, num_pixels * num_channels_, GetName() + " data");
}

ReLULayer::ReLULayer(const config::Layer& config) :
  LinearLayer(config), rectify_after_gaussian_dropout_(false) {
  store_dropout_noise_ = false;
}

void ReLULayer::ApplyActivation() {
  state_.LowerBound(0);
}

void ReLULayer::ApplyDropout(bool train) {
  Layer::ApplyDropout(train);
  if (gaussian_dropout_ && rectify_after_gaussian_dropout_) {
    ApplyActivation();
  }
}

void ReLULayer::ApplyDerivativeOfActivation() {
  deriv_.ApplyDerivativeOfReLU(state_);
}

void SoftmaxLayer::AllocateMemory(int batch_size) {
  Layer::AllocateMemory(batch_size);
  if (is_output_) data_.AllocateGPUMemory(batch_size, 1, GetName() + " data");
  Matrix::RegisterTempMemory(batch_size);
}

void SoftmaxLayer::ApplyActivation() {
  state_.ApplySoftmax();
}

void SoftmaxLayer::ApplyDerivativeOfActivation() {
  cerr << "Back prop through softmax is not implemented." << endl;
  exit(1);
}

void SoftmaxDistLayer::AllocateMemory(int batch_size) {
  Layer::AllocateMemory(batch_size);
  const int numdims = state_.GetCols();
  Matrix::RegisterTempMemory(batch_size * numdims);  // For computing CE.
  if (is_output_) data_.AllocateGPUMemory(batch_size, numdims, GetName() + " data");
}

LogisticLayer::LogisticLayer(const config::Layer& config) : Layer(config) {
  store_dropout_noise_ = false; 
}

void LogisticLayer::AllocateMemory(int batch_size) {
  Layer::AllocateMemory(batch_size);
  Matrix::RegisterTempMemory(batch_size);
  if (is_output_) data_.AllocateGPUMemory(batch_size, num_channels_, GetName() + " data");
}

void LogisticLayer::ApplyActivation() {
  state_.ApplyLogistic();
}

void LogisticLayer::ApplyDerivativeOfActivation() {
  deriv_.ApplyDerivativeOfLogistic(state_);
}
