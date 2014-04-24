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
  is_input_(config.is_input()),
  is_output_(config.is_output()),
  dropprob_(config.dropprob()),
  display_(config.display()),
  dropout_scale_up_at_train_time_(true),
  gaussian_dropout_(config.gaussian_dropout()),
  max_act_gaussian_dropout_(config.max_act_gaussian_dropout()),
  scale_targets_(0),
  image_size_(0),
  img_display_(NULL),
  gpu_id_(config.gpu_id()){}

Layer:: ~Layer() {
  if (img_display_ != NULL) delete img_display_;
}

void Layer::AddIncoming(Edge* e) {
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
  outgoing_edge_.push_back(e);
  int edge_gpu_id = e->GetGPUId();
  if (edge_gpu_id != gpu_id_) {
    other_outgoing_gpu_ids_.insert(edge_gpu_id);
    has_outgoing_to_other_gpus_ = true;
  } else {
    has_outgoing_to_same_gpu_ = true;
  }
}

void Layer::AllocateMemoryOnOtherGPUs() {
  set<int> other_gpu_ids = other_incoming_gpu_ids_;
  other_gpu_ids.insert(other_outgoing_gpu_ids_.begin(),
                       other_outgoing_gpu_ids_.end());

  for (int gpu_id : other_gpu_ids) {
    Matrix::SetDevice(gpu_id);
    other_states_[gpu_id].AllocateGPUMemory(state_.GetRows(), state_.GetCols(), GetName() + " other state");
  }
  for (int gpu_id : other_gpu_ids) {
    Matrix::SetDevice(gpu_id);
    other_derivs_[gpu_id].AllocateGPUMemory(deriv_.GetRows(), deriv_.GetCols(), GetName() + " other deriv");
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


void Layer::SyncIncomingState() {
  Matrix::SetDevice(gpu_id_);
  bool overwrite = !has_incoming_from_same_gpu_;
  for (int gpu_id : other_incoming_gpu_ids_) {
    Matrix& other = GetOtherState(gpu_id);
    other.WaitTillReady();
    if (overwrite) {
      state_.Set(other);
    } else {
      state_.Add(other);
    }
    overwrite = false;
  }
  state_.SetReady();
}

void Layer::SyncOutgoingState() {
  if (has_outgoing_to_other_gpus_) state_.WaitTillReady();
  for (int gpu_id : other_outgoing_gpu_ids_) {
    Matrix& other = GetOtherState(gpu_id);
    Matrix::SetDevice(other.GetGPUId());
    other.Set(state_);
  }
}

void Layer::SyncOutgoingDeriv() {
  Matrix::SetDevice(gpu_id_);
  bool overwrite = !has_outgoing_to_same_gpu_;
  for (int gpu_id : other_outgoing_gpu_ids_) {
    Matrix& other = GetOtherDeriv(gpu_id);
    other.WaitTillReady();
    if (overwrite) {
      deriv_.Set(other);
    } else {
      deriv_.Add(other);
    }
    overwrite = false;
  }
  deriv_.SetReady();
}


void Layer::SyncIncomingDeriv() {
  if (has_incoming_from_other_gpus_) deriv_.WaitTillReady();
  for (int gpu_id : other_incoming_gpu_ids_) {
    Matrix& other = GetOtherDeriv(gpu_id);
    Matrix::SetDevice(other.GetGPUId());
    other.Set(deriv_);
  }
}

void Layer::AllocateMemory(int image_size, int batch_size) {
  image_size_ = image_size;
  const int num_pixels = image_size * image_size;
  Matrix::SetDevice(gpu_id_);
  state_.AllocateGPUMemory(batch_size, num_pixels * num_channels_, GetName() + " state");
  deriv_.AllocateGPUMemory(batch_size, num_pixels * num_channels_, GetName() + " deriv");
  if (gaussian_dropout_) {
    rand_gaussian_.AllocateGPUMemory(batch_size, num_pixels * num_channels_);
  }
  AllocateMemoryOnOtherGPUs();

  if (display_) {
    if (num_channels_ == 3) {
      img_display_ = new ImageDisplayer(image_size, image_size, num_channels_, false, name_);
    } else {
      img_display_ = new ImageDisplayer(image_size, image_size, num_channels_, true, name_);
    }
  }
}

void Layer::AllocateMemoryEdges(int image_size) {
}

void Layer::ApplyDropoutAtTrainTime() {
  if (dropprob_ > 0) {
    cudamat* state = state_.GetMat();
    if (gaussian_dropout_) {
      rand_gaussian_.FillWithRandn();
      cudamat* rnd_g = rand_gaussian_.GetMat();
      mult_by_scalar(rnd_g, dropprob_, rnd_g);
      add_scalar(rnd_g, 1, rnd_g);
      mult_elementwise(state, rnd_g, state);
      //gaussian_dropout(&Matrix::rnd_, state, dropprob_);
      if (max_act_gaussian_dropout_ > 0) {
        // Clip the activations so that |act| <= max_act_gaussian_dropout_
        upper_bound_mod_scalar(state, max_act_gaussian_dropout_, state);
      }
    } else {
      if (dropout_scale_up_at_train_time_) {
        dropout(&Matrix::rnd_[gpu_id_], state, dropprob_, 0, 1.0 / (1 - dropprob_));
      } else {
        // Dropout only (scale down at test time).
        dropout(&Matrix::rnd_[gpu_id_], state, dropprob_, 0, 1.0);
      }
    }
  }
}

void Layer::ApplyDerivativeofDropout() {
  if (dropprob_ > 0) {
    cudamat* deriv = deriv_.GetMat();
    if (gaussian_dropout_) {
      cudamat* state = state_.GetMat();
      cudamat* rnd_g = rand_gaussian_.GetMat();
      mult_elementwise(deriv, rnd_g, deriv);
      // The real state must be used for backproping through the non linearity.
      // The gradient for the layer above has already been computed.
      // Undo dropout.
      divide_elementwise(state, rnd_g, state);
    } else if (dropout_scale_up_at_train_time_) {
      mult_by_scalar(deriv, 1./(1 - dropprob_), deriv);
    }
  }
}

void Layer::ApplyDropoutAtTestTime() {
  if (dropprob_ > 0) {
    // Scale down.
    if (!dropout_scale_up_at_train_time_ && !gaussian_dropout_) {
      mult_by_scalar(state_.GetMat(), 1 - dropprob_, state_.GetMat());
    }
  }
}

float Layer::GetLoss2() {
  return GetLoss();
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

void Layer::AccessStateBegin() {
  Matrix::SetDevice(gpu_id_);
}

void Layer::AccessDerivBegin() {
  Matrix::SetDevice(gpu_id_);
}

void Layer::AccessStateEnd() {
  if (has_outgoing_to_other_gpus_) {
    state_.SetReady();
  }
}

void Layer::AccessDerivEnd() {
  if (has_incoming_from_other_gpus_) {
    deriv_.SetReady();
  }
}

void LinearLayer::ApplyActivation(bool train) {
  // Linear layer, do nothing.
  // cout<< "Linear layer activation called." << endl;
  AccessStateBegin();
  ApplyDropout(train);
  AccessStateEnd();
}

void LinearLayer::ApplyDerivativeOfActivation() {
  AccessDerivBegin();
  ApplyDerivativeofDropout();
  AccessDerivEnd();
}

void LinearLayer::ComputeDeriv() {
  AccessStateBegin();
  int err_code = subtract_elementwise(state_.GetMat(), data_.GetMat(), deriv_.GetMat());
  if (err_code != 0) {
    cerr << "Error in compute deriv of linear unit." << endl;
    exit(1);
  }
  AccessDerivEnd();
}

float LinearLayer::GetLoss() {
  AccessStateBegin();
  Matrix temp;
  Matrix::GetTemp(data_.GetRows(), data_.GetCols(), temp);
  int err_code = subtract_elementwise(state_.GetMat(), data_.GetMat(), temp.GetMat());
  if (err_code != 0) {
    cerr << "Error in Get loss of linear unit." << endl;
    exit(1);
  }
  float norm = euclid_norm(temp.GetMat(), &err_code);
  float res = 0.5 * norm * norm;
  return res;
}

void LinearLayer::AllocateMemory(int image_size, int batch_size) {
  Layer::AllocateMemory(image_size, batch_size);
  const int num_pixels = image_size * image_size;
  if (is_output_) data_.AllocateGPUMemory(batch_size, num_pixels * num_channels_);
  //Matrix::RegisterTempMemory(batch_size * num_channels_ * num_pixels); why did I have this?
}

ReLULayer::ReLULayer(const config::Layer& config) :
  LinearLayer(config), rectify_after_gaussian_dropout_(false)
{}

void ReLULayer::ApplyActivation(bool train) {
  AccessStateBegin();
  cudamat* state = state_.GetMat();
  lower_bound_scalar(state, 0, state);
  ApplyDropout(train);
  if (gaussian_dropout_ && rectify_after_gaussian_dropout_) {
    lower_bound_scalar(state, 0, state);
  }
  AccessStateEnd();
}

void ReLULayer::ApplyDerivativeOfActivation() {
  AccessDerivBegin();
  ApplyDerivativeofDropout();
  cudamat* deriv = deriv_.GetMat();
  cudamat* state = state_.GetMat();
  apply_rectified_linear_deriv(deriv, state, deriv);
  AccessDerivEnd();
}

void SoftmaxLayer::AllocateMemory(int image_size, int batch_size) {
  Layer::AllocateMemory(image_size, batch_size);
  if (is_output_) data_.AllocateGPUMemory(batch_size, 1);
  Matrix::RegisterTempMemory(batch_size);
}

void SoftmaxLayer::ApplyActivation(bool train) {
  AccessStateBegin();
  cudamat* state = state_.GetMat();
  softmax_row_major_multi(state, state_.GetCols());
  ApplyDropout(train);
  AccessStateEnd();
}

void SoftmaxLayer::ApplyDerivativeOfActivation() {
  AccessDerivBegin();
  cerr << "Back prop through softmax is not implemented." << endl;
  exit(1);
  AccessDerivEnd();
}

void SoftmaxLayer::ComputeDeriv() {
  AccessStateBegin();
  cudamat* state = state_.GetMat();
  cudamat* target = data_.GetMat();
  cudamat* deriv = deriv_.GetMat();
  apply_softmax_grad_row_major(state, target, deriv);
  AccessDerivEnd();
}

float SoftmaxLayer::GetLoss() {
  AccessStateBegin();
  cudamat* state = state_.GetMat();
  Matrix temp;
  Matrix::GetTemp(data_.GetRows(), 1, temp);
  get_softmax_correct_row_major(state, data_.GetMat(), temp.GetMat());
  float res = temp.Sum();
  //cout << "Softmax loss " << res << " out of " << data_.GetRows() << endl;
  return res;
}

float SoftmaxLayer::GetLoss2() {
  AccessStateBegin();
  cudamat* state = state_.GetMat();
  Matrix temp;
  Matrix::GetTemp(data_.GetRows(), 1, temp);
  get_softmax_cross_entropy_row_major(state, data_.GetMat(), temp.GetMat(), 1e-10);
  float res = temp.Sum();
  return res;
}


void SoftmaxDistLayer::AllocateMemory(int imgsize, int batch_size) {
  Layer::AllocateMemory(imgsize, batch_size);
  const int numdims = state_.GetCols();
  cross_entropy_.AllocateGPUMemory(batch_size, numdims);
  if (is_output_) data_.AllocateGPUMemory(batch_size, numdims);
}

void SoftmaxDistLayer::ComputeDeriv() {
  AccessStateBegin();
  apply_logistic_grad(state_.GetMat(), data_.GetMat(), deriv_.GetMat());
  AccessDerivEnd();
}

float SoftmaxDistLayer::GetLoss() {
  AccessStateBegin();
  int err;
  err = compute_cross_entropy(data_.GetMat(), state_.GetMat(), cross_entropy_.GetMat(), 1e-10);
  if (err != 0) {
    cerr << "SoftmaxDistLayer::GetLoss CE Error : " << GetStringError(err) << endl;
    exit(1);
  }
  return cross_entropy_.Sum();
}

void LogisticLayer::AllocateMemory(int image_size, int batch_size) {
  Layer::AllocateMemory(image_size, batch_size);
  Matrix::RegisterTempMemory(batch_size);
  if (is_output_) data_.AllocateGPUMemory(batch_size, 1);
}

void LogisticLayer::ApplyActivation(bool train) {
  AccessStateBegin();
  // cout<< "Logistic layer activation called." << endl;
  cudamat* state = state_.GetMat();
  apply_sigmoid(state, state);
  ApplyDropout(train);
  AccessStateEnd();
}

void LogisticLayer::ApplyDerivativeOfActivation() {
  AccessDerivBegin();
  ApplyDerivativeofDropout();
  cudamat* deriv = deriv_.GetMat();
  cudamat* state = state_.GetMat();
  apply_logistic_deriv(deriv, state, deriv);
  AccessDerivEnd();
}

void LogisticLayer::ComputeDeriv() {
  AccessStateBegin();
  apply_logistic_grad(state_.GetMat(), data_.GetMat(), deriv_.GetMat());
  AccessDerivEnd();
}

float LogisticLayer::GetLoss() {
  AccessStateBegin();
  Matrix temp;
  Matrix::GetTemp(data_.GetRows(), 1, temp);
  get_logistic_correct_normalized(state_.GetMat(), data_.GetMat(), temp.GetMat());
  return temp.Sum();
}

