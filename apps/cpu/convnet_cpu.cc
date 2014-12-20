#include "convnet_cpu.h"
#include "util.h"

#include <google/protobuf/text_format.h>

#include <sstream>
#include <fstream>
#include <iostream>
#include <stack>

using namespace std;

namespace cpu {

ConvNetCPU::ConvNetCPU(const string& model_structure, const string& model_parameters, const string& mean_file, int batch_size) {
  model_ = new config::Model;
  stringstream ss;
  string line;
  ifstream file(model_structure.c_str());
  ss << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(ss.str(), model_)) {
    cerr << "Could not read text proto buffer : " << model_structure << endl;
    exit(1);
  }
  layers_.resize(model_->layer_size());
  edges_.resize(model_->edge_size());

  // Setup edges.
  for (int i = 0; i < edges_.size(); i++) {
    edges_[i] = new Edge(model_->edge(i));
  }

  // Communicate information about tied edges.
  map<string, Edge*> edge_name_map;
  for (Edge* e: edges_) {
    edge_name_map[e->GetName()] = e;
  }
  for (Edge* e: edges_) {
    if (e->IsTied()) {
      e->SetTiedTo(edge_name_map[e->GetTiedEdgeName()]);
    }
  }

  // Setup layers.
  for (int i = 0; i < layers_.size(); i++) {
    layers_[i] = new Layer(model_->layer(i));
    for (Edge* e : edges_) {
      if (layers_[i]->GetName().compare(e->GetSourceName()) == 0) {
        layers_[i]->AddOutgoing(e);
        e->SetSource(layers_[i]);
        e->SetInputChannels(layers_[i]->GetNumChannels());
      }
      if (layers_[i]->GetName().compare(e->GetDestName()) == 0) {
        layers_[i]->AddIncoming(e);
        e->SetDest(layers_[i]);
        e->SetOutputChannels(layers_[i]->GetNumChannels());
      }
    }
  }

  // Topologically sort layers.
  Sort();

  // Allocate memory. 
  int image_size_y, image_size_x, image_size_t;
  for (Layer* l : layers_) {
    // Find out the spatial size of the layer.
    if (l->IsInput()) {
      image_size_y = l->GetSizeY();
      image_size_x = l->GetSizeX();
      image_size_t = l->GetSizeT();
      if (image_size_y <= 0)
        image_size_y = model_->patch_size();
      if (image_size_x <= 0)
        image_size_x = model_->patch_size();
      if (image_size_t <= 0)
        image_size_t = model_->patch_size();
    } else {
      image_size_y = l->incoming_edge_[0]->GetNumModulesY();
      image_size_x = l->incoming_edge_[0]->GetNumModulesX();
      image_size_t = l->incoming_edge_[0]->GetNumModulesT();
    }
    l->SetSize(image_size_y, image_size_x, image_size_t);
    l->AllocateMemory(batch_size);
    for (Edge* e: l->outgoing_edge_) {
      e->SetImageSize(image_size_y, image_size_x, image_size_t);
    }
  }
  hid_t hdf5_model = H5Fopen(model_parameters.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  for (Edge* e : edges_) {
    e->AllocateMemory();
    e->LoadParameters(hdf5_model);
  }
  H5Fclose(hdf5_model);
  SetMean(mean_file);
}

void ConvNetCPU::SetMean(const string& mean_file) {
  hid_t means = H5Fopen(mean_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  int rows, cols;
  ReadHDF5Shape(means, "pixel_mean", &rows, &cols);
  mean_.AllocateMainMemory(rows, cols);
  ReadHDF5CPU(means, mean_.GetHostData(), mean_.GetNumEls(), "pixel_mean");
  ReadHDF5Shape(means, "pixel_std", &rows, &cols);
  std_.AllocateMainMemory(rows, cols);
  ReadHDF5CPU(means, std_.GetHostData(), std_.GetNumEls(), "pixel_std");
  H5Fclose(means);
}

void ConvNetCPU::Normalize(const unsigned char* i_data, float* o_data, int num_dims, int num_colors) {
  float* mean = mean_.GetHostData(), *std = std_.GetHostData();
#ifdef USE_OPENMP
  #pragma omp parallel for if(num_dims > 10000)
#endif
  for (int i = 0; i < num_dims; i++) {
    o_data[i] = (static_cast<float>(i_data[i]) - mean[i % num_colors]) / std[i % num_colors];
  }
}

void ConvNetCPU::Fprop(const unsigned char* data, int batch_size) {
  bool overwrite;
  for(Layer* l : layers_) {
    overwrite = true;
    for (Edge* e : l->incoming_edge_) {
      e->ComputeUp(e->GetSource()->GetFullState(), l->GetFullState(), overwrite, batch_size);
      overwrite = false;
    }
    if (!l->IsInput()) {
      l->ApplyActivation();
    } else {
      Normalize(data, l->GetState(), l->GetDims() * batch_size, l->GetNumChannels());
    }
  }
}

Layer* ConvNetCPU::GetLayerByName(const string& name) {
  for (Layer* l:layers_) if (l->GetName().compare(name) == 0) return l;
  cerr << "Error: No layer called " << name << endl;
  exit(1);
  return NULL;
}

void ConvNetCPU::Sort() {
  Layer *m, *n;
  vector<Layer*> L;
  stack<Layer*> S;
  for (Layer* l : layers_) if (l->IsInput()) S.push(l);
  if (S.empty()) {
    cerr << "Error: No layer is set to be input!" << endl;
    exit(1);
  }
  bool x;

  while (!S.empty()) {
    n = S.top();
    S.pop();
    L.push_back(n);
    for (Edge* e : n->outgoing_edge_) {
      e->SetMark();
      m = e->GetDest();
      x = true;
      for (Edge* f : m->incoming_edge_) x &= f->HasMark();
      if (x) S.push(m);
    }
  }
  x = true;
  for (Edge* f : edges_) x &= f->HasMark();
  if (!x) {
    cerr << "Error : Network has loop(s)!" << endl;
    exit(1);
  }

  // Re-order layers in the instance variable.
  for (int i = 0; i < layers_.size(); i++)
    layers_[i] = L[i];
}

Layer::Layer(const config::Layer& config) :
  activation_(config.activation()),
  name_(config.name()),
  num_channels_(config.num_channels()),
  is_input_(config.is_input()),
  is_output_(config.is_output()),
  batch_size_(0),
  num_dims_(0),
  image_size_y_(config.image_size_y()),
  image_size_x_(config.image_size_x()),
  image_size_t_(config.image_size_t()) {}

Layer::~Layer() {
}

void Layer::SetSize(int image_size_y, int image_size_x, int image_size_t) {
  image_size_y_ = image_size_y;
  image_size_x_ = image_size_x;
  image_size_t_ = image_size_t;
  cout << "Layer " << name_ << ": " << image_size_y << "x" << image_size_x << endl;
}

void Layer::Print() {
  state_.Print();
}

void Layer::AddIncoming(Edge* e) {
  incoming_edge_.push_back(e);
}

void Layer::AddOutgoing(Edge* e) {
  outgoing_edge_.push_back(e);
}

void Layer::AllocateMemory(int batch_size) {
  batch_size_ = batch_size;
  num_dims_ = image_size_y_ * image_size_x_ * image_size_t_ * num_channels_;
  state_.AllocateMainMemory(batch_size_, num_dims_);
  state_.SetShape4D(batch_size, image_size_x_, image_size_y_, num_channels_ * image_size_t_);
}

void Layer::ApplyActivation()
{
  switch (activation_)
  {
    case config::Layer::LINEAR:
      // no op.
      break;
    case config::Layer::LOGISTIC:
      state_.ApplyLogistic();
      break;
    case config::Layer::RECTIFIED_LINEAR:
      state_.LowerBound(0);
      break;
    case config::Layer::SOFTMAX:
      state_.ApplySoftmax2();
      break;
    default:
      cerr << "Undefined layer type." << endl;
      exit(1);
  }
}

Edge::Edge(const config::Edge& edge_config) :
  edge_type_(edge_config.edge_type()),
  source_(NULL),
  dest_(NULL),
  source_node_(edge_config.source()),
  dest_node_(edge_config.dest()),
  name_(source_node_ + ":" + dest_node_),
  tied_edge_name_(edge_config.tied_to()),
  tied_edge_(NULL),
  num_input_channels_(0),
  num_output_channels_(0),
  num_modules_y_(1),
  num_modules_x_(1),
  num_modules_t_(1),
  image_size_y_(0),
  image_size_x_(0),
  image_size_t_(0),
  mark_(false),
  kernel_size_(edge_config.kernel_size()),
  stride_(edge_config.stride()),
  padding_(edge_config.padding()),
  factor_(edge_config.sample_factor()),
  is_tied_(!tied_edge_name_.empty()),
  shared_bias_(edge_config.shared_bias()),
  blocked_(edge_config.response_norm_in_blocks()),
  add_scale_(edge_config.add_scale()),
  pow_scale_(edge_config.pow_scale()),
  frac_of_filters_response_norm_(edge_config.frac_of_filters_response_norm())
{}

void Edge::AllocateMemory() {
  int rows, cols, bias_size;
  switch (edge_type_) {
    case config::Edge::FC :
      rows = num_output_channels_;
      cols = num_input_channels_ * image_size_y_ * image_size_x_; 
      bias_size = num_output_channels_;
      break;
    case config::Edge::CONV_ONETOONE :
      rows = num_output_channels_;
      cols = num_input_channels_; 
      bias_size = num_output_channels_;
      break;
    case config::Edge::CONVOLUTIONAL :
      rows = num_output_channels_;
      cols = num_input_channels_ * kernel_size_ * kernel_size_;
      bias_size = num_output_channels_;
      if (!shared_bias_) {
        bias_size *= num_modules_y_ * num_modules_x_;
      }
      break;
    case config::Edge::LOCAL :
      rows = num_output_channels_;
      cols = num_input_channels_ * kernel_size_ * kernel_size_ * num_modules_y_ * num_modules_x_;
      bias_size = num_output_channels_ * num_modules_y_ * num_modules_x_;
      break;
    default:
      rows = 0;
      cols = 0;
      bias_size = 0;
  }
  if (rows * cols > 0) weights_.AllocateMainMemory(rows, cols);
  if (bias_size > 0) bias_.AllocateMainMemory(bias_size, 1);
  num_filters_response_norm_ = (int) (frac_of_filters_response_norm_ * num_input_channels_);
}

void Edge::SetImageSize(int image_size_y, int image_size_x, int image_size_t) {
  image_size_y_ = image_size_y;
  image_size_x_ = image_size_x;
  image_size_t_ = image_size_t;
  switch (edge_type_) {
    case config::Edge::FC:
      num_modules_y_ = 1;
      num_modules_x_ = 1;
      break;
    case config::Edge::CONVOLUTIONAL:
    case config::Edge::LOCAL:
    case config::Edge::MAXPOOL:
    case config::Edge::AVERAGE_POOL:
      num_modules_y_ = (image_size_y_ + 2 * padding_ - kernel_size_) / stride_ + 1;
      num_modules_x_ = (image_size_x_ + 2 * padding_ - kernel_size_) / stride_ + 1;
      break;
    case config::Edge::RESPONSE_NORM:
    case config::Edge::RGBTOYUV:
    case config::Edge::CONV_ONETOONE:
      num_modules_y_ = image_size_y_;
      num_modules_x_ = image_size_x_;
      break;
    case config::Edge::UPSAMPLE:
      num_modules_y_ = image_size_y_ * factor_;
      num_modules_x_ = image_size_x_ * factor_;
      break;
    case config::Edge::DOWNSAMPLE:
      num_modules_y_ = image_size_y_ / factor_;
      num_modules_x_ = image_size_x_ / factor_;
      break;
  }
}

void Edge::ComputeUp(CPUMatrix& input, CPUMatrix& output, bool overwrite, int batch_size) {
  if (is_tied_) {
    tied_edge_->ComputeUp(input, output, overwrite, batch_size, image_size_y_);
  } else {
    ComputeUp(input, output, overwrite, batch_size, image_size_y_);
  }
}

void Edge::ComputeUp(CPUMatrix& input, CPUMatrix& output, bool overwrite, int batch_size, int image_size)
{
  int input_dims = num_input_channels_ * image_size * image_size;
  int scale_targets = overwrite ? 0: 1;
  switch (edge_type_)
  {
    case config::Edge::FC:
      CPUMatrix::FCUp(input, weights_, output, batch_size, num_output_channels_, input_dims, scale_targets);
      CPUMatrix::AddBias(output, bias_, output, batch_size, num_output_channels_);
      break;

    case config::Edge::CONV_ONETOONE:
      CPUMatrix::FCUp(input, weights_, output, batch_size * image_size * image_size, num_output_channels_, num_input_channels_, scale_targets);
      CPUMatrix::AddBias(output, bias_, output, batch_size * image_size * image_size, num_output_channels_);
      break;

    case config::Edge::CONVOLUTIONAL:
    {
      ConvDesc conv_desc;
      conv_desc.num_input_channels = num_input_channels_;
      conv_desc.num_output_channels = num_output_channels_;
      conv_desc.kernel_size_y = kernel_size_;
      conv_desc.kernel_size_x = kernel_size_;
      conv_desc.stride_y = stride_;
      conv_desc.stride_x = stride_;
      conv_desc.padding_y = padding_;
      conv_desc.padding_x = padding_;
      CPUMatrix::ConvUp2(input, weights_, output, conv_desc, scale_targets);
      int num_modules = (image_size + 2 * padding_ - kernel_size_) / stride_ + 1;
      if (shared_bias_) {
        CPUMatrix::AddBias(output, bias_, output, batch_size * num_modules * num_modules, num_output_channels_);
      } else {
        CPUMatrix::AddBias(output, bias_, output, batch_size, num_output_channels_ * num_modules * num_modules);
      }
    }
      break;

    case config::Edge::MAXPOOL:
    {
      ConvDesc conv_desc;
      conv_desc.num_output_channels = num_output_channels_;
      conv_desc.kernel_size_y = kernel_size_;
      conv_desc.kernel_size_x = kernel_size_;
      conv_desc.stride_y = stride_;
      conv_desc.stride_x = stride_;
      conv_desc.padding_y = padding_;
      conv_desc.padding_x = padding_;
      CPUMatrix::ConvMaxPool2(input, output, conv_desc);
    }
      break;

    case config::Edge::RESPONSE_NORM:
      CPUMatrix::ConvResponseNormCrossMap2(input, output, num_output_channels_,
          num_filters_response_norm_, add_scale_, pow_scale_, blocked_);
      break;

    default:
      cerr << "Not implemented" << endl;
      exit(1);
  }
}

void Edge::LoadParameters(hid_t file)
{
  if (is_tied_)
    return;

  stringstream ss;
  if (weights_.GetNumEls() > 0) {
    ss << source_node_ << ":" << dest_node_ << ":" << "weight";

    float *data = new float[weights_.GetNumEls()];
    ReadHDF5CPU(file, data, weights_.GetNumEls(), ss.str());
    int kernel_size = (edge_type_ == config::Edge::FC) ? image_size_y_ : kernel_size_;
    CPUMatrix::Transpose(data, weights_.GetHostData(), num_output_channels_,
                         kernel_size, kernel_size, num_input_channels_);
    delete[] data;
    ss.str("");
  }
  if (bias_.GetNumEls() > 0) {
    ss << source_node_ << ":" << dest_node_ << ":" << "bias";
    if (!shared_bias_ && edge_type_ == config::Edge::CONVOLUTIONAL) {
      cerr << "Not implemented" << endl;
      exit(1);
    } else {
      ReadHDF5CPU(file, bias_.GetHostData(), bias_.GetNumEls(), ss.str());
    }
  }
}

}

