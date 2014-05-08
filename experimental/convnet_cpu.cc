#include "convnet_cpu.h"
#include <google/protobuf/text_format.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stack>
using namespace std;

ConvNetCPU::ConvNetCPU(const string& model_structure, const string& model_parameters, int batch_size) {
  model_ = new config::Model;
  stringstream ss;
  string line;
  ifstream file(model_structure.c_str());
  while (getline(file, line)) ss << line;
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
  int image_size;
  for (Layer* l : layers_) {
    // Find out the spatial size of the layer.
    if (l->IsInput()) {
      image_size = model_->patch_size();
    } else {
      image_size = l->incoming_edge_[0]->GetNumModules();
    }
    l->AllocateMemory(image_size, batch_size);
    for (Edge* e: l->outgoing_edge_) {
      e->SetImageSize(image_size);
    }
  }
  cout << "Loading model from " << model_parameters << endl;
  hid_t hdf5_model = H5Fopen(model_parameters.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  for (Edge* e : edges_) {
    e->AllocateMemory();
    e->LoadParameters(hdf5_model);
  }
  H5Fclose(hdf5_model);
  SetMean("/ais/gobi3/u/nitish/imagenet/pixel_mean.h5"); //model_parameters);
}

void ConvNetCPU::SetMean(const string& mean_file) {
  hid_t means = H5Fopen(mean_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  int rows, cols;
  CPUMatrix::ReadHDF5Shape(means, "pixel_mean", &rows, &cols);
  mean_.AllocateMemory(rows, cols);
  CPUMatrix::ReadHDF5(means, mean_.GetData(), mean_.GetSize(), "pixel_mean");
  CPUMatrix::ReadHDF5Shape(means, "pixel_std", &rows, &cols);
  std_.AllocateMemory(rows, cols);
  CPUMatrix::ReadHDF5(means, std_.GetData(), std_.GetSize(), "pixel_std");
  H5Fclose(means);
}

void ConvNetCPU::Normalize(const float* i_data, float* o_data, int num_dims, int num_colors) {
  float* mean = mean_.GetData(), *std = std_.GetData();
  for (int i = 0; i < num_dims; i++) {
    o_data[i] = (i_data[i] - mean[i % num_colors]) / std[i % num_colors];
  }
}

void ConvNetCPU::Fprop(const float* data, int batch_size) {
  bool overwrite;
  for(Layer* l : layers_) {
    overwrite = true;
    for (Edge* e : l->incoming_edge_) {
      e->ComputeUp(e->GetSource()->GetState(), l->GetState(), overwrite, batch_size);
      overwrite = false;
    }
    if (!l->IsInput()) {
      l->ApplyActivation();
    } else {
      Normalize(data, l->GetState(), l->GetDims(), l->GetNumChannels());
    }
    cout << l->GetName() << endl;
    l->Print();
  }
  cout << "---------------" << endl;
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
  for (int i = 0; i < layers_.size(); i++) layers_[i] = L[i];
}

Layer::Layer(const config::Layer& config) :
  activation_(config.activation()),
  name_(config.name()),
  num_channels_(config.num_channels()),
  is_input_(config.is_input()),
  is_output_(config.is_output()),
  image_size_(0), batch_size_(0), num_dims_(0) {}

Layer::~Layer() {
  state_.FreeMemory();
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

void Layer::AllocateMemory(int image_size, int batch_size) {
  image_size_ = image_size;
  batch_size_ = batch_size;
  num_dims_ = image_size * image_size * num_channels_;
  state_.AllocateMemory(batch_size_, num_dims_);
}

void Layer::ApplyActivation() {
  int num_dims = image_size_ * image_size_ * num_channels_;
  float* data = state_.GetData();
  switch (activation_) {
    case config::Layer::LINEAR :
      // no op.
      break;
    case config::Layer::LOGISTIC :
      CPUMatrix::Logistic(data, data, batch_size_ * num_dims);
      break;
    case config::Layer::RECTIFIED_LINEAR :
      CPUMatrix::LowerBound(data, data, batch_size_ * num_dims, 0);
      break;
    case config::Layer::SOFTMAX :
      CPUMatrix::Softmax(data, data, batch_size_, num_dims);
      break;
    default:
      cerr << "Undefined layer type." << endl;
      exit(1);
  }
}

Edge::Edge(const config::Edge& edge_config) :
  edge_type_(edge_config.edge_type()),
  source_(NULL), dest_(NULL),
  source_node_(edge_config.source()),
  dest_node_(edge_config.dest()),
  name_(source_node_ + ":" + dest_node_),
  tied_edge_name_(edge_config.tied_to()),
  tied_edge_(NULL),
  num_input_channels_(0),
  num_output_channels_(0),
  image_size_(0),
  num_modules_(1),
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
  num_filters_response_norm_((int)(edge_config.frac_of_filters_response_norm() * num_input_channels_))
{}

void Edge::AllocateMemory() {
  int rows, cols, bias_size;
  switch (edge_type_) {
    case config::Edge::FC :
      rows = num_output_channels_;
      cols = num_input_channels_ * image_size_ * image_size_; 
      bias_size = num_output_channels_;
      break;
    case config::Edge::CONVOLUTIONAL :
      rows = num_output_channels_;
      cols = num_input_channels_ * kernel_size_ * kernel_size_;
      bias_size = num_output_channels_;
      if (!shared_bias_) {
        bias_size *= num_modules_ * num_modules_;
      }
      break;
    case config::Edge::LOCAL :
      rows = num_output_channels_;
      cols = num_input_channels_ * kernel_size_ * kernel_size_ * num_modules_ * num_modules_;
      bias_size = num_output_channels_ * num_modules_ * num_modules_;
      break;
    default:
      rows = 0;
      cols = 0;
      bias_size = 0;
  }
  if (rows * cols > 0) weights_.AllocateMemory(rows, cols);
  if (bias_size > 0) bias_.AllocateMemory(bias_size, 1);
}

void Edge::SetImageSize(int image_size) {
  image_size_ = image_size;
  switch (edge_type_) {
    case config::Edge::FC :
      num_modules_ = 1;
      break;
    case config::Edge::CONVOLUTIONAL :
    case config::Edge::LOCAL :
    case config::Edge::MAXPOOL :
    case config::Edge::AVERAGE_POOL :
      num_modules_ = (image_size_ + 2 * padding_ - kernel_size_) / stride_ + 1;
      break;
    case config::Edge::RESPONSE_NORM :
    case config::Edge::RGBTOYUV:
      num_modules_ = image_size_;
      break;
    case config::Edge::UPSAMPLE :
      num_modules_ = image_size_ * factor_;
      break;
    case config::Edge::DOWNSAMPLE :
      num_modules_ = image_size_ / factor_;
      break;
  }
}

void Edge::ComputeUp(const float* input, float* output, bool overwrite, int batch_size) {
  if (is_tied_) {
    tied_edge_->ComputeUp(input, output, overwrite, batch_size, image_size_);
  } else {
    ComputeUp(input, output, overwrite, batch_size, image_size_);
  }
}

void Edge::ComputeUp(const float* input, float* output, bool overwrite, int batch_size, int image_size) {
  float* weight = weights_.GetData();
  float* bias = bias_.GetData();
  int num_modules = (image_size + 2 * padding_ - kernel_size_)/ stride_ + 1;
  int input_dims = num_input_channels_ * image_size * image_size;
  switch (edge_type_) {
    case config::Edge::FC :
      CPUMatrix::FCUp(input, weight, output, batch_size, num_output_channels_, input_dims, 0, overwrite);
      CPUMatrix::AddBias(output, bias, output, batch_size, num_output_channels_);
      break;
    case config::Edge::CONVOLUTIONAL :
      CPUMatrix::ConvUp(input, weight, output, batch_size, num_input_channels_, num_output_channels_,
          image_size, image_size, kernel_size_, kernel_size_, stride_, stride_, padding_, padding_, 0, overwrite);
      if (shared_bias_) {
        CPUMatrix::AddBias(output, bias_.GetData(), output, batch_size, num_output_channels_);
      } else {
        CPUMatrix::AddBias(output, bias_.GetData(), output, batch_size, num_output_channels_ * num_modules * num_modules);
      }
      break;
    case config::Edge::MAXPOOL :
      CPUMatrix::MaxPool(input, output, batch_size, num_output_channels_,
                         image_size, image_size, kernel_size_, kernel_size_, stride_, stride_, padding_, padding_, 0, overwrite);
      break;
    case config::Edge::RESPONSE_NORM :
      CPUMatrix::ResponseNormCrossMap(input, output, image_size * image_size * batch_size,
          num_output_channels_, num_filters_response_norm_, blocked_, add_scale_, pow_scale_, 0, overwrite);
      break;
    default:
      cerr << "Not implemented" << endl;
      exit(1);
  }
}


void Edge::LoadParameters(hid_t file) {
  if (is_tied_) return;
  stringstream ss;
  if (weights_.GetSize() > 0) {
    ss << source_node_ << ":" << dest_node_ << ":" << "weight";
    cout << "Loading " << ss.str() << endl;

    float *data = new float[weights_.GetSize()];
    CPUMatrix::ReadHDF5(file, data, weights_.GetSize(), ss.str());
    CPUMatrix::Transpose(data, weights_.GetData(), num_output_channels_, kernel_size_, kernel_size_, num_input_channels_);  
    delete data;
    ss.str("");
  }
  if (bias_.GetSize() > 0) {
    ss << source_node_ << ":" << dest_node_ << ":" << "bias";
    cout << "Loading " << ss.str() << endl;
    CPUMatrix::ReadHDF5(file, bias_.GetData(), bias_.GetSize(), ss.str());
  }
}
