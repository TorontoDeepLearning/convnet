#include "edge.h"
#include <iostream>
#include "edge_with_weight.h"
#include "fc_edge.h"
#include "conv_edge.h"
#include "maxpool_edge.h"
#include "avgpool_edge.h"
#include "local_edge.h"
#include "upsample_edge.h"
#include "downsample_edge.h"
#include "response_norm_edge.h"
#include "rgb_to_yuv_edge.h"
#include "conv_onetoone_edge.h"
using namespace std;

Edge* Edge::ChooseEdgeClass(const config::Edge& edge_config) {
  Edge* e = NULL;
  switch (edge_config.edge_type()) {
    case config::Edge::FC :
      e = new FCEdge(edge_config);
      break;
    case config::Edge::CONVOLUTIONAL :
      e = new ConvEdge(edge_config);
      break;
    case config::Edge::LOCAL :
      e = new LocalEdge(edge_config);
      break;
    case config::Edge::MAXPOOL :
      e = new MaxPoolEdge(edge_config);
      break;
    case config::Edge::AVERAGE_POOL :
      e = new AvgPoolEdge(edge_config);
      break;
    case config::Edge::RESPONSE_NORM :
      e = new ResponseNormEdge(edge_config);
      break;
    case config::Edge::UPSAMPLE :
      e = new UpSampleEdge(edge_config);
      break;
    case config::Edge::DOWNSAMPLE :
      e = new DownSampleEdge(edge_config);
      break;
    case config::Edge::RGBTOYUV :
      e = new RGBToYUVEdge(edge_config);
      break;
    case config::Edge::CONV_ONETOONE :
      e = new ConvOneToOneEdge(edge_config);
      break;
    default:
      cerr << "Error: Undefined edge type." << endl;
      exit(1);
  }
  return e;
}

Edge::Edge(const config::Edge& edge_config) :
  source_(NULL), dest_(NULL),
  source_node_(edge_config.source()),
  dest_node_(edge_config.dest()),
  source_node_slice_(edge_config.source_slice()),
  dest_node_slice_(edge_config.dest_slice()),
  tied_edge_name_(edge_config.tied_to()),
  tied_edge_(NULL),
  num_input_channels_(0),
  num_output_channels_(0),
  image_size_(0),
  num_modules_(1),
  mark_(false),
  block_backprop_(edge_config.block_backprop()),
  is_tied_(!tied_edge_name_.empty()),
  img_display_(NULL),
  gpu_id_(edge_config.gpu_id()),
  display_(edge_config.display()),
  grad_check_(edge_config.grad_check()),
  grad_check_num_params_(edge_config.grad_check_num_params()) {

  stringstream ss;
  ss << source_node_;
  if (!source_node_slice_.empty()) ss << "_" << source_node_slice_;
  ss << ":" << dest_node_;
  if (!dest_node_slice_.empty()) ss << "_" << dest_node_slice_;
  name_ = string(ss.str());
  for (float v : edge_config.grad_check_epsilon()) {
    grad_check_epsilon_.push_back(v);
  }


#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id_);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes_);
#else
  process_id_ = 0;
  num_processes_ = 1;
#endif
}

Edge::~Edge() {
  if (img_display_ != NULL) delete img_display_;
  // Other pointers are not owned by this class.
}

string Edge::GetDescription() {
  return "Default edge.";
}

void Edge::SetTiedTo(Edge* e) {
  tied_edge_ = e;
}

void Edge::SetInputChannels(int a) {
  num_input_channels_ = a;
}

void Edge::SetOutputChannels(int a) {
  num_output_channels_ = a;
  Matrix::SetDevice(gpu_id_);
  Matrix::RegisterTempMemory(num_output_channels_, "Used for computing average length of incoming weight vectors.");
}

void Edge::SaveParameters(hid_t file) {
  // no op.
  // Parameter saving implemented in EdgeWithWeight or derived classes thereof.
}

void Edge::LoadParameters(hid_t file) {
  // no op.
  // Parameter loading implemented in EdgeWithWeight or derived classes thereof.
}

void Edge::Initialize() {
  // no op. Initialization done in derived classes.
  Matrix::SetDevice(gpu_id_);
}

void Edge::SetMemory(Matrix& p) {
  Matrix::SetDevice(gpu_id_);
}

void Edge::SetGradMemory(Matrix& p) {
}

size_t Edge::GetParameterMemoryRequirement() {
  return 0;
}

void Edge::GradCheckEpsilon(vector<float>& epsilon_values) const {
  epsilon_values.clear();
  for (const float& v : grad_check_epsilon_) {
    epsilon_values.push_back(v);
  }
}

void Edge::DisplayWeights() {
  // no op.
}

void Edge::DisplayWeightStats() {
  // no op.
}

void Edge::ReduceLearningRate(float factor) {
  // no op.
}

void Edge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  // no op.
}

void Edge::UpdateWeights() {
  // no op.
}

float Edge::GetRMSWeight() {
  return 0;
}

void Edge::SetSource(Layer* source) {
  source_ = source;
}

void Edge::SetDest(Layer* dest) {
  dest_ = dest;
}

Layer* Edge::GetSource() {
  return source_;
}

Layer* Edge::GetDest() {
  return dest_;
}

const string& Edge::GetSourceName() {
  return source_node_;
}

const string& Edge::GetSourceSliceName() {
  return source_node_slice_;
}

const string& Edge::GetDestName() {
  return dest_node_;
}

const string& Edge::GetDestSliceName() {
  return dest_node_slice_;
}

const string& Edge::GetName() {
  return name_;
}

void Edge::SetMark() {
  mark_ = true;
}

bool Edge::HasMark() {
  return mark_;
}

bool Edge::HasNoParameters() const {
  return true;
}

int Edge::GetNumModules() const {
  return num_modules_;
}

string Edge::GetTiedEdgeName() {
  return tied_edge_name_;
}

bool Edge::IsTied() {
  return is_tied_;
}

void Edge::SetImageSize(int image_size) {
  image_size_ = image_size;
}

void Edge::FOV(int* size, int* sep, int* pad1, int* pad2) const {
}

void Edge::InsertPolyak() {
}

void Edge::BackupCurrent() {
}
void Edge::LoadCurrentOnGPU() {
}
void Edge::LoadPolyakOnGPU() {
}
