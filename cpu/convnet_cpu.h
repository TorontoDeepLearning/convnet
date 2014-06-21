#ifndef CONVNET_CPU
#define CONVNET_CPU
#include "cpuconv.h"
#include "hdf5.h"
#include "convnet_config.pb.h"
#include <vector>
#include <string>
using namespace std;
class Edge;
class Layer {
 public:
  Layer(const config::Layer& config);
  ~Layer();
  bool IsInput() const { return is_input_; }
  bool IsOutput() const { return is_output_; }
  void AllocateMemory(int image_size, int batch_size);
  const string& GetName() const { return name_; }
  int GetDims() const { return image_size_ * image_size_ * num_channels_; }
  float* GetState() { return state_.GetData();}
  void AddIncoming(Edge* e);
  void AddOutgoing(Edge* e);
  int GetNumChannels() { return num_channels_; }
  void ApplyActivation();
  void Print();

  vector<Edge*> incoming_edge_, outgoing_edge_;

 private:
  const config::Layer::Activation activation_;
  const string name_;
  const int num_channels_;
  const bool is_input_, is_output_;
  int image_size_, batch_size_, num_dims_;
  CPUMatrix state_;
};

class Edge {
 public:
  Edge(const config::Edge& edge_config);
  ~Edge();
  void SetSource(Layer* source) { source_ = source;}
  void SetDest(Layer* dest) { dest_ = dest;}
  Layer* GetSource() { return source_;}
  Layer* GetDest() { return dest_; }
  const string& GetSourceName() { return source_node_;}
  const string& GetDestName() { return dest_node_; }
  const string& GetName() { return name_;}
  void SetInputChannels(int a) { num_input_channels_ = a;}
  void SetOutputChannels(int a) { num_output_channels_ = a;}
  void SetMark() { mark_ = true;}
  bool HasMark() { return mark_;}
  string GetTiedEdgeName() { return tied_edge_name_;}
  bool IsTied() { return is_tied_;}
  void SetTiedTo(Edge* e) { tied_edge_ = e;}
  int GetNumModules() { return num_modules_;}
  void SetImageSize(int image_size);
  void LoadParameters(hid_t file);
  void AllocateMemory();

  void ComputeUp(const float* input, float* output, bool overwrite, int batch_size);
  void ComputeUp(const float* input, float* output, bool overwrite, int batch_size, int image_size);

 private:
  const config::Edge::EdgeType edge_type_;
  Layer *source_, *dest_;
  const string source_node_, dest_node_, name_, tied_edge_name_;
  Edge* tied_edge_;
  int num_input_channels_, num_output_channels_, image_size_, num_modules_;
  bool mark_;  // Used for topological sorting.

  const int kernel_size_, stride_, padding_, factor_;
  const bool is_tied_, shared_bias_, blocked_;
  const float add_scale_, pow_scale_, frac_of_filters_response_norm_;
  int num_filters_response_norm_;
  CPUMatrix weights_, bias_;
};

class ConvNetCPU {
 public:
  ConvNetCPU(const string& model_structure, const string& model_parameters, int batch_size);
  void Fprop(const unsigned char* data, int batch_size);
  int GetDims(const string& layer_name) const;
  Layer* GetLayerByName(const string& name);
  void SetMean(const string& mean_file);

 private:
  void Normalize(const unsigned char* i_data, float* o_data, int num_dims, int num_colors);
  void Sort();

  config::Model* model_;
  vector<Layer*> layers_;
  vector<Edge*> edges_;
  CPUMatrix mean_, std_;
};
#endif
