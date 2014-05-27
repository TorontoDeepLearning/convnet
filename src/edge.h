/* This class is intended to be used as a base class for implementing edges.*/

#ifndef EDGE_H_
#define EDGE_H_
#include "util.h"
#include "matrix.h"
#include <iostream>

class Layer;
class Edge {
 public:
  Edge(const config::Edge& edge_config);
  ~Edge();
  
  virtual void AllocateMemory(bool fprop_only);
  virtual void Initialize();
  virtual void SaveParameters(hid_t file);
  virtual void LoadParameters(hid_t file);
  virtual void InsertPolyak();
  virtual void BackupCurrent();
  virtual void LoadCurrentOnGPU();
  virtual void LoadPolyakOnGPU();

  virtual float GetRMSWeight();
  virtual void ReduceLearningRate(float factor);
  virtual bool HasNoParameters() const;
  virtual int GetNumModules() const;
  virtual void DisplayWeights();
  virtual void DisplayWeightStats();
  virtual void SetTiedTo(Edge* e);

  // Any derived class that implements an edge MUST override these.
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite) = 0;
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input,
                           bool overwrite) = 0;

  virtual void ComputeOuter(Matrix& input, Matrix& deriv_output);
  virtual void UpdateWeights();

  virtual bool RequiresMemoryForDeriv() const;
  virtual void SetImageSize(int image_size);

  bool IsBackPropBlocked() const { return block_backprop_; }

  void SetSource(Layer* source);
  void SetDest(Layer* dest);
  Layer* GetSource();
  Layer* GetDest();
  const string& GetSourceName();
  const string& GetDestName();
  const string& GetName();
  void SetInputChannels(int a);
  void SetOutputChannels(int a);
  void SetMark();
  bool HasMark();
  string GetTiedEdgeName();
  bool IsTied();
  int GetGPUId() const { return gpu_id_; }

  // Multi-gpu.
  void ComputeStart(Matrix& mat);
  void ComputeEnd(Matrix& mat);

  static Edge* ChooseEdgeClass(const config::Edge& edge_config);
  
 protected:
  Layer *source_, *dest_;
  const string source_node_, dest_node_, name_, tied_edge_name_;
  Edge* tied_edge_;
  int num_input_channels_, num_output_channels_, image_size_, num_modules_;
  bool mark_;  // Used for topological sorting.
  const bool block_backprop_, is_tied_;
  ImageDisplayer *img_display_;
  const int gpu_id_;
};
#endif
