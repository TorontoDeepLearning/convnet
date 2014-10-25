#ifndef EDGE_H_
#define EDGE_H_
#include "util.h"
#include "matrix.h"
#include <iostream>

class Layer;

/** This class is intended to be used as a base class for implementing edges.
 * This is an abstract class - ComputeUp and ComputeDown methods must be
 * implemented by derived classes.
 */
class Edge {
 public:
  /** Instatntiate an Edge from the config.*/
  Edge(const config::Edge& edge_config);
  virtual ~Edge();
  
  /** Returns a human-readable string describing the edge.*/
  virtual string GetDescription();

  /** Setup memory for parameters.
   * @param p The slice of memory given to this edge.
   */ 
  virtual void SetMemory(Matrix& p);

  /** Setup memory for storing the gradient of the parameters.
   * @param p The slice of memory given to this edge for this purpose.
   */ 
  virtual void SetGradMemory(Matrix& p);

  /** Returns number of floats needed by this edge to store parameters.*/
  virtual size_t GetParameterMemoryRequirement();
  
  /** Initialize the weights and biases.*/
  virtual void Initialize();

  /** Write the weights and biases in an hdf5 file.
   * @param file The file handle. The file has been opened for writing. Do not close it.
   */ 
  virtual void SaveParameters(hid_t file);
  
  /** Load the weights and biases from an hdf5 file.
   * @param file The file handle. The file has been opened for reading. Do not close it.
   */ 
  virtual void LoadParameters(hid_t file);

  virtual void InsertPolyak();
  virtual void BackupCurrent();
  virtual void LoadCurrentOnGPU();
  virtual void LoadPolyakOnGPU();

  /** Returns the root mean square weight value.*/
  virtual float GetRMSWeight();

  /** Reduce the learning rate by factor.*/
  virtual void ReduceLearningRate(float factor);

  /** Returns whether the edge has any parameters.*/
  virtual bool HasNoParameters() const;

  /** Returns the number of modules along y-axis.
   * This is relevant for convolution-like edges.
   */ 
  virtual int GetNumModulesY() const;

  /** Returns the number of modules along x-axis.
   * This is relevant for convolution-like edges.
   */ 
  virtual int GetNumModulesX() const;

  /** Displays the weights.
   * Supportsinput layer weights only.
   */
  virtual void DisplayWeights();

  /** Displays the statistics of the weights.*/
  virtual void DisplayWeightStats();

  /** Sets the edge to be tied to another edge.*/
  virtual void SetTiedTo(Edge* e);

  /** Computes the output layer state given the input.
   * Applies the weights and adds bias.
   */
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite) = 0;
  
  /** Computes the derivative w.r.t the inputs of this edge given the derivative
   * w.r.t the outputs of this edge.
   * @param deriv_output Derivative w.r.t outputs of this edge.(In)
   * @param input The input to this edge.(In)
   * @param output The output of this edge.(In)
   * @param deriv_input Derivative w.r.t inputs of this edge.(Out)
   */
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input,
                           bool overwrite) = 0;

  /** Computes the gradient for the weights and biases.
   * @param input The input to this edge.
   * @param deriv_output The derivative w.r.t the output of this edge.
   */ 
  virtual void ComputeOuter(Matrix& input, Matrix& deriv_output);
  
  /** Update the weights.*/
  virtual void UpdateWeights();

  //virtual bool RequiresMemoryForDeriv() const;
  
  /** Set the spatial size of the input to this edge.*/
  virtual void SetImageSize(int image_size_y, int image_size_x);

  /** Returns the size of the input field of view corresponding to an output
   * of 'size'.*/
  virtual void FOV(int* size, int* sep, int* pad1, int* pad2) const;

  /** Returns whether back prop is blocked through this edge.*/
  bool IsBackPropBlocked() const { return block_backprop_; }

  void SetSource(Layer* source);
  void SetDest(Layer* dest);
  Layer* GetSource();
  Layer* GetDest();
  const string& GetSourceName();
  const string& GetDestName();
  const string& GetName();
  const string& GetSourceSliceName();
  const string& GetDestSliceName();

  /** Set the number of input channels.*/
  void SetInputChannels(int a);
  /** Set the number of output channels.*/
  void SetOutputChannels(int a);

  void SetMark();
  bool HasMark();
  string GetTiedEdgeName();
  bool IsTied();
  int GetGPUId() const { return gpu_id_; }
  bool GradCheck() const { return grad_check_;}
  int GradCheckNumParams() const { return grad_check_num_params_;}
  void GradCheckEpsilon(vector<float>& epsilon_values) const;

  /** Selects the appropriate derived class for the edge config.*/
  static Edge* ChooseEdgeClass(const config::Edge& edge_config);
  
 protected:
  Layer *source_;  /** The source layer for this edge.*/
  Layer *dest_;  /** The destination layer for this edge.*/
  const string source_node_, dest_node_, source_node_slice_, dest_node_slice_,
        tied_edge_name_;
  string name_;
  Edge* tied_edge_;  /* The edge to which this edge is tied.*/
  int num_input_channels_, num_output_channels_,
      image_size_y_, image_size_x_,
      num_modules_y_, num_modules_x_;
  bool mark_;  /** A marker. Used for topological sorting.*/
  const bool block_backprop_, is_tied_;
  ImageDisplayer *img_display_;
  const int gpu_id_;  /** The GPU on which this edge should do its computation.*/
  const bool display_;

  int process_id_;  // MPI rank.
  int num_processes_;  // Number of data parallel processes.
  bool is_root_;  // The process that is root.
  const bool grad_check_;
  const int grad_check_num_params_;
  vector<float> grad_check_epsilon_;

};
#endif
