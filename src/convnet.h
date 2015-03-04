#ifndef CONVNET_H_
#define CONVNET_H_
#include "layer.h"
#include "hdf5.h"
#include "datahandler.h"
#include "datawriter.h"
#include <vector>
#include <string>

/**
 * A Convolutional Net Model.
 * This class provides the interface for training and using conv nets.
 */
class ConvNet {
 public:
  /**
  * Instantiate a model using the config in model_file.
  */ 
  ConvNet(const std::string& model_file);
  virtual ~ConvNet();
  virtual void SetupDataset(const std::string& train_data_config_file);
  virtual void SetupDataset(const std::string& train_data_config_file, const std::string& val_data_config_file);

  /** Start training.*/
  virtual void Train();

  /** Validate the model on the specfied dataset and return the error.
   * @param dataset The dataset.
   * @param error A vector of errors (one element for each output layer).
   */ 
  void Validate(DataHandler* dataset, std::vector<float>& error);
  
  /** Validate the model on the validation dataset and return the error.
   * @param error A vector of errors (one element for each output layer).
   */ 
  void Validate(std::vector<float>& error);

  /** Write the model to disk.*/
  void Save();

  /** Write the model to disk in the file specified. */
  void Save(const std::string& output_file);

  /** Load the model.*/
  void Load();

  /** Load the model from the file specified.*/
  void Load(const std::string& input_file);

  /** Display the state of the model.
   * Shows the layers and edges for which display is enabled.
   */ 
  void Display();
  
  /** Write the state of the layers to disk.
   * Runs the model on the dataset specified in config and writes
   * the requested layer states out to disk in a hdf5 file.
   * @param config Feature extractor configuartion.
   */ 
  void ExtractFeatures(const config::FeatureExtractorConfig& config);
  void ExtractFeatures(const std::string& config_file);

  /** Allocate memory for the model.
   * @param fprop_only If true, does not allocate memory needed for optimization.
   */ 
  void AllocateMemory(bool fprop_only);

  void SetBatchsize(const int batch_size);
  
  Layer* GetLayerByName(const std::string& name);

  /** Forward propagate through the network.
   * @param train If true, this forward prop is being done during training,
   * otherwise during test/validation. Used for determining whether to use drop
   * units stochastcially or use all of them.
   */ 
  virtual void Fprop(bool train);

 protected:
  /** Creates layers and edges.*/ 
  void BuildNet();

  /** Release all memory held by the model.*/
  void DestroyNet();

  /** Add a sub-network into this network.*/
  void AddSubnet(config::Model& model, const config::Subnet& subnet);

  /** Allocate layer memory for using mini-batches of batch_size_.*/
  void AllocateLayerMemory();

  /** Allocate memory for edges.
   * @param fprop_only If true, does not allocate memory needed for optimization.
   */ 
  void AllocateEdgeMemory(bool fprop_only);
  
  std::string GetCheckpointFilename();
  void TimestampModel();

  /** Sets up fields of view as seen by each location at the output layer.*/
  void FieldsOfView();
  
  /** Topologically sort layers.*/
  void Sort();

  /** Get a mini-batch from dataset.*/
  void GetBatch(DataHandler& dataset);

  /** Forward propagate one layer.
   * Passes up input through the edge and updates the state of the output.
   * @param input the input layer.
   * @param output the output layer.
   * @param edge the edge connecting the input to the output.
   */ 
  void Fprop(Layer& input, Layer& output, Edge& edge, bool train);
  
  /** Back propagate through one layer.
   * Passes down the gradients from the output layer to the input layer.
   * Also updates the weights on the edge (if update_weights is true).
   * @param output the output layer (gradients w.r.t this have been computed).
   * @param input the input layer (gradients w.r.t this will be computed here).
   * @param edge the edge connecting the input to the output.
   * @param train True if this Fprop is done at training time, false otherwise.
   */ 
  virtual void Bprop(Layer& output, Layer& input, Edge& edge);

  /** Backpropagate through the network.*/ 
  virtual void Bprop();
  
  /** Computes the derivative of the loss function.*/ 
  virtual void ComputeDeriv();

  /** Computes the loss function (to be displayed).*/ 
  virtual void GetLoss(std::vector<float>& error);
  
  virtual void UpdateWeights();

  /** Takes one optimization step.*/ 
  virtual void TrainOneBatch(std::vector<float>& error);
  virtual void DisplayLayers();
  void DisplayEdges();
  void InsertPolyak();
  void LoadPolyakWeights();
  void LoadCurrentWeights();
  void WriteLog(int current_iter, float time, float training_error);
  void WriteLog(int current_iter, float time, const std::vector<float>& training_error);
  void WriteValLog(int current_iter, const std::vector<float>& error);
  
  // Data parallel synchronization.
  void Accumulate(Matrix& mat, int tag);
  void Accumulate(std::vector<float>& v, int tag);
  void Broadcast(Matrix& mat);

  /** Decides if learning rate should be reduced.*/
  bool CheckReduceLearningRate(const std::vector<float>& val_error);

  /** Multiply learning rate by factor.*/
  void ReduceLearningRate(const float factor);

  void SetupLocalizationDisplay();
  void DisplayLocalization();

  config::Model model_;  /** The model protobuf config.*/
  std::vector<Layer*> layers_;  /** The layers in the network.*/
  std::vector<Layer*> data_layers_;  /** Layers which have data associated with them.*/
  std::vector<Layer*> input_layers_;  /** Input layers.*/
  std::vector<Layer*> output_layers_;  /** Output layers.*/
  std::vector<Edge*> edges_;  /** The edges in the network.*/
  int max_iter_, batch_size_, current_iter_, lr_reduce_counter_;
  DataHandler *train_dataset_, *val_dataset_;
  std::string checkpoint_dir_, output_file_, model_name_;
  ImageDisplayer displayer_;
  std::string model_filename_, timestamp_, log_file_, val_log_file_;

  // Field of view.
  int fov_size_, fov_stride_, fov_pad1_, fov_pad2_;
  int num_fov_x_, num_fov_y_;
  bool localizer_;

  int process_id_;  // MPI rank.
  int num_processes_;  // Number of data parallel processes.
  bool is_root_;  // The process that is root.

  Matrix parameters_, grad_parameters_;

  ImageDisplayer* localization_display_;

  // Polyak averaging.
  std::vector<Matrix> polyak_parameters_;
  Matrix parameters_backup_;
  int polyak_queue_size_, polyak_index_;
  bool polyak_queue_full_;

  // Data tying.
  std::map<Layer*, Layer*> input_tied_data_layers_, output_tied_data_layers_;
};

#endif
