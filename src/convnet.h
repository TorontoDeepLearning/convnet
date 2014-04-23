#ifndef CONVNET_H_
#define CONVNET_H_
#include "layer.h"
#include "hdf5.h"
#include "datahandler.h"
#include <vector>
#include <string>
using namespace std;

class ConvNet {
 public:
  ConvNet(const string& model_file);
  ~ConvNet();
  void SetupDataset(const string& train_data_config_file);
  void SetupDataset(const string& train_data_config_file, const string& val_data_config_file);
  virtual void Train();
  void Validate(DataHandler* dataset, vector<float>& error);
  void Validate(vector<float>& error);

  void Save();
  void Save(const string& output_file);
  void Load();
  void Load(const string& input_file);
  string GetCheckpointFilename();
  void Display();
  void DumpOutputs(const string& output_file, const vector<string>& layer_names);
  void DumpOutputs(const string& output_file, DataHandler* dataset, const vector<string>& layer_names);

 protected:
  void BuildNet();
  void DestroyNet();
  void AllocateMemory(int batch_size);
  void TimestampModel();
  void Sort();
  void Fprop(Layer& input, Layer& output, Edge& edge, bool overwrite);
  virtual void Bprop(Layer& output, Layer& input, Edge& edge, bool overwrite, bool update_weights);
  virtual void Fprop(bool train);
  virtual void Bprop(bool update_weights);
  virtual void ComputeDeriv();
  virtual void GetLoss(vector<float>& error);
  virtual void TrainOneBatch(vector<float>& error);
  void DisplayLayers();
  void DisplayEdges();
  void InsertPolyak();
  void LoadPolyakWeights();
  void LoadCurrentWeights();
  void WriteLog(int current_iter, float time, float training_error);
  void WriteLog(int current_iter, float time, const vector<float>& training_error);
  void WriteValLog(int current_iter, const vector<float>& error);
  Layer* GetLayerByName(const string& name);
  void DumpOutputs(const string& output_file, DataHandler* dataset, vector<Layer*>& layers);
  
  // Decides if learning rate should be reduced.
  bool CheckReduceLearningRate(const vector<float>& val_error);

  // Multiply learning rate by factor.
  void ReduceLearningRate(const float factor);

  config::Model* model_;
  vector<Layer*> layers_, data_layers_, input_layers_, output_layers_;
  vector<Edge*> edges_;
  int max_iter_, batch_size_, current_iter_, lr_reduce_counter_;
  DataHandler *train_dataset_, *val_dataset_;
  string checkpoint_dir_, output_file_, model_name_;
  ImageDisplayer displayer_;
  string model_filename_, timestamp_, log_file_, val_log_file_;

  // a+=b;
  static void AddVectors(vector<float>& a, vector<float>& b);
};

#endif
