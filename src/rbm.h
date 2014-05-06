#ifndef RBM_H_
#define RBM_H_
#include "layer.h"
#include "hdf5.h"
#include "datahandler.h"
#include <vector>
#include <string>
using namespace std;

class RBM : public ConvNet {
 public:
  RBM(const string& model_file);
  ~RBM();
  void SetupDataset(const string& train_data_config_file);
  virtual void SetupDataset(const string& train_data_config_file, const string& val_data_config_file);
  virtual void Train();
  void Validate(DataHandler* dataset, vector<float>& error);
  void Validate(vector<float>& error);

  void Save();
  void Save(const string& output_file);
  void Load(bool fprop_only);
  void Load(const string& input_file, bool fprop_only);
  string GetCheckpointFilename();
  void Display();
  void DumpOutputs(const string& output_file, const vector<string>& layer_names);
  void DumpOutputs(const string& output_file, DataHandler* dataset, const vector<string>& layer_names);
  void AllocateMemory(bool fprop_only);
};

#endif
