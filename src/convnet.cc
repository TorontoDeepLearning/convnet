#include "convnet.h"
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <sstream>
#include <stack>
#include <queue>
#include <chrono>
#include <csignal>
#include <iomanip>

using namespace std;
ConvNet::ConvNet(const string& model_file):
  max_iter_(0), batch_size_(0), current_iter_(0),
  lr_reduce_counter_(0),
  train_dataset_(NULL), val_dataset_(NULL),
  model_filename_(model_file),
  polyak_index_(0),
  polyak_queue_full_(false) {

#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id_);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes_);
#else
  process_id_ = 0;
  num_processes_ = 1;
#endif
  is_root_ = process_id_ == 0;
  if (num_processes_ > 1) {
    cout << "Process " << process_id_ << " of " << num_processes_ << endl;
  }

  ReadPbtxt<config::Model>(model_file, model_);
  for (const config::Subnet& subnet : model_.subnet()) {
    AddSubnet(model_, subnet);
  }
  model_.clear_subnet();

  // Use default optimizer for weights/biases whose optimizer is not specified.
  const config::Optimizer& default_w_opt = model_.default_weight_optimizer();
  const config::Optimizer& default_b_opt = model_.default_bias_optimizer();
  for (config::Edge& e : *(model_.mutable_edge())) {
    if (Edge::HasParameters(e)) {
      config::Optimizer w_opt(default_w_opt);
      w_opt.MergeFrom(e.weight_optimizer());
      e.mutable_weight_optimizer()->CopyFrom(w_opt);
      if (!e.has_no_bias()) {
        config::Optimizer b_opt(default_b_opt);
        b_opt.MergeFrom(e.bias_optimizer());
        e.mutable_bias_optimizer()->CopyFrom(b_opt);
      }
    }
  }
  
  Matrix::InitRandom(model_.seed() + process_id_);
  srand(model_.seed() + process_id_);

  localizer_ = model_.localizer();
  model_name_ = model_.name();
  checkpoint_dir_ = model_.checkpoint_dir();

  int num_tstampts = model_.timestamp_size();
  if (num_tstampts > 0) {
    timestamp_ = model_.timestamp(num_tstampts - 1);
  }
  BuildNet();  // Build a net using the connectivity specified in model_.
  polyak_queue_size_ = model_.polyak_queue_size();
  polyak_parameters_.resize(polyak_queue_size_);
}

ConvNet::~ConvNet() {
  DestroyNet();
  if (val_dataset_ != NULL) delete val_dataset_;
  if (train_dataset_ != NULL) delete train_dataset_;
}

void ConvNet::DestroyNet() {
  for (Layer* l: layers_) delete l;
  for (Edge* e: edges_) delete e;
}

void ConvNet::AddSubnet(config::Model& model, const config::Subnet& subnet) {
  config::Model submodel;
  ReadPbtxt<config::Model>(subnet.model_file(), submodel);

  // Recursively add subnets.
  for (const config::Subnet& s : submodel.subnet()) {
    AddSubnet(submodel, s);
  }
  submodel.clear_subnet();

  const string& name = subnet.name();
  int gpu_offset = subnet.gpu_id_offset();
  int num_channels_multiplier = subnet.num_channels_multiplier();
  cout << "Adding subnet " << name << endl;
  map<string, string> merge_layers;
  set<string> remove_layers;
  for (const config::Subnet_MergeLayer& ml : subnet.merge_layer()) {
    merge_layers[ml.subnet_layer()] = ml.net_layer();
  }
  for (const string& l : subnet.remove_layer()) {
    remove_layers.insert(l);
  }
  for (const config::Layer& layer : submodel.layer()) {
    if (merge_layers.find(layer.name()) == merge_layers.end() &&
        remove_layers.find(layer.name()) == remove_layers.end()) {
      config::Layer* l = model.add_layer();
      l->MergeFrom(layer);
      l->set_name(name + "_" + l->name());
      l->set_gpu_id(gpu_offset + l->gpu_id());
      l->set_num_channels(num_channels_multiplier * l->num_channels());
    }
  }
  int soa = subnet.start_optimization_after(); 
  for (const config::Edge& edge : submodel.edge()) {
    if (remove_layers.find(edge.source()) != remove_layers.end() ||
        remove_layers.find(edge.dest()) != remove_layers.end()) continue;
    config::Edge* e = model.add_edge();
    e->MergeFrom(edge);
    e->set_gpu_id(gpu_offset + e->gpu_id());
    if (!subnet.parameters_file().empty()) {
      e->set_initialization(config::Edge::PRETRAINED);
      e->set_pretrained_model(subnet.parameters_file());
      e->set_pretrained_edge_name(edge.source() + ":" + edge.dest());
    }
    map<string, string>::iterator it1 = merge_layers.find(edge.source());
    map<string, string>::iterator it2 = merge_layers.find(edge.dest());
    e->set_source((it1 == merge_layers.end()) ? (name + "_" + e->source()):
                                                it1->second);
    e->set_dest((it2 == merge_layers.end()) ? (name + "_" + e->dest()):
                                               it2->second);
    e->set_block_backprop(edge.block_backprop() | subnet.block_backprop());
    e->mutable_weight_optimizer()->set_start_optimization_after(soa);
    e->mutable_bias_optimizer()->set_start_optimization_after(soa);
  }
}

void ConvNet::BuildNet() {
  for (const config::Layer& l : model_.layer()) {
    layers_.push_back(Layer::ChooseLayerClass(l));
  }

  // Setup edges.
  for (const config::Edge& e : model_.edge()) {
    edges_.push_back(Edge::ChooseEdgeClass(e));
  }

  // Communicate information about tied edges.
  map<string, Edge*> edge_name_map;
  for (Edge* e: edges_) {
    edge_name_map[e->GetName()] = e;
  }
  for (Edge* e: edges_) {
    if (e->IsTied()) {
      e->SetTiedTo(edge_name_map[e->GetTiedEdgeName()]);  // TODO: Check if not found.
    }
  }

  // Setup layers.
  for (Layer* l : layers_) {
    const string& name = l->GetName();
    for (Edge* e : edges_) {
      if (name.compare(e->GetSourceName()) == 0) {
        l->AddOutgoing(e);
        e->SetSource(l);
        e->SetInputChannels(l->GetNumChannels(e->GetSourceSliceName()));
      }
      if (name.compare(e->GetDestName()) == 0) {
        l->AddIncoming(e);
        e->SetDest(l);
        e->SetOutputChannels(l->GetNumChannels(e->GetDestSliceName()));
      }
    }
  }

  // Topologically sort layers.
  Sort();
  // layers_ now contains the layers in an fprop-safe order.

  for (Layer* l : layers_) {
    bool is_input = l->incoming_edge_.size() == 0;
    bool is_output = l->outgoing_edge_.size() == 0;
    bool has_tied_data = l->HasTiedData();
    Layer* t = has_tied_data? GetLayerByName(l->GetTiedDataLayerName()) : NULL;
    if (is_input) {
      input_layers_.push_back(l);
      if (has_tied_data) {
        input_tied_data_layers_[l] = t; 
      } else {
        data_layers_.push_back(l);
      }
    }
    if (is_output) {
      output_layers_.push_back(l);
      if (has_tied_data) {
        output_tied_data_layers_[l] = t; 
      } else {
        data_layers_.push_back(l);
      }
    }
  }

  int image_size_y, image_size_x, image_size_t;
  for (Layer* l : layers_) {
    if (l->IsInput()) {
      image_size_y = l->GetSizeY();
      image_size_x = l->GetSizeX();
      image_size_t = l->GetSizeT();
      if (image_size_y <= 0) image_size_y = model_.patch_size();
      if (image_size_x <= 0) image_size_x = model_.patch_size();
      if (image_size_t <= 0) image_size_t = 1;
    } else {
      image_size_y = l->incoming_edge_[0]->GetNumModulesY();
      image_size_x = l->incoming_edge_[0]->GetNumModulesX();
      image_size_t = l->incoming_edge_[0]->GetNumModulesT();
    }
    l->SetSize(image_size_y, image_size_x, image_size_t);
    for (Edge* e: l->outgoing_edge_) {
      e->SetImageSize(image_size_y, image_size_x, image_size_t);
    }
  }

  if (localizer_) {
    FieldsOfView();
  }
}

void ConvNet::FieldsOfView() {
  Layer* l = output_layers_[0];
  fov_size_ = 1, fov_stride_ = 1, fov_pad1_ = 0, fov_pad2_ = 0;
  while(!l->IsInput()) {
    Edge* e = l->incoming_edge_[0];
    e->FOV(&fov_size_, &fov_stride_, &fov_pad1_, &fov_pad2_);
    l = e->GetSource();
  }
  float image_size = (float)model_.patch_size();

  cout << "FOV: " << fov_size_ << " " << fov_stride_ << " " << fov_pad1_ << " " << fov_pad2_ << endl;
  cout << "Image size " << image_size << endl;
  
  num_fov_x_ = output_layers_[0]->GetSizeX();
  num_fov_y_ = output_layers_[0]->GetSizeY();
}

void ConvNet::AllocateLayerMemory() {
  for (Layer* l : layers_) {
    l->AllocateMemory(batch_size_);
  }
}

void ConvNet::AllocateEdgeMemory(bool fprop_only) {
  size_t total_memory_usage = 0;
  map<Edge*, size_t> memory_usage;
  for (Edge* e : edges_) {
    size_t mem = e->GetParameterMemoryRequirement();  // Number of floats
    memory_usage[e] = mem;
    // Align to 512-bytes. Weights being used through texture memory need 512-byte aligned address.
    mem = ((mem + 127) / 128) * 128;
    total_memory_usage += mem;
  }
  parameters_.AllocateGPUMemory(1, total_memory_usage);
  if (!fprop_only) grad_parameters_.AllocateGPUMemory(1, total_memory_usage);
  size_t offset = 0;

  for (Edge* e : edges_) {
    size_t mem = memory_usage[e];
    if (mem == 0) continue;
    Matrix slice;
    parameters_.GetSlice(slice, offset, offset + mem);
    e->SetMemory(slice);
    if (!fprop_only) {
      Matrix grad_slice;
      grad_parameters_.GetSlice(grad_slice, offset, offset + mem);
      e->SetGradMemory(grad_slice);
    }
    offset += ((mem + 127) / 128) * 128;
  }

  if (is_root_) {
    if (timestamp_.empty()) {
      // Initialize randomly.
      for (Edge* e: edges_) e->Initialize();
    } else {
      // Initialize from a saved model.
      Load();
    }
  }
  if (num_processes_ > 1) Broadcast(parameters_);
}

void ConvNet::Sort() {
  Layer *m, *n;
  vector<Layer*> L;
  //stack<Layer*> S;  // Depth-first sort.
  queue<Layer*> S;  // Breadth-first sort.
  // Breadth-first usually works well for multi-gpu multi-column nets where
  // each column has roughly the same amount of work.
  
  for (Layer* l : layers_) if (l->IsInput()) S.push(l);
  if (S.empty()) {
    cerr << "Error: No layer is set to be input!" << endl;
    exit(1);
  }
  bool x;

  while (!S.empty()) {
    //n = S.top();
    n = S.front();
    S.pop();
    L.push_back(n);
    for (Edge* e : n->outgoing_edge_) {
      e->SetMark();
      m = e->GetDest();
      if (m == NULL) {
        cerr << "Edge " << e->GetName() << " has no destination layer" << endl;
        exit(1);
      }
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

void ConvNet::Fprop(Layer& input, Layer& output, Edge& edge) {
  Matrix& input_state = input.GetState(edge.GetSourceSliceName());
  Matrix& output_state = output.GetState(edge.GetDestSliceName());
  bool overwrite = output.AddOrOverwriteState(edge.GetDestSliceName());
  //cout << "Fprop from " << input.GetName() << " slice " << edge.GetSourceSliceName() << " to " << output.GetName() << " slice " << edge.GetDestSliceName() << " overwrite " << overwrite << endl;
  edge.ComputeUp(input_state, output_state, overwrite);
}

void ConvNet::Bprop(Layer& output, Layer& input, Edge& edge) {
  if (edge.IsBackPropBlocked()) return;
  Matrix& input_state = input.GetState(edge.GetSourceSliceName());
  Matrix& output_state = output.GetState(edge.GetDestSliceName());
  Matrix& input_deriv = input.GetDeriv(edge.GetSourceSliceName());
  Matrix& output_deriv = output.GetDeriv(edge.GetDestSliceName());

  edge.ComputeOuter(input_state, output_deriv);
  if (!input.IsInput()) {
    bool overwrite = input.AddOrOverwriteDeriv(edge.GetSourceSliceName());
  //cout << "Bprop to " << input.GetName() << " slice " << edge.GetSourceSliceName() << " from " << output.GetName() << " slice " << edge.GetDestSliceName() << " overwrite " << overwrite << endl;
    edge.ComputeDown(output_deriv, input_state, output_state,
                     input_deriv, overwrite);
  }
}

void ConvNet::Fprop(bool train) {
  for(Layer* l : layers_) {
    for (Edge* e : l->incoming_edge_) {
      Fprop(*(e->GetSource()), *l, *e);
    }
    if (!l->IsInput()) l->ApplyActivation();
    l->ApplyDropout(train);
  }
}

void ConvNet::Bprop() {
  Layer *l;
  for (int i = layers_.size() - 1; i >= 0; i--) {
    l = layers_[i];
    for (Edge* e : l->outgoing_edge_) {
      Bprop(*(e->GetDest()), *l, *e);
    }
    l->ApplyDerivativeofDropout();
    if (!l->IsInput() && !l->IsOutput()) {
      l->ApplyDerivativeOfActivation();
    }
  }
}

void ConvNet::Broadcast(Matrix& mat) {
#ifdef USE_MPI
  if (is_root_) mat.CopyToHost();
  MPI_Bcast(mat.GetHostData(), mat.GetNumEls(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  if (!is_root_) mat.CopyToDevice();
#endif
}

void ConvNet::Accumulate(Matrix& mat, int tag) {
#ifdef USE_MPI
  float* data = mat.GetHostData();
  int buffer_size = mat.GetNumEls();
  mat.CopyToHost();
  if (is_root_) {
    float* buffer = new float[buffer_size];
    MPI_Status stat;
    for (int pid = 1; pid < num_processes_; pid++) {
      MPI_Recv(buffer, buffer_size, MPI_FLOAT, pid, tag, MPI_COMM_WORLD, &stat);
      /*
      if (stat != MPI_SUCCESS) {
        cerr << "Error: Could not receive message from pid " << pid << endl;
      }*/
      for (size_t i = 0; i < buffer_size; i++) data[i] += buffer[i];
    }
    for (size_t i = 0; i < buffer_size; i++) data[i] /= num_processes_;
    delete[] buffer;
    mat.CopyToDevice();
  } else {
    MPI_Send(data, buffer_size, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
  }
#endif
}

void ConvNet::UpdateWeights() {
  if (num_processes_ > 1) Accumulate(grad_parameters_, MPITAG_WEIGHTGRAD);
  if (num_processes_ > 1) Broadcast(grad_parameters_);
  //if (is_root_) {
    for (Edge* e : edges_) {
      if (!e->IsBackPropBlocked()) e->UpdateWeights();
    }
  //}
  //parameters_.Print();
  //if (num_processes_ > 1) Broadcast(parameters_);
}

void ConvNet::ComputeDeriv() {
  for (Layer* l: output_layers_) l->ComputeDeriv();
}

void ConvNet::GetLoss(vector<float>& error) {
  error.clear();
  for (Layer* l: output_layers_) {
    error.push_back(l->GetPerformanceMetric());
  }
}

void ConvNet::GetBatch(DataHandler& dataset) {
  dataset.GetBatch(data_layers_);
  for (auto& kv : input_tied_data_layers_) {
    Layer* src = kv.second, *dst = kv.first;
    dst->GetState().Set(src->GetState());
  }
  for (auto& kv : output_tied_data_layers_) {
    Layer* src = kv.second, *dst = kv.first;
    dst->GetData().Set(src->GetData());
  }
}

void ConvNet::TrainOneBatch(vector<float>& error) {
  for (Layer* l : layers_) l->ResetAddOrOverwrite();
  for (Edge* e : edges_) e->NotifyStart();
  GetBatch(*train_dataset_);
  Fprop(true);
  ComputeDeriv();
  GetLoss(error);
  Bprop();
  UpdateWeights();
}

void ConvNet::SetupDataset(const string& train_data_config_file) {
  SetupDataset(train_data_config_file, "");
}

void ConvNet::SetBatchsize(const int batch_size) {
  batch_size_ = batch_size;
}

void ConvNet::SetupDataset(const string& train_data_config_file,
                           const string& val_data_config_file) {

  if (!train_data_config_file.empty()) {
    model_.clear_train_dataset();
    ReadPbtxt<config::DatasetConfig>(train_data_config_file,
                                     *(model_.mutable_train_dataset()));
  }
  if (!val_data_config_file.empty()) {
    model_.clear_valid_dataset();
    ReadPbtxt<config::DatasetConfig>(val_data_config_file,
                                     *(model_.mutable_valid_dataset()));
  }

  if (!model_.has_train_dataset()) {
    cerr << "Error: No training set provided." << endl;
    exit(1);
  }
  if (!model_.has_valid_dataset()) {
    cerr << "Warning: No validation set provided." << endl;
  }

  train_dataset_ = new DataHandler(model_.train_dataset());
  if (localizer_) {
    train_dataset_->SetFOV(fov_size_, fov_stride_, fov_pad1_, fov_pad2_,
                           model_.patch_size(), num_fov_x_, num_fov_y_);
  }
  SetBatchsize(train_dataset_->GetBatchSize());
  int dataset_size = train_dataset_->GetDataSetSize();
  train_dataset_->AllocateMemory();
  cout << "Training data set size " << dataset_size << endl;
  if (model_.has_valid_dataset()) {
    val_dataset_ = new DataHandler(model_.valid_dataset());
    if (localizer_) {
      val_dataset_->SetFOV(fov_size_, fov_stride_, fov_pad1_, fov_pad2_,
                           model_.patch_size(), num_fov_x_, num_fov_y_);
    }
    dataset_size = val_dataset_->GetDataSetSize();
    val_dataset_->AllocateMemory();
    cout << "Validation data set size " << dataset_size << endl;
  }
}

void ConvNet::AllocateMemory(bool fprop_only) {
  AllocateLayerMemory();
  AllocateEdgeMemory(fprop_only);
  if (is_root_) {
    for (Edge* e : edges_) {
      cout << e->GetDescription() << endl;
    }
  }
}

void ConvNet::Accumulate(vector<float>& v, int tag) {
#ifdef USE_MPI
  int buffer_size = v.size();
  if (is_root_) {
    float* buffer = new float[buffer_size];
    MPI_Status stat;
    for (int pid = 1; pid < num_processes_; pid++) {
      MPI_Recv(buffer, buffer_size, MPI_FLOAT, pid, tag, MPI_COMM_WORLD, &stat);
      /*if (stat != MPI_SUCCESS) {
        cerr << "Error: Could not receive message from pid " << pid << endl;
      }*/
      for (int i = 0; i < buffer_size; i++) v[i] += buffer[i];
    }
    delete buffer;
  } else {
    float* data = new float[v.size()];
    for (int i = 0; i < v.size(); i++) data[i] = v[i];
    MPI_Send(data, buffer_size, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
    delete data;
  }
#endif
}

void ConvNet::Validate(DataHandler* dataset, vector<float>& total_error) {
  if (dataset == NULL) return;
  vector<float> error;
  dataset->Seek(0);
  int dataset_size = dataset->GetDataSetSize(),
      batch_size = dataset->GetBatchSize(),
      num_batches = dataset_size / batch_size;
  for (int k = 0; k < num_batches; k++) {
    for (Layer* l : layers_) l->ResetAddOrOverwrite();
    GetBatch(*dataset);
    Fprop(false);
    GetLoss(error);
    if (total_error.size() != error.size()) total_error.resize(error.size());
    for (int i = 0; i < error.size(); i++) {
      total_error[i] = (total_error[i] * k) / (k+1) + error[i] / (batch_size * (k+1));
    }
  }
  dataset->Sync();
}

Layer* ConvNet::GetLayerByName(const string& name) {
  for (Layer* l:layers_) {
    if (l->GetName().compare(name) == 0) return l;
  }
  cerr << "Error: No layer called " << name << endl;
  exit(1);
  return NULL;
}

void ConvNet::ExtractFeatures(const string& config_file) {
  config::FeatureExtractorConfig config;
  ReadPbtxt<config::FeatureExtractorConfig>(config_file, config);
  ExtractFeatures(config);
}

void ConvNet::ExtractFeatures(const config::FeatureExtractorConfig& config) {
  DataHandler dataset = DataHandler(config.input());
  int dataset_size = dataset.GetDataSetSize(),
      batch_size = dataset.GetBatchSize(),
      num_batches = dataset_size / batch_size,
      left_overs = dataset_size % batch_size,
      multiplicity = dataset.GetMultiplicity();
  SetBatchsize(batch_size);
  dataset.AllocateMemory();
  AllocateMemory(true);

  const int display_after = model_.display_after();
  const bool display = model_.display();
  if (display && localizer_) {
    SetupLocalizationDisplay();
  }
 
  cout << "Extracting features for dataset of size " << dataset_size
       << " # batches " << num_batches
       << " # left overs " << left_overs << endl;
  if (left_overs > 0) num_batches++;
  cout << "Writing to " << config.output_file() << endl;
 
  DataWriter* data_writer = new DataWriter(config);
  data_writer->SetDataSetSize(dataset_size * multiplicity);
  vector<Layer*> layers;
  for (const config::FeatureStreamConfig& feature : config.feature()) {
    Layer* l = GetLayerByName(feature.layer());
    int numdims = l->GetState().GetCols();
    data_writer->SetNumDims(l->GetName(), numdims);
    layers.push_back(l);
  }
  int numcases;
  for (int k = 0; k < num_batches; k++) {
    cout << "\rBatch " << (k+1);
    cout.flush();
    numcases = (left_overs > 0 && k == num_batches - 1) ? left_overs : batch_size;
    for (int m = 0; m < multiplicity; m++) {
      for (Layer* l : layers_) l->ResetAddOrOverwrite();
      GetBatch(dataset);
      if (display && k % display_after == 0) {
        DisplayLayers();
        if (localizer_) DisplayLocalization();
      }
      Fprop(false);
      data_writer->Write(layers, numcases);
    }
  }
  cout << endl;
  delete data_writer;
  dataset.Sync();
}

void ConvNet::Save() {
  const string& fname = GetCheckpointFilename();
  Save(fname);
  if (model_.polyak_after() > 0) {
    LoadPolyakWeights();  // Load averaged weights.
    Save(fname + "polyak");
    LoadCurrentWeights();  // Restore original weights.
  }
}

void ConvNet::Save(const string& output_file) {
  cout << "Saving model to " << output_file << endl;
  // Save to temp file.
  string output_file_temp = output_file + "temp";
  hid_t file = H5Fcreate(output_file_temp.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  for (Edge* e : edges_) e->SaveParameters(file);
  WriteHDF5IntAttr(file, "__lr_reduce_counter__", &lr_reduce_counter_);
  WriteHDF5IntAttr(file, "__current_iter__", &current_iter_);
  H5Fclose(file);
  cout << " .. Done" << endl;
  // Move to original file.
  int result = rename(output_file_temp.c_str(), output_file.c_str());
  if (result != 0) {
    cerr << "Error renaming file." << endl;
  }
}

void ConvNet::InsertPolyak() {
  if (polyak_queue_size_ == 0) return;
  
  // Weights are stored in main memory to minimize GPU memory usage.
  // This means that we have to copy parameters to the main memory.
  // But that should be ok because this needs to be done a few times
  // only before validation and checkpointing, and not all the time.
  Matrix& w = polyak_parameters_[polyak_index_];
  if (w.GetNumEls() == 0) {
    w.AllocateMainMemory(parameters_.GetRows(), parameters_.GetCols());
  }
  parameters_.CopyToHost();
  w.CopyFromMainMemory(parameters_);

  polyak_index_++;
  if (polyak_index_ == polyak_queue_size_) {
    polyak_index_ = 0;
    polyak_queue_full_ = true;
  }
}

void ConvNet::LoadPolyakWeights() {
  if (polyak_queue_size_ == 0) return;
  if (parameters_backup_.GetNumEls() == 0) {
    parameters_backup_.AllocateMainMemory(parameters_.GetRows(), parameters_.GetCols());
  }
  parameters_.CopyToHost();
  parameters_backup_.CopyFromMainMemory(parameters_);

  int max_ind = polyak_queue_full_ ? polyak_queue_size_: polyak_index_;

  float *parameter_avg = parameters_.GetHostData(), *w;
  for (int j = 0; j < parameters_.GetNumEls(); j++) parameter_avg[j] = 0;
  for (int i = 0; i < max_ind; i++) {
    w = polyak_parameters_[i].GetHostData();
    for (int j = 0; j < parameters_.GetNumEls(); j++) parameter_avg[j] += w[j];
  }
  for (int j = 0; j < parameters_.GetNumEls(); j++) parameter_avg[j] /= max_ind;
  parameters_.CopyToDevice();
}

void ConvNet::LoadCurrentWeights() {
  parameters_.CopyFromMainMemory(parameters_backup_);
  parameters_.CopyToDevice();
}

string ConvNet::GetCheckpointFilename() {
  string filename = checkpoint_dir_ + "/" + model_name_ + "_" + timestamp_ + ".h5";
  return filename;
}

void ConvNet::Load() {
  Load(GetCheckpointFilename());
}

void ConvNet::Load(const string& input_file) {
  cout << "Loading model from " << input_file << endl;
  hid_t file = H5Fopen(input_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  for (Edge* e : edges_) e->LoadParameters(file);
  ReadHDF5IntAttr(file, "__lr_reduce_counter__", &lr_reduce_counter_);
  for (int i = 0; i < lr_reduce_counter_; i++) {
    ReduceLearningRate(model_.reduce_lr_factor());
  }
  ReadHDF5IntAttr(file, "__current_iter__", &current_iter_);
  H5Fclose(file);
}

void ConvNet::DisplayLayers() {
  for (int i = 0; i < layers_.size(); i++){
    layers_[i]->Display(0);
  }
}

void ConvNet::DisplayEdges() {
  for (int i = 0; i < edges_.size(); i++){
    edges_[i]->DisplayWeights();
  }
}

void ConvNet::WriteLog(int current_iter, float time, float error) {
  vector<float> temp(1);
  temp[0] = error;
  WriteLog(current_iter, time, temp);
}

void ConvNet::WriteLog(int current_iter, float time, const vector<float>& error) {
  ofstream f(log_file_, ofstream::out | ofstream::app);
  f << current_iter << " " << time;
  for (const float& val: error) f << " " << val;
  f << endl;
  f.close();
}

void ConvNet::WriteValLog(int current_iter, const vector<float>& error) {
  ofstream f(val_log_file_, ofstream::out | ofstream::app);
  f << current_iter;
  for (const float& val: error) f << " " << val;
  f << endl;
  f.close();
}

// Look at the history of val_error to see if we should reduce the learning rate now.
bool ConvNet::CheckReduceLearningRate(const vector<float>& val_error) {
  const int len = val_error.size();
  /*
  const int threshold = 1;
  int argmax = 0;
  for (int i = 0; i < len; i++) {
    if (val_error[argmax] < val_error[i]) argmax = i;
  }
  cout << "Argmax : " << argmax << " len " << len << endl;
  return (len - argmax > threshold);
  */
  const int num_steps = model_.reduce_lr_num_steps();
  /*
  bool r = len >= num_steps;
  for (int i = 0; i < num_steps - 1 && r; i++) {
    float v1 = val_error[len - i - 1], v2 = val_error[len - i - 2];
    r &= smaller_is_better_ ? (v1 >= v2) : (v1 <= v2);
  }
  */
  if (len < num_steps) return false;
  int i = len - num_steps;
  float mean1 = 0, mean2 = 0;
  for (int j = 0; j < num_steps/2; j++) {
    mean1 = (mean1 * j) / (j+1) + val_error[i++] / (j+1);
  }
  for (int j = 0; j < num_steps - num_steps/2; j++) {
    mean2 = (mean2 * j) / (j+1) + val_error[i++] / (j+1);
  }
  float diff = model_.smaller_is_better() ? mean1 - mean2 : mean2 - mean1;
  return diff < model_.reduce_lr_threshold();
}

void ConvNet::ReduceLearningRate(const float factor) {
  for (Edge* e : edges_) {
    e->ReduceLearningRate(factor);
  }
}

void ConvNet::Validate(vector<float>& error) {
  Validate(val_dataset_, error);
}

void ConvNet::TimestampModel() {
  timestamp_ = GetTimeStamp();
  string fname = checkpoint_dir_ + "/" + model_name_ + "_" + timestamp_;
  cout << "Checkpointing at " << fname << endl;
  model_.add_timestamp(timestamp_);
  WritePbtxt<config::Model>(fname + ".pbtxt", model_);
  log_file_ = fname + "_train.log";
  val_log_file_ = fname + "_valid.log";
}

void ConvNet::SetupLocalizationDisplay() {
  int image_size = model_.patch_size();
  localization_display_ = new ImageDisplayer(image_size, image_size, 3, false,
                                          "localization");
  localization_display_->SetFOV(fov_size_, fov_stride_, fov_pad1_, fov_pad2_,
                                image_size, num_fov_x_, num_fov_y_);
}

void ConvNet::DisplayLocalization() {
  Layer *input_layer = input_layers_[0],
        *output_layer = output_layers_[0];
  Matrix& input = input_layer->GetState();
  Matrix& output = output_layer->GetState();
  Matrix& ground_truth = output_layer->GetData();

  input.CopyToHost();
  output.CopyToHost();
  ground_truth.CopyToHost();

  float *data = input.GetHostData(),
        *gt = ground_truth.GetHostData(),
        *preds = output.GetHostData();
  
  localization_display_->DisplayLocalization(data, preds, gt, input.GetRows());
}

void ConvNet::Train() {

  // Check if train data is available.
  if (train_dataset_ == NULL) {
    cerr << "Error: Train dataset is NULL." << endl;
    exit(1);
  }

  // Before starting the training, mark this model with a timestamp.
  if (is_root_) TimestampModel();

  const int display_after = model_.display_after(),
            print_after = model_.print_after(),
            validate_after = model_.validate_after(),
            save_after = model_.save_after(),
            polyak_after = model_.polyak_after(),
            start_polyak_queue_val = validate_after - polyak_after * model_.polyak_queue_size(),
            start_polyak_queue_save = save_after - polyak_after * model_.polyak_queue_size();

  int lr_reduce_layer_id = 0;
  const string& reduce_lr_layer_name = model_.reduce_lr_layer_name();
  if (!reduce_lr_layer_name.empty()) {
    int layer_id = 0;
    lr_reduce_layer_id = -1;
    for (Layer *l : output_layers_) {
      if (l->GetName().compare(reduce_lr_layer_name) == 0) {
        lr_reduce_layer_id = layer_id;
      }
      layer_id++;
    }
    if (lr_reduce_layer_id < 0) {
      cerr << "No such output layer " << reduce_lr_layer_name << endl;
      exit(1);
    }
  }

  const bool display = model_.display(), print_weights = model_.print_weights();

  if (display && localizer_) {
    SetupLocalizationDisplay();
  }
  const float learning_rate_reduce_factor = model_.reduce_lr_factor();

  // Time keeping.
  chrono::time_point<chrono::system_clock> start_t, end_t;
  chrono::duration<double> time_diff;
  start_t = chrono::system_clock::now();

  vector<float> train_error, this_train_error;
  vector<float> val_error, this_val_error;
  int dont_reduce_lr = 0;
  const int lr_max_reduce = model_.reduce_lr_max();
  bool newline;

  Matrix::ShowMemoryUsage();
  for(int i = current_iter_; i < model_.max_iter(); i++) {
    current_iter_++;
    if (is_root_) {
      cout << "\rStep " << current_iter_;
      cout.flush();
    }

    TrainOneBatch(this_train_error);
    AddVectors(train_error, this_train_error);

    if (i % display_after == 0 && display) {
      DisplayLayers();
      DisplayEdges();
      if (localizer_) DisplayLocalization();
    }
    newline = false;
    if ((i+1) % print_after == 0) {
#ifdef USE_MPI
      if (num_processes_ > 1) Accumulate(train_error, MPITAG_TRAINERROR);
#endif
      if (is_root_) {
        end_t = chrono::system_clock::now();
        time_diff = end_t - start_t;
        cout << setprecision(5);
        cout << " Time " << time_diff.count() << " s";
        cout << " Train Acc :";
        for (float& err : train_error) err /= print_after * batch_size_ * num_processes_;
        for (const float& err : train_error) cout << " " << err;
        WriteLog(current_iter_, time_diff.count(), train_error);
        if (print_weights) {
          cout << " Weight length:";
          for (Edge* e : edges_) {
            if (e->HasNoParameters() || e->IsTied()) continue;
            cout << " " << e->GetRMSWeight();
          }
        }
        start_t = end_t;
      }
      train_error.clear();
      newline = true;
    }

    // Start inserting parameters into a queue so that they can be averaged
    // for validation or for checkpointing.
    if (polyak_after > 0 && (i+1) % polyak_after == 0 &&
        (((i+1) % validate_after) >= start_polyak_queue_val ||
         ((i+1) % save_after) >= start_polyak_queue_save)) {
      InsertPolyak();
    }

    if (val_dataset_ != NULL && validate_after > 0 &&
        (i+1) % validate_after == 0) {
      if (polyak_after > 0) LoadPolyakWeights();  // Load averaged weights.
      train_dataset_->Sync();
      Validate(this_val_error);
      //if (polyak_after > 0) LoadCurrentWeights();  // Restore original weights.

      val_error.push_back(this_val_error[lr_reduce_layer_id]);
      if (is_root_) {
        cout << " Val Acc :";
        for (const float& val: this_val_error) cout << " " << val;
        WriteValLog(current_iter_, this_val_error);
      }

      // Should we reduce the learning rate ?
      if (learning_rate_reduce_factor < 1.0) {
        bool reduce_learning_rate = CheckReduceLearningRate(val_error);
        if (reduce_learning_rate && lr_reduce_counter_ < lr_max_reduce
            && dont_reduce_lr-- < 0) {
          dont_reduce_lr = model_.reduce_lr_num_steps();
          cout << "Learning rate reduced " << ++lr_reduce_counter_ << " time(s).";
          ReduceLearningRate(learning_rate_reduce_factor);
        }
      }
      newline = true;
    }
    if (is_root_ && newline) cout << endl;
    if ((i+1) % save_after == 0) {
      train_dataset_->Sync();
      if (is_root_) Save();
    }
  }
  if (model_.max_iter() % save_after != 0) {
    train_dataset_->Sync();
    if (is_root_) Save();
  }
  if (is_root_) cout << "End of training." << endl;
  if (display && localizer_) {
    delete localization_display_;
  }
}
