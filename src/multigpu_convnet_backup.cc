#include "multigpu_convnet.h"
#include <thread>

//mutex MultiGPUConvNet::fprop_start_mutex_, MultiGPUConvNet::fprop_finished_mutex_;
//condition_variable MultiGPUConvNet::fprop_finish_cond_var_, MultiGPUConvNet::fprop_start_cond_var_;
mutex fprop_start_mutex_, fprop_finished_mutex_;
condition_variable fprop_finish_cond_var_, fprop_start_cond_var_;

MultiGPUConvNet::MultiGPUConvNet(const string& model_file):
  ConvNet(model_file), ready_for_fprop_(false), fprop_finish_(false), stop_fprop_(false) {}

void MultiGPUConvNet::WaitForFpropStartSignal() {
  unique_lock<mutex> lock1(fprop_start_mutex_);
  while (!ready_for_fprop_) {
    fprop_start_cond_var_.wait(lock1);
  }
  ready_for_fprop_ = false;
}

void MultiGPUConvNet::NotifyFpropComplete() {
  unique_lock<mutex> lock2(fprop_finished_mutex_);
  fprop_finish_ = true;
  fprop_finish_cond_var_.notify_all();
}

void MultiGPUConvNet::StartFprop() {
  unique_lock<mutex> lock1(fprop_start_mutex_);
  ready_for_fprop_ = true;
  fprop_start_cond_var_.notify_all();
}

void MultiGPUConvNet::WaitForFprop() {
  unique_lock<mutex> lock2(fprop_finished_mutex_);
  while (!fprop_finish_) {
    fprop_finish_cond_var_.wait(lock2);
  }
  fprop_finish_ = false;
}

void MultiGPUConvNet::Fprop(bool train) {
  if (train) {
    while (!stop_fprop_) {
      WaitForFpropStartSignal();
      ConvNet::Fprop(train);
      NotifyFpropComplete();  
    }
  } else {
    ConvNet::Fprop(train);
  }
}

void MultiGPUConvNet::TrainOneBatch(vector<float>& error) {
  train_dataset_->GetBatch(data_layers_);

  StartFprop();
  WaitForFprop();

  ComputeDeriv();
  GetLoss(error);
  Bprop(true);
}

void MultiGPUConvNet::Train() {

  // Check if train data is available.
  if (train_dataset_ == NULL) {
    cerr << "Error: Train dataset is NULL." << endl;
    exit(1);
  }

  // If timestamp is present, then initialize model at that timestamp.
  if (!timestamp_.empty()) Load();

  // Before starting the training, mark this model with a timestamp.
  TimestampModel();

  const int display_after = model_->display_after(),
            print_after = model_->print_after(),
            validate_after = model_->validate_after(),
            save_after = model_->save_after(),
            polyak_after = model_->polyak_after(),
            start_polyak_queue = validate_after - polyak_after * model_->polyak_queue_size();

  const bool display = model_->display();
             //display_spatial_output = model_->display_spatial_output();

  const float learning_rate_reduce_factor = model_->reduce_lr_factor();

  // Time keeping.
  chrono::time_point<chrono::system_clock> start_t, end_t;
  chrono::duration<double> time_diff;
  start_t = chrono::system_clock::now();

  vector<float> train_error, this_train_error;
  vector<float> val_error, this_val_error;
  int dont_reduce_lr = 0;
  const int lr_max_reduce = model_->reduce_lr_max();
  bool newline;

  thread train_fprop_thread(&MultiGPUConvNet::Fprop, this, true);
  for(int i = current_iter_; i < model_->max_iter(); i++) {
    current_iter_++;
    cout << "\rStep " << current_iter_;
    cout.flush();

    TrainOneBatch(this_train_error);
    AddVectors(train_error, this_train_error);

    if (i % display_after == 0 && display) {
      DisplayLayers();
      DisplayEdges();
    }
    newline = false;
    if ((i+1) % print_after == 0) {
      end_t = chrono::system_clock::now();
      time_diff = end_t - start_t;
      printf(" Time %f s Train Acc :", time_diff.count());
      for (float& err : train_error) err /= print_after * batch_size_;
      for (const float& err : train_error) printf(" %.5f", err);
      WriteLog(current_iter_, time_diff.count(), train_error);
      printf(" Weight length: " );
      for (Edge* e : edges_) {
        if (e->HasNoParameters() || e->IsTied()) continue;
        printf(" %.5f", e->GetRMSWeight());
      }
      train_error.clear();
      start_t = end_t;
      newline = true;
    }

    if (polyak_after > 0 && (i+1) % polyak_after == 0 && ((i+1) % validate_after) >= start_polyak_queue) {
      InsertPolyak();
    }

    if (val_dataset_ != NULL && validate_after > 0 && (i+1) % validate_after == 0) {
      if (polyak_after > 0) LoadPolyakWeights();
      Validate(this_val_error);
      if (polyak_after > 0) LoadCurrentWeights();

      val_error.push_back(this_val_error[0]);
      cout << " Val Acc :";
      for (const float& val: this_val_error) printf(" %.5f", val);
      WriteValLog(current_iter_, this_val_error);

      // Should we reduce the learning rate ?
      if (learning_rate_reduce_factor < 1.0) {
        bool reduce_learning_rate = CheckReduceLearningRate(val_error);
        if (reduce_learning_rate && lr_reduce_counter_ < lr_max_reduce
            && dont_reduce_lr-- < 0) {
          dont_reduce_lr = model_->reduce_lr_num_steps();
          cout << "Learning rate reduced " << ++lr_reduce_counter_ << " time(s)."
               << endl;
          ReduceLearningRate(learning_rate_reduce_factor);
        }
      }
      newline = true;
    }
    if (newline) cout << endl;
    if ((i+1) % save_after == 0) Save();
  }
  Save();
  cout << "End of training." << endl;
}
