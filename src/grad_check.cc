#include "grad_check.h"
#include "edge_with_weight.h" 
#include <iostream>
using namespace std;

GradChecker::GradChecker(const string& model_file) : ConvNet(model_file) {}

void GradChecker::GetLoss(vector<float>& error) {
  Fprop(false);
  error.clear();
  for (Layer* l: output_layers_) {
    error.push_back(l->GetLoss2());
  }
}

void GradChecker::Run(Matrix& w, Matrix& dLbydw_analytical, float epsilon, const vector<float>& analytical_error) {
  w.CopyToHost();
  dLbydw_analytical.CopyToHost();
  float* w_dataptr = w.GetHostData();
  float* dLbydw_analytical_ptr = dLbydw_analytical.GetHostData();
  int num_params = w.GetNumEls();
  float analytical_grad;
  vector<float> diffs;
  int index = 0;
  cout << "Batch size " << batch_size_ << endl;
  for (; index < num_params && index < 100; index++) {
    w_dataptr[index] += epsilon;
    if (index > 0) w_dataptr[index-1] -= epsilon;
    w.CopyToDevice();
    analytical_grad = dLbydw_analytical_ptr[index];

    // Get loss with the perturbed weight.
    vector<float> error;
    GetLoss(error);

    float numerical_grad = (error[0] - analytical_error[0]) / (batch_size_ * epsilon);
    // Compute diff.
    float this_diff = fabs(numerical_grad - analytical_grad) / fabs((analytical_grad + numerical_grad) / 2);
    if (numerical_grad != analytical_grad && this_diff != this_diff) {
      cout << "Nan! Numerical " << numerical_grad << " Analytical " << analytical_grad << endl;
    }
    printf("Weight %d Analytical %.7f Numerical %.7f diff %.7f\n", index+1, analytical_grad, numerical_grad, this_diff);
    diffs.push_back(this_diff);
  }
  w_dataptr[index - 1] -= epsilon;
  w.CopyToDevice();
  double mean = 0;
  for (const float& val : diffs) mean += val;
  mean /= diffs.size();
  printf("\nAvg diff %.7f\n", mean);
}

void GradChecker::Run() {
  float epsilon = 5 * 1e-3;
  train_dataset_->GetBatch(data_layers_);

  // Get analytical gradient.
  vector<float> error;
  Fprop(false);
  ComputeDeriv();
  GetLoss(error);
  Bprop(false);  // false means do not update weights.
  // Analytical gradients are now stored in grad_weights of each edge.


  // Pick an edge.
  //for (Edge* ed : edges_) {
  Edge* ed = edges_[0];
    EdgeWithWeight* e = dynamic_cast<EdgeWithWeight*>(ed);
    if (e == NULL) {
      cout << "Edge does not have weights." << endl;
      //continue;
    }
    cout << "Edge " << ed->GetName() << endl;
    cout << "Weight" << endl;
    Run(e->GetWeight(), e->GetGradWeight(), epsilon, error);
    cout << "Bias" << endl;
    Run(e->GetBias(), e->GetGradBias(), epsilon, error);
  //}
}

