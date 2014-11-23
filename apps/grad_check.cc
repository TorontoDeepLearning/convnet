#include "grad_check.h"
#include "edge_with_weight.h" 
#include <iostream>
#include <iomanip>
using namespace std;

GradChecker::GradChecker(const string& model_file) : ConvNet(model_file) {
}

void GradChecker::GetLoss(vector<float>& error) {
  for (Layer* l : layers_) l->ResetAddOrOverwrite();
  Fprop(false);
  error.clear();
  for (Layer* l: output_layers_) {
    float err = l->GetLoss();
    error.push_back(err);
  }
}

void GradChecker::ComputeNumericGrad(Matrix& w, float epsilon, vector<float>& numerical_grad, int max_params) {
  int num_params = w.GetNumEls();
  if (max_params > 0 && num_params > max_params) num_params = max_params;
  numerical_grad.resize(num_params);
  vector<float> error1, error2;
  float val;
  for (int i = 0; i < num_params; i++) {
    val = w.ReadValue(i);
    w.WriteValue(i, val + epsilon);
    GetLoss(error1);
    w.WriteValue(i, val - epsilon);
    GetLoss(error2);
    numerical_grad[i] = (error1[0] - error2[0]) / (batch_size_ * 2 * epsilon);
    w.WriteValue(i, val);
  }
}

bool GradChecker::GradCheck(Matrix& w, vector<float>& eps_values, int num_params, const float* analytical_g, const string& output_dset, hid_t& output_file) {
  float* numerical_grad = new float[num_params * eps_values.size()];
  vector<float> this_numerical_grad;
  float scaled_diff, diff_sum = 0, test_pass_epsilon = 0;
  int non_zero_grad = 0;
  bool test_pass = false;
  for (int j = 0; j < eps_values.size() && !test_pass; j++) {
    cout << "Epsilon : " << eps_values[j] << endl;
    cout << "Analytical\tNumerical\tDiff\t\tScaled diff" << endl;
    ComputeNumericGrad(w, eps_values[j], this_numerical_grad, num_params);
    for (int k = 0; k < num_params; k++) {
      float diff = analytical_g[k] - this_numerical_grad[k];
      float scale = (analytical_g[k] + this_numerical_grad[k]) / 2;
      if (scale == 0 && diff == 0) {
        scaled_diff = 0;
      } else {
        scaled_diff = abs(diff / scale);
        diff_sum += scaled_diff;
        non_zero_grad++;
      }
      cout << analytical_g[k] << "\t" << this_numerical_grad[k] << "\t" << diff << "\t" << scaled_diff << endl;
    }
    diff_sum /= non_zero_grad;
    cout << "Mean of scaled diffs (non-zero grad) " << diff_sum << endl;
    if (diff_sum < 0.01) {
      test_pass = true;
      test_pass_epsilon = eps_values[j];
    }
    for (int k = 0; k < num_params; k++) numerical_grad[j * num_params + k] = this_numerical_grad[k];
  }
  if (test_pass) {
    cout << " PASSED for eps = " << test_pass_epsilon << endl;
  } else {
    cout << " FAILED" << endl;
  }
  WriteHDF5CPU(output_file, numerical_grad, num_params, eps_values.size(), output_dset);
  delete[] numerical_grad;
  return test_pass;
}

void GradChecker::Run(const string& output_file) {
  hid_t file = H5Fcreate(output_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  
  // Get analytical gradient.
  for (Layer* l : layers_) l->ResetAddOrOverwrite();
  for (Layer* l : data_layers_) {
    if (l->IsInput()) {
      l->GetState().FillWithRandn();
    } else {
      l->GetData().FillWithRand();
    }
  }
  Fprop(false);
  ComputeDeriv();
  Bprop();
  // Analytical gradients are now stored in grad_weights of each edge.

  vector<float> this_numerical_grad;
  cout << fixed << setprecision(9);
  map<string, bool> w_pass, b_pass;
  for (Edge* ed : edges_) {
    if (!ed->GradCheck()) continue;
    EdgeWithWeight* e = dynamic_cast<EdgeWithWeight*>(ed);
    if (e == NULL) continue;  // Some edges don't have weights, skip them.

    const string& name = e->GetName();
    int num_params_w = ed->GradCheckNumParams();
    int num_params_b = ed->GradCheckNumParams();
    vector<float> eps_values;
    ed->GradCheckEpsilon(eps_values);
    WriteHDF5CPU(file, eps_values, eps_values.size(), 1, name + "_epsilon_values");
    
    Matrix& analytical_grad_w = e->GetGradWeight();
    Matrix& analytical_grad_b = e->GetGradBias();
    analytical_grad_w.CopyToHost();
    analytical_grad_b.CopyToHost();
    float* analytical_g_w = analytical_grad_w.GetHostData();
    float* analytical_g_b = analytical_grad_b.GetHostData();
    int max_params_w = analytical_grad_w.GetNumEls();
    int max_params_b = analytical_grad_b.GetNumEls();
    if (num_params_w > max_params_w) num_params_w = max_params_w;
    if (num_params_b > max_params_b) num_params_b = max_params_b;
    WriteHDF5CPU(file, analytical_g_w, num_params_w, 1, name + "_weights_analytical");
    WriteHDF5CPU(file, analytical_g_b, num_params_b, 1, name + "_bias_analytical");

    cout << "Running grad check for " << name << " weights " << " Num params " << num_params_w << endl;
    w_pass[name] = GradCheck(e->GetWeight(), eps_values, num_params_w, analytical_g_w, name + "_weights_numerical", file);
    cout << "Running grad check for " << name << " bias " << " Num params " << num_params_w << endl;
    b_pass[name] = GradCheck(e->GetBias(), eps_values, num_params_b, analytical_g_b, name + "_bias_numerical", file);
  }
  H5Fclose(file);
  cout << "SUMMARY" << endl;
  cout << "WEIGHTS" << endl;
  for (auto& kv : w_pass) {
    string result = kv.second ? "PASSED" : "FAILED";
    cout << kv.first << " " << result << endl;
  }
  cout << "BIAS" << endl;
  for (auto& kv : b_pass) {
    string result = kv.second ? "PASSED" : "FAILED";
    cout << kv.first << " " << result << endl;
  }
}
