#include "local_edge.h"
#include "cudamat_conv.cuh"
#include <iostream>

LocalEdge::LocalEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config),
  kernel_size_(edge_config.kernel_size()),
  stride_(edge_config.stride()),
  padding_(edge_config.padding()) {}

void LocalEdge::SetTiedTo(Edge* e) {
  EdgeWithWeight::SetTiedTo(e);
  /*
  LocalEdge* conv_e = dynamic_cast<LocalEdge*>(e);
  if (conv_e == NULL) {
    cerr << "Error: Local edge should be tied to other conv edges only." << endl;
    exit(1);
  }
  kernel_size_ = conv_e->GetKernelSize();
  stride_ = conv_e->GetStride();
  padding_ = conv_e->GetPadding();
  shared_bias_ = conv_e->IsBiasShared();
  */
}

void LocalEdge::DisplayWeights() {
  if (img_display_ != NULL) {
    weights_.CopyToHost();
    img_display_->DisplayWeights(weights_.GetHostData(), kernel_size_, num_output_channels_, 250, false);
  }
}


void LocalEdge::AllocateMemory(int image_size) {
  Edge::AllocateMemory(image_size);
  
  num_modules_ = (image_size + 2 * padding_ - kernel_size_) / stride_ + 1;

  cout << name_ << " ";
  printf("Kernel: %d-%d-%d to %d ", kernel_size_, kernel_size_,
         num_input_channels_, num_output_channels_);
  printf("Layer: %d-%d-%d (%d) ", image_size, image_size, num_input_channels_,
         image_size * image_size * num_input_channels_);
  
  int weight_input_size = kernel_size_ * kernel_size_ * num_input_channels_
                          * num_modules_ * num_modules_;

  // Weights for this convolution.
  weights_.AllocateGPUMemory(num_output_channels_, weight_input_size);

  // Matrix for storing the current gradient.
  grad_weights_.AllocateGPUMemory(weights_.GetRows(), weights_.GetCols());

  // Matrix for storing gradient history (used for implementing momentum or
  // other accelerated gradient descent schemes).
  weight_optimizer_->AllocateMemory(weights_.GetRows(), weights_.GetCols());

  cout << " Allocated weight " << weights_.GetRows() << " " << weights_.GetCols();
 
  const int bias_locs = num_modules_ * num_modules_;
  bias_.AllocateGPUMemory(1, num_output_channels_ * bias_locs);
  grad_bias_.AllocateGPUMemory(bias_.GetRows(), bias_.GetCols());
  bias_optimizer_->AllocateMemory(bias_.GetRows(), bias_.GetCols());
  
  cout << " Locally Connected" << endl;
  if (num_input_channels_ == 3) {
    int num_filters = num_output_channels_;
    int num_filters_w = int(sqrt(num_filters));
    int num_filters_h = num_filters / num_filters_w +  (((num_filters % num_filters_w) > 0) ? 1 : 0);
    int width = 250;
    int height = (width * num_filters_h) / num_filters_w;
    img_display_ = new ImageDisplayer(width, height, 3, false, "weights");
  }
}

void LocalEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  ComputeStart(input);
  cudamat *input_mat = input.GetMat(),
          *output_mat = output.GetMat(),
          *w_mat = weights_.GetMat();
  int scale_targets = overwrite ? 0 : 1;
  localUp(input_mat, w_mat, output_mat, num_modules_, -padding_, stride_,
          num_input_channels_, 1, scale_targets);
  if (!has_no_bias_) {
    cudamat* b_mat = bias_.GetMat();
    add_row_vec(output_mat, b_mat, output_mat);
  }
  ComputeEnd(output);
}

void LocalEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite) {
  ComputeStart(deriv_output);
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();

  // Deriv w.r.t input of this edge (which is to be computed).
  cudamat* deriv_input_mat = deriv_input.GetMat();
  
  cudamat* w_mat = weights_.GetMat();
  int scale_targets = overwrite ? 0 : 1;
  localDown(deriv_output_mat, w_mat, deriv_input_mat, image_size_, -padding_,
            stride_, num_input_channels_, 1, scale_targets);
  ComputeEnd(deriv_input);
}

void LocalEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  ComputeStart(deriv_output);
  // Input to this edge.
  cudamat* input_mat = input.GetMat();
  
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();

  cudamat* dw_mat = grad_weights_.GetMat();
  cudamat* db_mat = grad_bias_.GetMat();

  const int batch_size = input.GetRows();

  int scale_targets = 0;

  localOutp(input_mat, deriv_output_mat, dw_mat, num_modules_, kernel_size_,
            -padding_, stride_, num_input_channels_, 1,
            scale_targets, scale_gradients_ / batch_size);
  Matrix ones;
  Matrix::GetOnes(1, batch_size, ones);
  dot(ones.GetMat(), deriv_output_mat, db_mat, scale_targets, 1.0 / batch_size);
}
