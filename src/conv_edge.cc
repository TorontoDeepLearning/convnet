#include "conv_edge.h"
#include "cudamat_conv.cuh"
#include <iostream>

ConvEdge::ConvEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config),
  kernel_size_(edge_config.kernel_size()),
  stride_(edge_config.stride()),
  padding_(edge_config.padding()),
  partial_sum_(edge_config.partial_sum()),
  shared_bias_(edge_config.shared_bias()),
  num_modules_(0) {}

void ConvEdge::SetTiedTo(Edge* e) {
  EdgeWithWeight::SetTiedTo(e);
}

void ConvEdge::DisplayWeights() {
  if (img_display_ != NULL) {
    weights_.CopyToHost();
    img_display_->DisplayWeights(weights_.GetHostData(), kernel_size_, num_output_channels_, 250, false);
  }
}

void ConvEdge::AllocateMemory(int image_size) {
  Edge::AllocateMemory(image_size);
  
  num_modules_ = (image_size + 2 * padding_ - kernel_size_) / stride_ + 1;

  cout << name_ << " ";
  printf("Kernel: %d-%d-%d to %d ", kernel_size_, kernel_size_,
         num_input_channels_, num_output_channels_);
  printf("Layer: %d-%d-%d (%d) ", image_size, image_size, num_input_channels_,
         image_size * image_size * num_input_channels_);
  
  int weight_input_size = kernel_size_ * kernel_size_ * num_input_channels_;

  // Weights for this convolution.
  weights_.AllocateGPUMemory(num_output_channels_, weight_input_size);

  // Matrix for storing the current gradient.
  grad_weights_.AllocateGPUMemory(weights_.GetRows(), weights_.GetCols());

  // Matrix for storing gradient history (used for implementing momentum or
  // other accelerated gradient descent schemes).
  weight_optimizer_->AllocateMemory(weights_.GetRows(), weights_.GetCols());

  cout << " Allocated weight " << weights_.GetRows() << " " << weights_.GetCols();
  if (partial_sum_ > 0) {
    int partial_sums = (num_modules_ * num_modules_) / partial_sum_;
    cout << "Partial sum " << partial_sum_ << " must divide number of modules "
         << num_modules_ * num_modules_ << "." << endl;
    if (partial_sum_ * partial_sums != num_modules_ * num_modules_) {
      cerr << "Error: Partial sum " << partial_sum_
           << " must divide number of modules " << num_modules_ * num_modules_
           << "." << endl;
      exit(1);
    }
    Matrix::RegisterTempMemory(num_output_channels_ * weight_input_size * partial_sums, "partial sums " + GetName());
    Matrix::RegisterOnes(partial_sums);
  }
  cout << " Convolutional" << endl;
  //if (is_sparse_conv_) AllocateGPUMatrix(&w_slice_, 1, weight_input_size);
  
  int bias_locs = shared_bias_ ? 1: (num_modules_ * num_modules_);
  bias_.AllocateGPUMemory(1, num_output_channels_ * bias_locs);
  grad_bias_.AllocateGPUMemory(bias_.GetRows(), bias_.GetCols());
  bias_optimizer_->AllocateMemory(bias_.GetRows(), bias_.GetCols());
  if (shared_bias_) {
    Matrix::RegisterTempMemory(num_output_channels_ * num_modules_ * num_modules_, "shared bias");
  }
  if (num_input_channels_ == 3) {
    int num_filters = num_output_channels_;
    int num_filters_w = int(sqrt(num_filters));
    int num_filters_h = num_filters / num_filters_w +  (((num_filters % num_filters_w) > 0) ? 1 : 0);
    int width = 250;
    int height = (width * num_filters_h) / num_filters_w;
    img_display_ = new ImageDisplayer(width, height, 3, false, "weights");
  }
}

void ConvEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  ComputeStart(input);
  cudamat *input_mat = input.GetMat(),
          *output_mat = output.GetMat(),
          *w_mat = weights_.GetMat();
  int scale_targets = overwrite ? 0 : 1;
  convUp(input_mat, w_mat, output_mat, num_modules_, -padding_, stride_,
         num_input_channels_, 1, scale_targets);

  if (!has_no_bias_) {
    cudamat* b_mat = bias_.GetMat();
    if (shared_bias_) {
      reshape(output_mat, -1, num_output_channels_);
      add_row_vec(output_mat, b_mat, output_mat);
      reshape(output_mat, -1, num_output_channels_ * num_modules_ * num_modules_);
    } else {
      add_row_vec(output_mat, b_mat, output_mat);
    }
  }
  ComputeEnd(output);
}

void ConvEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite) {
  ComputeStart(deriv_output);
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();

  // Deriv w.r.t input of this edge (which is to be computed).
  cudamat* deriv_input_mat = deriv_input.GetMat();
  
  cudamat* w_mat = weights_.GetMat();

  //cout << "Target rows " << deriv_input.GetRows() << " cols " << deriv_input.GetCols() << endl;
  //cout << "NumImgColors " << num_input_channels_ << endl;
  // cout << "Image size " << image_size_ << endl;

  int scale_targets = overwrite ? 0 : 1;
  convDown(deriv_output_mat, w_mat, deriv_input_mat, image_size_, -padding_,
           stride_, num_input_channels_, 1, scale_targets);
  ComputeEnd(deriv_input);
}

void ConvEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  ComputeStart(deriv_output);
  // Input to this edge.
  cudamat* input_mat = input.GetMat();
  
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();

  cudamat* dw_mat = grad_weights_.GetMat();
  cudamat* db_mat = grad_bias_.GetMat();
  const int batch_size = input.GetRows();

  int scale_targets = 0;

  if (partial_sum_ > 0) {
    Matrix dw_temp;

    int filter_input_size = num_input_channels_ * kernel_size_ * kernel_size_;
    int partial_sums = (num_modules_ * num_modules_) / partial_sum_;
    Matrix::GetTemp(num_output_channels_, filter_input_size * partial_sums, dw_temp);
    cudamat* dw_temp_mat = dw_temp.GetMat();
    convOutp(input_mat, deriv_output_mat, dw_temp_mat, num_modules_, kernel_size_,
             -padding_, stride_, num_input_channels_, 1, partial_sum_, 0, 1);
    reshape(dw_temp_mat, num_output_channels_ * filter_input_size, partial_sums);
    reshape(dw_mat, -1, 1);
    Matrix ones;
    Matrix::GetOnes(partial_sums, 1, ones);
    dot(dw_temp_mat, ones.GetMat(), dw_mat, scale_targets, scale_gradients_ / batch_size);
    reshape(dw_mat, num_output_channels_, filter_input_size);
  } else {
    convOutp(input_mat, deriv_output_mat, dw_mat, num_modules_, kernel_size_,
             -padding_, stride_, num_input_channels_, 1, partial_sum_,
             scale_targets, scale_gradients_ / batch_size);
  }

  if (shared_bias_) {
    /*
    reshape(deriv_output_mat, -1, num_output_channels_);
    Matrix ones;
    Matrix::GetOnes(1, batch_size * num_modules_ * num_modules_, ones);
    dot(ones.GetMat(), deriv_output_mat, db_mat, scale_targets_, 1.0 / batch_size);
    reshape(deriv_output_mat, batch_size, -1);
    */
    // 2 step addition is SIGNFICANTLY faster (Why ?)
    Matrix ones, db_temp;
    Matrix::GetOnes(1, batch_size, ones);
    Matrix::GetTemp(1, deriv_output.GetCols(), db_temp);
    cudamat* db_temp_mat = db_temp.GetMat();
    dot(ones.GetMat(), deriv_output_mat, db_temp_mat, 0, 1);
    reshape(db_temp_mat, -1, num_output_channels_);
    Matrix::GetOnes(1, num_modules_ * num_modules_, ones);
    dot(ones.GetMat(), db_temp_mat, db_mat, scale_targets, 1.0 / batch_size);
  } else {
    Matrix ones;
    Matrix::GetOnes(1, batch_size, ones);
    dot(ones.GetMat(), deriv_output_mat, db_mat, scale_targets, 1.0 / batch_size);
  }
}
