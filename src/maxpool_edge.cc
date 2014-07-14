#include "maxpool_edge.h"
#include "cudamat_conv.cuh"

MaxPoolEdge::MaxPoolEdge(const config::Edge& edge_config) :
  Edge(edge_config),
  kernel_size_(edge_config.kernel_size()),
  stride_(edge_config.stride()),
  padding_(edge_config.padding()){}

void MaxPoolEdge::SetTiedTo(Edge* e) {
  Edge::SetTiedTo(e);
  MaxPoolEdge* ee = dynamic_cast<MaxPoolEdge*> (e);
  kernel_size_ = ee->GetKernelSize();
  stride_ = ee->GetStride();
  padding_ = ee->GetPadding();
}


void MaxPoolEdge::SetImageSize(int image_size) {
  Edge::SetImageSize(image_size);
  num_modules_ = (image_size + 2 * padding_ - kernel_size_) / stride_ + 1;
}

void MaxPoolEdge::FOV(int* size, int* sep, int* pad1, int* pad2) const {
  *size = kernel_size_ + stride_ * ((*size) - 1);
  *sep = (*sep) * stride_;
  *pad1 = (*pad1) * stride_ + padding_;
  int k = (image_size_ + 2*padding_ - kernel_size_) / stride_;
  int effective_right_pad = k * stride_ - (image_size_ + padding_ - kernel_size_);
  *pad2 = (*pad2) * stride_ + effective_right_pad;
}

void MaxPoolEdge::AllocateMemory(bool fprop_only) {
  
  cout << name_ << " ";
  printf("Kernel: %d-%d-%d to %d ", kernel_size_, kernel_size_,
         num_input_channels_, num_output_channels_);
  printf("Layer: %d-%d-%d (%d) ", image_size_, image_size_, num_input_channels_,
         image_size_ * image_size_ * num_input_channels_);
  cout << " Maxpool." << endl;
 
  // This edge has no memory requirements.
}

void MaxPoolEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  cudamat* input_mat = input.GetMat();
  cudamat* output_mat = output.GetMat();
  MaxPool(input_mat, output_mat, num_input_channels_, kernel_size_, -padding_,
          stride_, num_modules_);
}

void MaxPoolEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                              Matrix& output, Matrix& deriv_input, bool overwrite) {
  cudamat* input_mat = input.GetMat();
  cudamat* output_mat = output.GetMat();
  
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();

  // Deriv w.r.t input of this edge (which is to be computed).
  cudamat* deriv_input_mat = deriv_input.GetMat();
  
  MaxPoolUndo(input_mat, deriv_output_mat, output_mat, deriv_input_mat,
              kernel_size_, -padding_, stride_, num_modules_);
}

