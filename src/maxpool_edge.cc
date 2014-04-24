#include "maxpool_edge.h"
#include "cudamat_conv.cuh"

MaxPoolEdge::MaxPoolEdge(const config::Edge& edge_config) :
  Edge(edge_config),
  kernel_size_(edge_config.kernel_size()),
  stride_(edge_config.stride()),
  padding_(edge_config.padding()){}

void MaxPoolEdge::SetImageSize(int image_size) {
  Edge::SetImageSize(image_size);
  num_modules_ = (image_size + 2 * padding_ - kernel_size_) / stride_ + 1;
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
  ComputeStart(input);
  cudamat* input_mat = input.GetMat();
  cudamat* output_mat = output.GetMat();
  MaxPool(input_mat, output_mat, num_input_channels_, kernel_size_, -padding_,
          stride_, num_modules_);
  ComputeEnd(output);
}

void MaxPoolEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                              Matrix& output, Matrix& deriv_input, bool overwrite) {
  ComputeStart(deriv_output);
  cudamat* input_mat = input.GetMat();
  cudamat* output_mat = output.GetMat();
  
  // Deriv w.r.t output of this edge.
  cudamat* deriv_output_mat = deriv_output.GetMat();

  // Deriv w.r.t input of this edge (which is to be computed).
  cudamat* deriv_input_mat = deriv_input.GetMat();
  
  MaxPoolUndo(input_mat, deriv_output_mat, output_mat, deriv_input_mat,
              kernel_size_, -padding_, stride_, num_modules_);
  ComputeEnd(deriv_input);
}

