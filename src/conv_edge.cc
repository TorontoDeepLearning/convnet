#include "conv_edge.h"
#include <iostream>

ConvEdge::ConvEdge(const config::Edge& edge_config) :
  EdgeWithWeight(edge_config),
  conv_desc_(Edge::GetConvDesc(edge_config)),
#ifdef USE_GEMM
  partial_sum_y_(0),
  partial_sum_x_(0),
#else
  partial_sum_y_(edge_config.partial_sum()),
  partial_sum_x_(edge_config.partial_sum()),
#endif
  shared_bias_(edge_config.shared_bias()) {}

void ConvEdge::SetTiedTo(Edge* e) {
  EdgeWithWeight::SetTiedTo(e);
  ConvEdge* ee = dynamic_cast<ConvEdge*>(e);
  conv_desc_ = ee->GetConvDesc();
  if (partial_sum_x_ == 0) partial_sum_x_ = ee->GetPartialSumX();
  if (partial_sum_y_ == 0) partial_sum_y_ = ee->GetPartialSumY();
  shared_bias_ = ee->GetSharedBias();
}

void ConvEdge::SetImageSize(int image_size_y, int image_size_x, int image_size_t) {
  Edge::SetImageSize(image_size_y, image_size_x, image_size_t);
  conv_desc_.num_input_channels = num_input_channels_;
  conv_desc_.num_output_channels = num_output_channels_;
  Edge::GetNumModules(conv_desc_, image_size_y, image_size_x, image_size_t,
                      num_modules_y_, num_modules_x_, num_modules_t_);
  if (partial_sum_y_ == 0) partial_sum_y_ = num_modules_y_;
  if (partial_sum_x_ == 0) partial_sum_x_ = num_modules_x_;
}

string ConvEdge::GetDescription() {
  stringstream ss;
  ss << name_
     << " Convolutional Kernel: " << Edge::GetDescription(conv_desc_)
     << " Layer: " << image_size_y_ << "-" << image_size_x_ << " : "
     <<  num_modules_y_ << "-" << num_modules_x_;
  return ss.str();
}

void ConvEdge::FOV(int* size, int* sep, int* pad1, int* pad2) const {
  /*
  *size = kernel_size_ + stride_ * ((*size) - 1);
  *sep = (*sep) * stride_;
  *pad1 = (*pad1) * stride_ + padding_;
  int k = (image_size_x_ + 2*padding_ - kernel_size_) / stride_;
  int effective_right_pad = k * stride_ - (image_size_x_ + padding_ - kernel_size_);
  *pad2 = (*pad2) * stride_ + effective_right_pad;
  */
}

void ConvEdge::DisplayWeights() {
  if (img_display_ != NULL && display_) {
    weights_.CopyToHost();
    img_display_->DisplayWeights(
        weights_.GetHostData(), conv_desc_.kernel_size_y,
        conv_desc_.num_output_channels, 250, false);
  }
}

size_t ConvEdge::GetParameterMemoryRequirement() {
  if (is_tied_) return 0;
  int input_size = conv_desc_.kernel_size_y * conv_desc_.kernel_size_x *
                   conv_desc_.kernel_size_t * conv_desc_.num_input_channels;
  int bias_locs = shared_bias_ ? 1: (num_modules_y_ * num_modules_x_ * num_modules_t_);
  return conv_desc_.num_output_channels *  (input_size + (has_no_bias_ ? 0 : bias_locs));
}

void ConvEdge::SetMemory(Matrix& p) {
  if (is_tied_) return;
  Edge::SetMemory(p);

  int input_size = conv_desc_.kernel_size_y * conv_desc_.kernel_size_x *
                   conv_desc_.kernel_size_t * conv_desc_.num_input_channels;
  int bias_locs = shared_bias_ ? 1: (num_modules_y_ * num_modules_x_ * num_modules_t_);
  
  // Weights for this convolution.
  p.Reshape(conv_desc_.num_output_channels, -1);
  p.GetSlice(weights_, 0, input_size);
  weights_.SetShape4D(conv_desc_.num_output_channels, conv_desc_.kernel_size_x,
                      conv_desc_.kernel_size_y, conv_desc_.num_input_channels * conv_desc_.kernel_size_t);
  if (!has_no_bias_) {
    p.GetSlice(bias_, input_size, input_size + bias_locs);
    bias_.Reshape(1, -1);
  }

  if (conv_desc_.num_input_channels == 3) {
    int num_filters = conv_desc_.num_output_channels;
    int num_filters_w = int(sqrt(num_filters));
    int num_filters_h = num_filters / num_filters_w + (((num_filters % num_filters_w) > 0) ? 1 : 0);
    int width = 250;
    int height = (width * num_filters_h) / num_filters_w;
    img_display_ = new ImageDisplayer(width, height, 3, false, "weights");
  }
}

void ConvEdge::SetGradMemory(Matrix& p) {
  int input_size = conv_desc_.kernel_size_y * conv_desc_.kernel_size_x *
                   conv_desc_.kernel_size_t * conv_desc_.num_input_channels;
  int num_locs = num_modules_y_ * num_modules_x_ * num_modules_t_;
  int bias_locs = shared_bias_ ? 1 : num_locs;

  if (!is_tied_) {
    p.Reshape(conv_desc_.num_output_channels, -1);
    p.GetSlice(grad_weights_, 0, input_size);
    grad_weights_.SetShape4D_like(weights_);
    weight_optimizer_->AllocateMemory(conv_desc_.num_output_channels, input_size);
  }

  int num_partial_sum_locs = DIVUP(num_modules_y_, partial_sum_y_) * DIVUP(num_modules_x_, partial_sum_x_);
  if (num_partial_sum_locs > 1) {
    Matrix::RegisterTempMemory(conv_desc_.num_output_channels * input_size * num_partial_sum_locs,
                               "partial sums " + GetName());
    Matrix::RegisterOnes(num_partial_sum_locs);
  }
 
  if (!has_no_bias_ && !is_tied_) {
    p.GetSlice(grad_bias_, input_size, input_size + bias_locs);
    grad_bias_.Reshape(1, -1);
    bias_optimizer_->AllocateMemory(1, conv_desc_.num_output_channels * bias_locs);
    if (shared_bias_) {
      Matrix::RegisterTempMemory(conv_desc_.num_output_channels * num_locs, "shared bias");
    }
  }
}

void ConvEdge::ComputeUp(Matrix& input, Matrix& output, bool overwrite) {
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  float scale_targets = overwrite ? 0 : 1;
  if (image_size_t_ == 1) {
    Matrix::ConvUp(input, w, output, conv_desc_, scale_targets);
    if (!has_no_bias_) {
      Matrix& b = is_tied_? tied_edge_->GetBias() : bias_;
      if (shared_bias_) {
        output.Reshape(-1, conv_desc_.num_output_channels);
        output.AddRowVec(b);
        output.Reshape(-1, conv_desc_.num_output_channels * num_modules_y_ * num_modules_x_ * num_modules_t_);
      } else {
        output.AddRowVec(b);
      }
    }
  } else {  // 3D convolution.
    Matrix::Conv3DUp(input, w, output, conv_desc_, scale_targets);
    if (!has_no_bias_) {
      Matrix& b = is_tied_? tied_edge_->GetBias() : bias_;
      if (shared_bias_) {
        output.Reshape(-1, conv_desc_.num_output_channels * num_modules_t_);
        for (int m = 0; m < num_modules_t_; m++) {
          Matrix output_slice;
          output.GetSlice(output_slice, m * conv_desc_.num_output_channels, (m+1)*conv_desc_.num_output_channels);
          output_slice.AddRowVec(b);
        }
        output.Reshape(-1, conv_desc_.num_output_channels * num_modules_y_ * num_modules_x_ * num_modules_t_);
      } else {
        output.AddRowVec(b);
      }
    }
  }
}

void ConvEdge::ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite) {
  Matrix& w = is_tied_? tied_edge_->GetWeight() : weights_;
  float scale_targets = overwrite ? 0 : 1;
  if (image_size_t_ == 1) {
    Matrix::ConvDown(deriv_output, w, deriv_input, conv_desc_, scale_targets);
  } else {
    Matrix::Conv3DDown(deriv_output, w, deriv_input, conv_desc_, scale_targets);
  }
}

void ConvEdge::ComputeOuter(Matrix& input, Matrix& deriv_output) {
  Matrix& dw = is_tied_ ? tied_edge_->GetGradWeight() : grad_weights_;
  const int batch_size = input.GetRows();
  int scale_targets = GetNumGradsReceived() > 0 ? 1 : 0;

  int input_size = conv_desc_.kernel_size_y * conv_desc_.kernel_size_x *
                   conv_desc_.kernel_size_t * conv_desc_.num_input_channels;
  if (image_size_t_ == 1) {
    int partial_sum_locs = DIVUP(num_modules_y_, partial_sum_y_) * DIVUP(num_modules_x_, partial_sum_x_);
    if (partial_sum_locs > 1) {
      Matrix dw_temp;
      Matrix::GetTemp(conv_desc_.num_output_channels, input_size * partial_sum_locs, dw_temp);
      dw_temp.SetShape4D(conv_desc_.num_output_channels, conv_desc_.kernel_size_x,
                        conv_desc_.kernel_size_y, conv_desc_.num_input_channels
                        * conv_desc_.kernel_size_t * partial_sum_locs);
      Matrix::ConvOutp(input, deriv_output, dw_temp, conv_desc_, partial_sum_y_,
                       partial_sum_x_, 0, 1);

      dw_temp.Reshape(conv_desc_.num_output_channels * input_size, partial_sum_locs);
      dw.Reshape(-1, 1);
      dw_temp.SumCols(dw, scale_targets, scale_gradients_ / batch_size);
      dw.Reshape(conv_desc_.num_output_channels, input_size);
    } else {
      Matrix::ConvOutp(input, deriv_output, dw, conv_desc_, partial_sum_y_,
                       partial_sum_x_, scale_targets,
                       scale_gradients_ / batch_size);
    }
    if (!has_no_bias_) {
      Matrix& db = is_tied_ ? tied_edge_->GetGradBias() : grad_bias_;
      if (shared_bias_) {
        // 2 step addition is SIGNFICANTLY faster (Why ?)
        Matrix db_temp;
        Matrix::GetTemp(1, deriv_output.GetCols(), db_temp);
        deriv_output.SumRows(db_temp, 0, 1);
        db_temp.Reshape(-1, conv_desc_.num_output_channels);
        db_temp.SumRows(db, scale_targets, scale_gradients_ / batch_size);
      } else {
        deriv_output.SumRows(db, scale_targets, scale_gradients_ / batch_size);
      }
    }
  } else {  // 3D convolutions.
    Matrix::Conv3DOutp(input, deriv_output, dw, conv_desc_, scale_targets,
                       scale_gradients_ / batch_size);
    if (!has_no_bias_) {
      Matrix& db = is_tied_ ? tied_edge_->GetGradBias() : grad_bias_;
      if (shared_bias_) {
        Matrix db_temp;
        Matrix::GetTemp(1, deriv_output.GetCols(), db_temp);
        deriv_output.SumRows(db_temp, 0, 1);
        db_temp.Reshape(-1, conv_desc_.num_output_channels * num_modules_t_);
        db.Mult(scale_targets);
        for (int m = 0; m < num_modules_t_; m++) {
          Matrix db_temp_slice;
          db_temp.GetSlice(db_temp_slice, m * conv_desc_.num_output_channels, (m+1)*conv_desc_.num_output_channels);
          db_temp_slice.SumRows(db, 1.0, scale_gradients_ / batch_size);
        }
      } else {
        deriv_output.SumRows(db, scale_targets, scale_gradients_ / batch_size);
      }
    }
  }
  IncrementNumGradsReceived();
}
