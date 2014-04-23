class BoundingBoxTargets {
 public:
  BoundingBoxTargets() :add_mean_image_(false), YUV_(false) {};
  void Configure(int num_objects, int output_image_size, int input_image_size, bool use_other_classes_as_negatives);
  void AllocateMemory(int batch_size);
  void ComputeDeriv(Matrix& state, const vector<vector<box>>& boxes, Matrix& deriv, int label);
  void ComputeDeriv2(Matrix& state, const vector<vector<box>>& boxes, Matrix& deriv);
  void Display(Layer& input, Layer& predictions, const vector<vector<box>>& boxes);
  void Display(Layer& input, Layer& predictions, const vector<vector<box>>& boxes, int num_images, int obj_id);
  void SetMeanImage(const string& mean_image_file);

 private:
  int num_objects_, output_image_size_, input_image_size_, num_channels_;
  bool add_mean_image_;
  CImgDisplay *disp_;
  CImg<float> mean_image_;  // used to invert the input transform so that inputs can be displayed properly.
  bool YUV_, use_other_classes_as_negatives_;
};


void BoundingBoxTargets::Configure(int num_objects, int output_image_size, int input_image_size, bool use_other_classes_as_negatives) {
  num_objects_ = num_objects;
  if (num_objects_ == 0) num_objects_ = 1;
  output_image_size_ = output_image_size;
  input_image_size_ = input_image_size;

  if (output_image_size == 1) {
    // No spatial structure in output, can use standard methods.
    cerr << "This class is not designed for output size 1!" << endl;
    exit(1);
  }
  //num_channels_ = DIVUP(num_objects, 16);  // ConvUp wants (numFilters % 16) == 0 //fixed!
  num_channels_ = num_objects;
  disp_ = new CImgDisplay();
  use_other_classes_as_negatives_ = use_other_classes_as_negatives;
}


void BoundingBoxTargets::SetMeanImage(const string& mean_image_file) {
  mean_image_.assign(mean_image_file.c_str());
  if (YUV_) mean_image_.RGBtoYUV();
  add_mean_image_ = true;
}

void BoundingBoxTargets::ComputeDeriv2(Matrix& state, const vector<vector<box>>& boxes, Matrix& deriv) {
  deriv.Set(0);
  deriv.CopyToHost();
  state.CopyToHost();
  const float scale = (float)output_image_size_ / input_image_size_;
  const int num_images = state.GetRows();
  const int width = output_image_size_;
  const int height = output_image_size_;
  int xmin, xmax, ymin, ymax;
  float* deriv_data = deriv.GetHostData();
  float* state_data = state.GetHostData();
  for (int image_id = 0; image_id < num_images; image_id++) {
    set<int> labels;
    for (box b : boxes[image_id]) {
      if (labels.find(b.label) == labels.end()) {
        labels.insert(b.label);
        for (int row = 0; row < height; row++) {
          for (int col = 0; col < width; col++) {
            unsigned long i = image_id + num_images * (col + width * (row + height * b.label));
            deriv_data[i] = state_data[i];
          }
        }
      }
    }

    for (box b : boxes[image_id]) {
      xmin = Bound((int)(scale * b.xmin), 0, output_image_size_);
      ymin = Bound((int)(scale * b.ymin), 0, output_image_size_);
      xmax = Bound((int)(scale * b.xmax), 0, output_image_size_);
      ymax = Bound((int)(scale * b.ymax), 0, output_image_size_);
      for (int row = ymin; row < ymax; row++) {
        for (int col = xmin; col < xmax; col++) {
          unsigned long i = image_id + num_images * (col + width * (row + height * b.label));
          deriv_data[i] = state_data[i] - 1;
        }
      }
    }
  }
  deriv.CopyToDevice();
}


void BoundingBoxTargets::ComputeDeriv(Matrix& state, const vector<vector<box>>& boxes, Matrix& deriv, int label) {
  deriv.Set(0);
  int start, end;
  if (label >= 0) {
    cerr << "This should not happen any more" << endl;
    exit(1);
    start = label * output_image_size_ * output_image_size_;
    end = start + output_image_size_ * output_image_size_;
  } else {
    start = 0;
    end = state.GetCols();
  }
  int num_objects = state.GetCols() / (output_image_size_ * output_image_size_);
  const float weight_factor = 1.0f / num_objects;

  deriv.CopyToHostSlice(start, end);
  state.CopyToHostSlice(start, end);
  float* deriv_data = deriv.GetHostData();
  float* state_data = state.GetHostData();
  const int border_width = 0;
  const float scale = (float)output_image_size_ / input_image_size_;
  const int num_images = state.GetRows();
  const int width = output_image_size_;
  const int height = output_image_size_;
  int xmin, xmax, ymin, ymax;
  for (int image_id = 0; image_id < num_images; image_id++) {
    // cout << "Computing deriv for image " << image_id << endl;
    // cout << "Number of obejcts " << num_objects << endl;
    for (box b : boxes[image_id]) {
      if (b.is_neg) {
        // negative image.
        // Target is 0's for the heat map for b.label
        // Don't care for all other heat maps.
        for (int row = 0; row < height; row++) {
          for (int col = 0; col < width; col++) {
            unsigned long i = image_id + num_images * (col + width * (row + height * b.label));
            deriv_data[i] = state_data[i];
          }
        }
      } else if (use_other_classes_as_negatives_) {
        xmin = Bound((int)(scale * b.xmin), 0, width);
        xmax = Bound((int)(scale * b.xmax), 0, width);
        ymin = Bound((int)(scale * b.ymin), 0, height);
        ymax = Bound((int)(scale * b.ymax), 0, height);

        // The heat map of every label other than b.label
        // Target is 0.
        for (int l = 0; l < num_objects; l++) {
          if (l == b.label) {
            for (int row = 0; row < height; row++) {
              for (int col = 0; col < width; col++) {
                unsigned long i = image_id + num_images * (col + width * (row + height * b.label));
                deriv_data[i] = state_data[i];
              }
            }
          } else {
            // cout << "Image id " << image_id << " has label " << b.label << " setting target 0 for label " << l << endl;
            for (int row = ymin; row < ymax; row++) {
              for (int col = xmin; col < xmax; col++) {
                unsigned long i = image_id + num_images * (col + width * (row + height * l));
                deriv_data[i] = weight_factor * state_data[i];
              }
            }
          }
        }
      }
    }

    for (box b : boxes[image_id]) {
      if (b.is_neg) continue;
      // Positive image.
      // Target is 1 inside the box for b.label, 0 just outside the box for b.label and don't care else where.
      xmin = Bound((int)(scale * b.xmin), 0, width);
      xmax = Bound((int)(scale * b.xmax), 0, width);
      ymin = Bound((int)(scale * b.ymin), 0, height);
      ymax = Bound((int)(scale * b.ymax), 0, height);

      for (int row = ymin - border_width; row < ymax + border_width; row++) {
        if (row < 0 || row >= height) continue;
        for (int col = xmin - border_width; col < xmax + border_width; col++) {
          if (col < 0 || col >= width) continue;
          unsigned long i = image_id + num_images * (col + width * (row + height * b.label));
          deriv_data[i] = state_data[i] - ((row >= ymin && row < ymax && col >= xmin && col < xmax) ? 1:0);
        }
      }
    }
  }
  deriv.CopyToDeviceSlice(start, end);
}

void BoundingBoxTargets::Display(Layer& input, Layer& predictions, const vector<vector<box>>& boxes) {
  Display(input, predictions, boxes, 1, 0);
}

void BoundingBoxTargets::Display(Layer& input, Layer& predictions, const vector<vector<box>>& boxes, int num_images_to_display, int obj_id) {
  if (obj_id < 0) obj_id = 0;  // < 0 indictaes this_label_batch is not relevant.
  cudamat* image = input.GetState().GetMat();
  cudamat* deriv = predictions.GetDeriv().GetMat();
  cudamat* preds = predictions.GetState().GetMat();
  const int num_images = image->size[0];
  copy_to_host(image);
  copy_to_host(preds);
  copy_to_host(deriv);

  const int input_width = input.GetSize();
  const int input_height = input.GetSize();
  for (int image_id = 0; image_id < num_images_to_display; image_id++) {
    if (boxes[image_id].size() > 0) {
      obj_id = boxes[image_id][0].label;
    } else {
      obj_id = 0;
    }
    //cout << "Displaying object id " << obj_id << endl;
    CImg<unsigned char> cimage(input_width, input_height, 1, 3);
    for (int c = 0; c < 3; c++) {
      for (int i = 0; i < input_height; i++) {
        for (int j = 0; j < input_width; j++) {
          int index = image_id + num_images * (j + input_width * (i + input_height * c));
          if (add_mean_image_) {
            cimage(j, i, 0, c) = mean_image_(j, i, 0, c) + image->data_host[index]; 
          } else {
            cimage(j, i, 0, c) = 120 + image->data_host[index]; 
          }
        }
      }
    }
    if (YUV_) cimage.YUVtoRGB();

    //CImg<float> cimage2(cimage);
    // draw bounding box on cimage, use cimage2 to show predictions.

    //cout << endl;
    //cout << "Source image id " << src_image_id << endl;
    const unsigned char color[] = {0, 0, 255};
    const unsigned char text_foreground_color[] = {255, 255, 255};
    for (box b : boxes[image_id]) {
      for (int p = -1; p < 1; p++) {
        cimage.draw_rectangle(b.xmin + p, b.ymin + p, b.xmax + p, b.ymax + p, color, 1, ~0U);
      }
      stringstream ss;
      ss << "label " << b.label;
      cimage.draw_text(b.xmin, b.ymin, ss.str().c_str(), text_foreground_color, color);
    }

    unsigned char colors[][3] = { {0, 255, 0}, {255, 0, 0}, {0, 0, 255}};

    const int width = output_image_size_;
    const int height = output_image_size_;
    const float height_scale = (float)input_height / height;
    const float width_scale = (float)input_width / width;
    CImg<unsigned char> cimage2(input_width, input_height, 1, 3);
    CImg<unsigned char> cimage3(input_width, input_height, 1, 3);
    int i, j, c, output_i, output_j;
    float prob, grad;
    unsigned long index;

    for (i = 0; i < input_height; i++) {
      output_i = (int)(i / height_scale);
      for (j = 0; j < input_width; j++) {
        output_j = (int)(j / width_scale);

        for (c = 0; c < 3; c++) {
          cimage2(j, i, 0, c) = 0;
          cimage3(j, i, 0, c) = 0;
        }
        index = image_id + num_images * (output_j + width * (output_i + height * obj_id));
        prob = preds->data_host[index];
        grad = deriv->data_host[index];
        for (c = 0; c < 3; c++) {
          cimage2(j, i, 0, c) += (unsigned char)(prob * colors[0][c]);
          cimage3(j, i, 0, c) += (unsigned char)((grad+1)/2 * 255);
        }
      }
    }
    
    CImgList<float> cimagelist(cimage, cimage2, cimage3);
    cimagelist.display(*disp_);
    if (num_images_to_display > 1) {
      usleep(500000);
    }
  }
}


