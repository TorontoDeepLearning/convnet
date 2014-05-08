#define cimg_use_jpeg
#include "CImg.h"
#include "convnet_cpu.h"
#include <string>
#include <iostream>
using namespace cimg_library;
using namespace std;

CImgDisplay disp;
void GetCoordinates(int image_size, int width, int height, int position,
                    int* left, int* top, bool* flip) {
  *flip = position >= 5;
  position %= 5;
  int x_slack = width - image_size;
  int y_slack = height - image_size;
  switch(position) {
    case 0 :  // Center. 
            *left = x_slack / 2;
            *top = y_slack / 2;
            break;
    case 1 :  // Top left.
            *left = 0;
            *top = 0;
            break;
    case 2 :  // Top right.
            *left = x_slack;
            *top = 0;
            break;
    case 3 :  // Bottom right.
            *left = x_slack;
            *top = y_slack;
            break;
    case 4 :  // Bottom left.
            *left = 0;
            *top = y_slack;
            break;
  }
}


void LoadImage(const string& filename, int image_size, int big_image_size, int position, float* data, bool display) {
  CImg<float> image;
  image.assign(filename.c_str());
    
  // Resize it so that the shorter side is big_image_size.
  int width = image.width(), height = image.height();
  int new_width, new_height;
  if (width > height) {
    new_height = big_image_size;
    new_width = (width * big_image_size) / height;
  } else {
    new_width = big_image_size;
    new_height = (height * big_image_size) / width;
  }
  image.resize(new_width, new_height, 1, -100, 3);

  int left, top;
  bool flip;
  GetCoordinates(image_size, image.width(), image.height(), position,
                 &left, &top, &flip);

  CImg<float> img = image.get_crop(left, top, left + image_size - 1,
                                   top + image_size - 1, true);

  if (flip) img.mirror('x');
  if (display) img.display(disp);

  int num_image_colors = img.spectrum();
  int num_pixels = image_size * image_size;
  if (num_image_colors >= 3) {  // Image has at least 3 channels.
    float *R = img.data(),
          *G = img.data() + num_pixels,
          *B = img.data() + 2 * num_pixels;
    for (int i = 0; i < 3 * num_pixels; i+=3) {
      data[i] = *(R++);
      data[i+1] = *(G++);
      data[i+2] = *(B++);
    }
  } else if (num_image_colors == 1) {  // Image has 1 channel.
    float *R = img.data();
    for (int i = 0; i < 3 * num_pixels; i+=3) {
      data[i] = *R;
      data[i+1] = *R;
      data[i+2] = *R;
      R++;
    }
  } else {
    cerr << "Image has " << num_image_colors << "colors." << endl;
    exit(1);
  }
}

int main(int argc, char** argv) {
  string model_structure(argv[1]);
  string model_parameters(argv[2]);

  cout << "Loading model from " << model_structure << endl;
  ConvNetCPU net(model_structure, model_parameters, 1);
  cout << "Done" << endl;
  vector<Layer*> layers;
  int big_image_size = 256;
  int image_size = 224;
  int position = 0;
  float* data = new float[image_size * image_size * 3];

  for (int c = 3; c < argc; c++) {
    layers.push_back(net.GetLayerByName(string(argv[c])));
  }
  while (true) {
    string imgfile;
    cin >> imgfile;
    if (cin.eof()) break;
    if (imgfile.empty()) break;
    cout << imgfile << endl;
    LoadImage(imgfile, image_size, big_image_size, position, data, false);
    for (int i = 0; i < 10; i++) cout << data[i] << " ";
    cout << endl;
    net.Fprop(data, 1);
    for (Layer* l : layers) {
      const float* state = l->GetState();
      const int num_dims = l->GetDims();
      for (int i = 0; i < num_dims && i < 10; i++) {
        cout << state[i] << " ";
      }
      cout << endl;
    }
  }
  delete data;
}
