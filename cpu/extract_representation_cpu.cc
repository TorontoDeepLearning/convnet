#define cimg_use_jpeg
#include "CImg/CImg.h"

#include "convnet_cpu.h"

#include <opencv2/core.hpp>

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>

using namespace cimg_library;
using namespace std;
using namespace cv;

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


void LoadImage(const string& filename, int image_size, int big_image_size, int position, unsigned char* data, bool display) {
  CImg<unsigned char> image;
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

  int left = 0, top = 0;
  bool flip = false;
  GetCoordinates(image_size, image.width(), image.height(), position,
                 &left, &top, &flip);

  CImg<unsigned char> img = image.get_crop(left, top, left + image_size - 1,
                                   top + image_size - 1, true);

  if (flip) img.mirror('x');
  if (display) img.display(disp);

  int num_image_colors = img.spectrum();
  int num_pixels = image_size * image_size;
  if (num_image_colors >= 3) {  // Image has at least 3 channels.
    unsigned char *R = img.data(),
                  *G = img.data() + num_pixels,
                  *B = img.data() + 2 * num_pixels;
    for (int i = 0; i < 3 * num_pixels; i+=3) {
      data[i] = *(R++);
      data[i+1] = *(G++);
      data[i+2] = *(B++);
    }
  } else if (num_image_colors == 1) {  // Image has 1 channel.
    unsigned char *R = img.data();
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

void split(const string &s, vector<string> &elems, char delim)
{
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim))
    {
        elems.push_back(item);
    }
}

int main(int argc, char** argv) {
    const char *keys =
            "{ layer      l || Layer name }"
            "{ model      m || Model file }"
            "{ parameters p || Parameter file }"
            "{ mean       s || Pixel mean file }"
            "{ output     o || Output directory }";
    CommandLineParser parser(argc, argv, keys);
    string layer_name(parser.get<string>("layer"));
    string model(parser.get<string>("model"));
    string param(parser.get<string>("parameters"));
    string mean_file(parser.get<string>("mean"));
    string output_dir(parser.get<string>("output"));
    if (layer_name.empty() || model.empty() ||
        param.empty() || mean_file.empty() || output_dir.empty())
    {
        parser.printMessage();
        return -1;
    }

    vector<string> layer_names;
    split(layer_name, layer_names, ';');


    cpu::ConvNetCPU net(model, param, mean_file, 1);
    vector<cpu::Layer*> layers;
    int big_image_size = 256;
    int image_size = 224;
    int position = 0;
    unsigned char* data = new unsigned char[image_size * image_size * 3];

    vector<ofstream> outf(layer_names.size());
    cout << "Writing outputs to " << endl;
    int i = 0;
    for (const string& layer_name : layer_names) {
      layers.push_back(net.GetLayerByName(layer_name));
      string filename = output_dir + "/" + layer_name + ".txt";
      cout << filename << endl;
      outf[i++].open(filename, ofstream::out);
    }

    bool show_time = false;
    chrono::time_point<chrono::system_clock> start_t, load_t, fprop_t, end_t;
    chrono::duration<double> time_diff1, time_diff2, time_diff3;
    while (true) {
      start_t = chrono::system_clock::now();

      string imgfile;
      cin >> imgfile;
      if (cin.eof()) break;
      if (imgfile.empty()) break;
      cout << imgfile;

      LoadImage(imgfile, image_size, big_image_size, position, data, false);
      load_t = chrono::system_clock::now();

      net.Fprop(data, 1);
      fprop_t = chrono::system_clock::now();

      int k = 0;
      for (cpu::Layer* l : layers) {
        const float* state = l->GetState();
        const int num_dims = l->GetDims();
        for (int i = 0; i < num_dims; i++) {
          outf[k] << state[i] << " ";
        }
        outf[k] << endl;
        k++;
      }

      end_t = chrono::system_clock::now();
      time_diff1 = load_t - start_t;
      time_diff2 = fprop_t  - load_t;
      time_diff3 = end_t  - fprop_t;
      if (show_time) {
        printf(" Time load %f s fprop %f s write %f s",
               time_diff1.count(), time_diff2.count(), time_diff3.count());
      }
      start_t = end_t;
      cout << endl;
    }
    delete[] data;
    for (ofstream& f : outf) {
      f.close();
    }

    return 0;
}
