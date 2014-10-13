#include "convnet_cpu.h"
#include "../src/image_iterators.h"

#include <opencv2/core.hpp>

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;

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
    int num_pixels = image_size * image_size;
    unsigned char* data = new unsigned char[image_size * image_size * 3];
    unsigned char* image_buf = new unsigned char[image_size * image_size * 3];

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
      vector<string> filenames;
      filenames.push_back(imgfile);

      RawImageFileIterator<unsigned char> it = RawImageFileIterator<unsigned char>(
        filenames, image_size, big_image_size, false, false, false);
      it.GetNext(image_buf);
      unsigned char *R = image_buf;
      unsigned char *G = image_buf +   num_pixels;
      unsigned char *B = image_buf + 2*num_pixels;
      for (int i=0; i<3*num_pixels; i+=3) {
        data[i  ] = *(R++);
        data[i+1] = *(G++);
        data[i+2] = *(B++);
      }

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
    delete[] image_buf;
    for (ofstream& f : outf) {
      f.close();
    }

    return 0;
}
