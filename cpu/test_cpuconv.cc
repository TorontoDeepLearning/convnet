#include "cpuconv.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv) {
  float image[10*10], filters[2*2*2], output[5*5*2];
  for (int i = 0; i < 10*10; i++) image[i] = 1;
  for (int j = 0; j < 2*2; j++) filters[j] = 1;
  for (int j = 2*2; j < 2*2*2; j++) filters[j] = 2;
  
  CPUMatrix::ConvUp(image, filters, output,
            1, 1, 2,
            10, 10,
            2, 2,
            2, 2,
            0, 0,
            1, 0);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cout << output[2*(5*i+j)] << ":" << output[2*(5*i+j) + 1] << " ";
    }
    cout << endl;
  }

  return 0;
}
