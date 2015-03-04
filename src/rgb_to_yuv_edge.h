#ifndef RGB_TO_YUV_EDGE_H_
#define RGB_TO_YUV_EDGE_H_
#include "edge.h"

/** Implements an edge that maps RGB to YUV.*/
class RGBToYUVEdge : public Edge {
 public:
  RGBToYUVEdge(const config::Edge& edge_config);
  virtual std::string GetDescription();
  virtual void ComputeUp(Matrix& input, Matrix& output, bool overwrite, bool train);
  virtual void ComputeDown(Matrix& deriv_output, Matrix& input,
                           Matrix& output, Matrix& deriv_input, bool overwrite);
};
#endif
