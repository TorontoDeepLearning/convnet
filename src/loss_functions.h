#ifndef LOSS_FUNCTIONS_H_
#define LOSS_FUNCTIONS_H_

#ifdef USE_CUDA
#include "matrix.h"
#else
#include "CPUMatrix.h"
#endif

#include "util.h"

class LossFunction {
 public:
  LossFunction();
  virtual ~LossFunction() {}

  virtual float GetLoss(Matrix& y, Matrix& t) = 0;
  virtual void GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy) = 0;

  static LossFunction* ChooseLossFunction(const config::Layer::LossFunction& lf);
};

// L = 0.5 * ||y-t||^2.
class SquaredError : public LossFunction {
 public:
  SquaredError();
  float GetLoss(Matrix& y, Matrix& t);
  void GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy);
};

// L = y-t (This is used for grad check because it makes things linear
// even over large deviations).
class LinearError : public LossFunction {
 public:
  LinearError();
  float GetLoss(Matrix& y, Matrix& t);
  void GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy);
};

// L = sum_i t_i * log(y_i), where only one t_i = 1, rest 0.
class CrossEntropyMultinomial : public LossFunction {
 public:
  CrossEntropyMultinomial();
  float GetLoss(Matrix& y, Matrix& t);  // t is a column vector of indices.
  void GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy);
};

// L = sum_i t_i * log(y_i).
class CrossEntropyDistributed : public LossFunction {
 public:
  CrossEntropyDistributed();
  float GetLoss(Matrix& y, Matrix& t);
  void GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy);
};

// L = t * log(y) + (1-t) * log (1-y).
class CrossEntropyBinary : public LossFunction {
 public:
  CrossEntropyBinary();
  float GetLoss(Matrix& y, Matrix& t);
  void GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy);
};

// L = # {(argmax_i y_i) != t}.
class ClassificationMultinomial: public LossFunction {
 public:
  ClassificationMultinomial();
  float GetLoss(Matrix& y, Matrix& t);
  void GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy);
};

// L = # {|y - t| < 0.5}.
class ClassificationBinary: public LossFunction {
 public:
  ClassificationBinary();
  float GetLoss(Matrix& y, Matrix& t);
  void GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy);
};
#endif
