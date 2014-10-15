#include "loss_functions.h"
#include <iostream>
using namespace std;

LossFunction* LossFunction::ChooseLossFunction(const config::Layer::LossFunction& lf) {
  LossFunction* f = NULL;
  switch (lf) {
    case config::Layer::SQUARED_ERROR:
      f = new SquaredError();
      break;
    case config::Layer::LINEAR_ERROR:
      f = new LinearError();
      break;
    case config::Layer::CROSS_ENTROPY_MULTINOMIAL:
      f = new CrossEntropyMultinomial();
      break;
    case config::Layer::CROSS_ENTROPY_MULTINOMIAL_DISTRIBUTED:
      f = new CrossEntropyDistributed();
      break;
    case config::Layer::CROSS_ENTROPY_BINARY:
      f = new CrossEntropyBinary();
      break;
    case config::Layer::CLASSIFICATION_MULTINOMIAL:
      f = new ClassificationMultinomial();
      break;
    case config::Layer::CLASSIFICATION_BINARY:
      f = new ClassificationBinary();
      break;
    default:
      cerr << "Unknown loss function " << endl;
      exit(1);
  }
  return f;
}

LossFunction::LossFunction() {
}

SquaredError::SquaredError() : LossFunction() {
}

float SquaredError::GetLoss(Matrix& y, Matrix& t) {
  Matrix temp;
  Matrix::GetTemp(t.GetRows(), t.GetCols(), temp);
  y.Subtract(t, temp);
  float norm = temp.EuclidNorm();
  float res = 0.5 * norm * norm;
  return res;
}

void SquaredError::GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy) {
  y.Subtract(t, dLbydy);
}

LinearError::LinearError() : LossFunction() {
}

float LinearError::GetLoss(Matrix& y, Matrix& t) {
  Matrix temp;
  Matrix::GetTemp(t.GetRows(), t.GetCols(), temp);
  y.Subtract(t, temp);
  float res = temp.Sum();
  return res;
}

void LinearError::GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy) {
  dLbydy.Set(1);
}

CrossEntropyMultinomial::CrossEntropyMultinomial() : LossFunction() {
}

float CrossEntropyMultinomial::GetLoss(Matrix& y, Matrix& t) {
  Matrix temp;
  Matrix::GetTemp(t.GetRows(), 1, temp);
  Matrix::SoftmaxCE(y, t, temp);
  float res = temp.Sum();
  return res;
}

void CrossEntropyMultinomial::GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy) {
  Matrix::SoftmaxCEDeriv(y, t, dLbydy);
}

CrossEntropyBinary::CrossEntropyBinary() : LossFunction() {
}

float CrossEntropyBinary::GetLoss(Matrix& y, Matrix& t) {
  cerr<< "Not implemented" << endl;
  return 0;
}

void CrossEntropyBinary::GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy) {
  Matrix::LogisticCEDeriv(y, t, dLbydy);
}

CrossEntropyDistributed::CrossEntropyDistributed() : LossFunction() {
}

float CrossEntropyDistributed::GetLoss(Matrix& y, Matrix& t) {
  Matrix temp;
  Matrix::GetTemp(t.GetRows(), t.GetCols(), temp);
  Matrix::SoftmaxDistCE(y, t, temp);
  return temp.Sum();
}

void CrossEntropyDistributed::GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy) {
  y.Subtract(t, dLbydy);
}

ClassificationMultinomial::ClassificationMultinomial() : LossFunction() {
}

float ClassificationMultinomial::GetLoss(Matrix& y, Matrix& t) {
  Matrix temp;
  Matrix::GetTemp(t.GetRows(), 1, temp);
  Matrix::SoftmaxCorrect(y, t, temp);
  float res = temp.Sum();
  return res;
}

void ClassificationMultinomial::GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy) {
  cerr << "Derivative is not nice for ClassificationMultinomial loss." << endl;
  dLbydy.Set(0);
}

ClassificationBinary::ClassificationBinary() : LossFunction() {
}

float ClassificationBinary::GetLoss(Matrix& y, Matrix& t) {
  Matrix temp;
  Matrix::GetTemp(t.GetRows(), 1, temp);
  Matrix::LogisticCorrect(y, t, temp);
  return temp.Sum();
}

void ClassificationBinary::GetLossDerivative(Matrix& y, Matrix& t, Matrix& dLbydy) {
  cerr << "Derivative is not nice for ClassificationBinary loss." << endl;
  dLbydy.Set(0);
}
/*
 HingeQuadraticLayer::HingeQuadraticLayer(const config::Layer& config) :
  SoftmaxLayer(config), margin_(config.hinge_margin()) {}

void HingeQuadraticLayer::ApplyActivation(bool train) {
  ApplyDropout(train);
}

void HingeQuadraticLayer::ApplyDerivativeOfActivation() {
  ApplyDerivativeofDropout();
}

void HingeQuadraticLayer::ComputeDeriv() {
  Matrix::HingeLossDeriv(y, t, deriv_, true, margin_);
}

HingeLinearLayer::HingeLinearLayer(const config::Layer& config) :
  HingeQuadraticLayer(config) {}

void HingeLinearLayer::ComputeDeriv() {
  Matrix::HingeLossDeriv(y, t, deriv_, false, margin_);
}
 * */
