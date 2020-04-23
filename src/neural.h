#pragma once

#include "matrix.h"
#include "utils.h"

#include <vector>

using namespace std;


enum Activation { LINEAR, LOGISTIC, TANH, RELU, LRELU, SOFTMAX };
enum LossFunction { CROSS_ENTROPY, L2_LOSS, L1_LOSS };

struct Layer {
  // Runtime Data terms
  Matrix in;              // Input to a layer (aka x)
  Matrix out1;            // Output before activation (aka xw)
  Matrix out2;            // Output after activation (actual output, aka y)
  // Backpass saved terms
  Matrix grad_out1;
  Matrix grad_in;

  // Weight and weight management
  Matrix w;               // Current weights for a layer
  Matrix grad_w;          // Current weight updates
  Matrix v;               // Past weight updates (for use with momentum)

  // Type
  Activation activation;  // Activation the layer uses


  // Constructors
  Layer() = default;
  Layer(int input, int output, Activation activation);

  // Operations

  Matrix forward(const Matrix &in);
  Matrix backward(const Matrix &dl);

  void update_weights(double rate, double momentum, double decay);
};

struct Data {
  Matrix X;
  Matrix y;
  mutable std::mt19937 mt;
  
  Data() = default;
  Data(int N, int size_x, int size_y) : X(N, size_x), y(N, size_y) {}

  Data random_batch(int batch_size) const;

};

struct Dataset { Data train, test; };

struct Model {
  std::vector<Layer> layers;
  LossFunction loss;
  
  
  double compute_loss(const Matrix &y, const Matrix &p);
  Matrix loss_derivative(const Matrix &y, const Matrix &p);
  
  
  Matrix forward(Matrix in);
  void backward(Matrix grad_loss);

  void update_weights(double rate, double momentum, double decay);
  void train(const Data &data, int batch_size, int iters, double rate, double momentum, double decay);

  double accuracy(const Data &d);   // RUNS FORWARD
  double accuracy2(const Data &d, const Matrix &p);  // DOES NOT RUN FORWARD
};

void set_verbose(bool verbose);

Matrix forward_activate_matrix(const Matrix &matrix, Activation a);
Matrix backward_activate_matrix(const Matrix &out, const Matrix &grad, Activation a);

Matrix forward_weights(const Layer &l, const Matrix &in);
Matrix forward_activation(const Layer &l, const Matrix &out1);

Matrix backward_xw(const Layer &l, const Matrix &grad_y);
Matrix backward_w(const Layer &l);
Matrix backward_x(const Layer &l);

double cross_entropy_loss(const Matrix &y, const Matrix &p);
double l2_loss(const Matrix &y, const Matrix &p);
double l1_loss(const Matrix &y, const Matrix &p);

Data read_cifar(const std::string &file, int dataset, int labels);
Data read_mnist(const std::string &image_file, const std::string &label_file);

Matrix forward_linear(const Matrix &mat);
Matrix backward_linear(const Matrix &out, const Matrix &prev_grad);
Matrix forward_logistic(const Matrix &mat);
Matrix backward_logistic(const Matrix &out, const Matrix &grad);
Matrix forward_tanh(const Matrix &mat);
Matrix backward_tanh(const Matrix &out, const Matrix &grad);
Matrix forward_relu(const Matrix &mat);
Matrix backward_relu(const Matrix &out, const Matrix &grad);
Matrix forward_lrelu(const Matrix &mat);
Matrix backward_lrelu(const Matrix &out, const Matrix &grad);
Matrix forward_softmax(const Matrix &mat);
Matrix backward_softmax(const Matrix &out, const Matrix &prev_grad);
