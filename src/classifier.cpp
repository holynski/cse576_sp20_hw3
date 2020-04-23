
#include <cmath>

#include "matrix.h"
#include "neural.h"

bool verbose = false;

void set_verbose(bool v) {
  verbose = v;
}

void debug_print(const char *name, const Matrix &m, int max_rows = 2, int max_cols = 4) {
  if (verbose) {
    printf("%s: ", name);
    m.print_size();
    m.print(max_rows, max_cols);
  }
}


/////////////////////////// FORWARD PASS /////////////////////////////
// Forward propagate information through a layer
// Function is split into 2 parts:
// 1: applying weight matrix
// 2: applying activations

// const Layer& l: the layer
// const Matrix& in: input to layer
// returns: matrix that is output before the activation layer
Matrix forward_weights(const Layer &l, const Matrix &in) {
  Matrix output;
  // TODO: Multiply input by weights and return the result
  NOT_IMPLEMENTED();

  assert(output.rows == in.rows);
  assert(output.cols == l.w.cols);
  return output;
}

// const Layer& l: the layer
// const Matrix& out1: output before activation
// returns: matrix that is output of the layer after activation
Matrix forward_activation(const Layer &l, const Matrix &out1) {
  Matrix output;
  // TODO: Apply activation function and return
  // Hint: Use forward_activate_matrix in activations.cpp.
  NOT_IMPLEMENTED();

  return output;
}

// READ THIS FUNCTION
// BUT DO NOT MODIFY
Matrix Layer::forward(const Matrix &in) {
  Layer &l = *this;
  l.in = in; // Save the input for backpropagation
  l.out1 = forward_weights(l, in);     // applying weights
  l.out2 = forward_activation(l, l.out1); // applying activation
  return l.out2;
}

/////////////////////////// BACKWARD PASS /////////////////////////////
// Backward propagate derivatives through a layer
// Function is split into 3 parts:
// 1: Computing grad_xw ( partial derivative of loss w.r.t. preactivated output)
// 2: Computing grad_w ( partial derivative of loss w.r.t. weight)
// 3: Computing grad_x ( partial derivative of loss w.r.t. input)


// const Layer& l: the layer
// const Matrix& grad_y: partial derivative of loss w.r.t. output of layer
// returns: Matrix, partial derivative of loss w.r.t. input to (xw)
Matrix backward_xw(const Layer &l, const Matrix &grad_y) {
  Matrix grad_xw;
  // TODO (1.4.1): compute dL/d(xw) and return it
  // Hint:
  //  grad_y is dL/dy
  //  dL/d(xw) = dL/dy * dy/d(xw)
  //           = dL/dy * df(xw)/d(xw)
  //           = dL/dy * f'(xw)
  // Hint: Use backward_activate_matrix in activations.cpp.
  NOT_IMPLEMENTED();

  return grad_xw;
}

// const Layer& l: the layer
// returns: Matrix, partial derivative of loss w.r.t. input to the weights
Matrix backward_w(const Layer &l) {
  // Get the relevant quantities from the layer (see forward() and backward() function for reference)
  // TODO (1.4.2): then calculate dL/dw and return it
  // Hint:
  //  dL/dw = d(xw)/dw * dL/d(xw) = x * dL/d(xw)
  Matrix grad_w;
  NOT_IMPLEMENTED();

  assert_same_size(grad_w, l.w);
  return grad_w;
}

// const Layer& l: the layer
// returns: Matrix, partial derivative of loss w.r.t. input to the input
Matrix backward_x(const Layer &l) {
  // Get the relevant quantities from the layer (see forward() and backward() function for reference)
  // TODO (1.4.3): finally, calculate dL/dx and return it
  Matrix grad_x;
  NOT_IMPLEMENTED();

  assert_same_size(grad_x, l.in);
  return grad_x;
}

// READ THIS FUNCTION
// BUT DO NOT MODIFY
Matrix Layer::backward(const Matrix &grad_y) {
  Layer &l = *this;
  grad_out1 = backward_xw(l, grad_y);
  grad_w = backward_w(l);
  grad_in = backward_x(l);
  return grad_in;
}

// Update the weights at Layer l
// Layer& l: the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(Layer &l, double rate, double momentum, double decay) {
  // TODO: calculate the weight updates
  // Hint: Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1} and save it to l.v
  NOT_IMPLEMENTED();

  // TODO: update the weights and save to l.w.
  // Hint: w_{t+1} = w_t + ηΔw_t
  NOT_IMPLEMENTED();
}

// DO NOT MODIFY.
// Layer Constructor
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// Activation activation: the activation function to use
Layer::Layer(int input, int output, Activation activation)
    : w(random_matrix(input, output) * sqrt(2. / input)), // random initialization
      grad_w(input, output),
      v(input, output),
      activation(activation) {

}

// DO NOT MODIFY.
// Run a model on input X
// Model& m: model to run
// Matrix X: input to model
// returns: result matrix
Matrix Model::forward(Matrix X) {
  for (auto &layer:layers) {
    X = layer.forward(X);
  }
  return X;
}

// DO NOT MODIFY.
// Run a model backward given gradient dL
// Model& m: model to run
// Matrix grad: partial derivative of loss w.r.t. model output dL/dy
void Model::backward(Matrix grad) {
  for (int i = (int) layers.size() - 1; i >= 0; i--) {
    grad = layers[i].backward(grad);
  }
}

// DO NOT MODIFY.
// Update the model weights
// Model& m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void Model::update_weights(double rate, double momentum, double decay) {
  for (auto &layer: layers) {
    layer.update_weights(rate, momentum, decay);
  }
}

// DO NOT MODIFY.
// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(const double *a, int n) {
  return int(std::max_element(a, a + n) - a);
}

// Calculate the accuracy of a model on some data d
// DOES NOT RUN FORWARD
// const Data& d: data to run on
// const Matrix& p: predictions
// returns: accuracy, number correct / total
double Model::accuracy2(const Data &d, const Matrix &p) {
  int correct = 0;
  for (int i = 0; i < d.y.rows; i++) {
    correct += max_index(d.y[i], d.y.cols) == max_index(p[i], p.cols);
  }

  return (double) correct / d.y.rows;
}

// DO NOT MODIFY.
// Calculate the accuracy of a model on some data d
// const Data& d: data to run on
// returns: accuracy, number correct / total
double Model::accuracy(const Data &d) {
  Matrix p = this->forward(d.X);
  return accuracy2(d, p);
}

// DO NOT MODIFY.
// Calculate the cross-entropy loss for a set of predictions
// const Matrix& y: the correct values
// const Matrix& p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(const Matrix &y, const Matrix &p) {
  assert_same_size(y, p);
  double sum = 0;
  for (int i = 0; i < y.rows; i++) {
    for (int j = 0; j < y.cols; j++) {
      sum += -y(i, j) * log(p(i, j));
    }
  }
  return sum / y.rows;
}

// Calculate the L2 loss for a set of predictions
// const Matrix& y: the correct values
// const Matrix& p: the predictions
// returns: average L2 loss over data points
double l2_loss(const Matrix &y, const Matrix &p) {
  assert_same_size(y, p);
  // TODO

  NOT_IMPLEMENTED();

  return 0;
}

// Calculate the L1 loss for a set of predictions
// const Matrix& y: the correct values
// const Matrix& p: the predictions
// returns: average L1 loss over data points
double l1_loss(const Matrix &y, const Matrix &p) {
  assert_same_size(y, p);
  // TODO

  NOT_IMPLEMENTED();

  return 0;
}

// Calculate the loss for a set of predictions
// const Matrix& y: the correct values
// const Matrix& p: the predictions
// returns: average loss over data points
double Model::compute_loss(const Matrix &y, const Matrix &p)
  {
       if(loss==CROSS_ENTROPY)return cross_entropy_loss(y, p);
  else if(loss==L2_LOSS)      return l2_loss           (y, p);
  else if(loss==L1_LOSS)      return l1_loss           (y, p);
  else assert(false && "Invalid loss function");
  }

// Calculate the derivative of loss with respect to a set of predictions
// const Matrix& y: the correct values
// const Matrix& p: the predictions
// returns: derivative with respect to each element in predictions
// NOTE: not averaging here. Averaging is happening in Model::train
Matrix Model::loss_derivative(const Matrix &y, const Matrix &p)
  {
       if(loss==CROSS_ENTROPY)return elementwise_divide(y, p);
  else if(loss==L2_LOSS)      return 2*(y-p);
  else if(loss==L1_LOSS)
    {
    Matrix d=y*0;
    for(int q1=0;q1<d.rows;q1++)for(int q2=0;q2<d.cols;q2++)d(q1,q2)=2*(y(q1,q2)>p(q1,q2))-1;
    return d;
    }
  else assert(false && "Invalid loss function");
  }


// DO NOT MODIFY.
// Train a model on a dataset using SGD
// Data& d: dataset to train on
// int batch_size: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void Model::train(const Data &data, int batch_size, int iters, double rate, double momentum, double decay) {
  for (int iter = 0; iter < iters; iter++) {
    Data batch = data.random_batch(batch_size);

    Matrix y = this->forward(batch.X);

    double loss = this->compute_loss(batch.y, y);
    double accu = this->accuracy2(batch, y);
    
    printf("Iteration: %6d: Loss: %12.6lf   Batch Accuracy: %8.3lf \n", iter, loss, accu);
    
    // partial derivative of loss dL/dprob
    Matrix dLoss=this->loss_derivative(batch.y, y)/batch_size;
    
    this->backward(dLoss);
    this->update_weights(rate, momentum, decay);
  }
}

//////////////////////////////// C++ class member functions
void Layer::update_weights(double rate, double momentum, double decay) { update_layer(*this, rate, momentum, decay); }
