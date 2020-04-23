#include "matrix.h"
#include "neural.h"

Dataset get_mnist(void) {
  return {read_mnist("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte"),
          read_mnist("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte")};
}

Dataset get_cifar10(void) {
  vector<Data> data;
  for (int q1 = 1; q1 <= 5; q1++)
    data.push_back(read_cifar("cifar/cifar-10-batches-bin/data_batch_" + to_string(q1) + ".bin", 10, 10));
  int n = 0;
  for (auto &d:data)n += d.X.rows;
  Data train(n, data[0].X.cols, data[0].y.cols);
  n = 0;
  for (auto &d:data)
    for (int q1 = 0; q1 < d.X.rows; q1++) {
      for (int q2 = 0; q2 < d.X.cols; q2++) {
        train.X(n, q2) = d.X(q1, q2);
      }
      for (int q2 = 0; q2 < d.y.cols; q2++) {
        train.y(n, q2) = d.y(q1, q2);
      }
      n++;
    }

  Data test = read_cifar("cifar/cifar-10-batches-bin/test_batch.bin", 10, 10);
  return {train, test};
}

Dataset get_cifar100_coarse(void) {
  return {read_cifar("cifar/cifar-100-binary/train.bin", 100, 20),
          read_cifar("cifar/cifar-100-binary/test.bin", 100, 20)};
}

Dataset get_cifar100_fine(void) {
  return {read_cifar("cifar/cifar-100-binary/train.bin", 100, 100),
          read_cifar("cifar/cifar-100-binary/test.bin", 100, 100)};
}

Model softmax_model(int inputs, int outputs) {
  return {{Layer(inputs, outputs, SOFTMAX)}, // linear layer with SOFTMAX activation
           CROSS_ENTROPY};                   // using CROSS_ENTROPY loss function
}

Model neural_net(int inputs, int outputs) {
  return {{
              Layer(inputs, 32, LOGISTIC),
              Layer(32, outputs, SOFTMAX)
          },  CROSS_ENTROPY};
}

int main(int argc, char **argv) {
  // Set the verbose flag to true to enable debug prints!
  set_verbose(true);

  printf("Loading dataset\n");
  Dataset d = get_mnist();
  //Dataset d = get_cifar10();

  double batch = 128;
  double iters = 1000;
  double rate = .01;
  double momentum = .9;
  double decay = .0;
  
  Model model = softmax_model(d.train.X.cols, d.train.y.cols);
  //Model model = neural_net(d.train.X.cols,d.train.y.cols);
  
  printf("Training model...\n");
  
  model.train(d.train, batch, iters, rate, momentum, decay);

  printf("evaluating model...\n");
  printf("training accuracy: %lf\n", model.accuracy(d.train));
  printf("test accuracy:     %lf\n", model.accuracy(d.test));

  return 0;
}
