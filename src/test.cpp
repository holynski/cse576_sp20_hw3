#include "matrix.h"
#include "utils.h"
#include "activations.h"

#include <string>
#include <iostream>

using namespace std;

const static double EPS = 10e-8;

int tests_total = 0;
int tests_fail = 0;

bool matrix_within_eps(const Matrix& a, const Matrix& b, double eps) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);

  Matrix diff = (a - b).abs();
  for (int i = 0; i < a.rows; i++) {
    for (int j = 0; j < b.cols; j++) {
      if (diff(i, j) > eps) {
        printf("Difference exceeds eps at row %d, col %d (%f != %f)\n",
               i, j, a(i, j), b(i, j));
        return false;
      }
    }
  }
  return true;
}

void test_forward_linear() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix gt = load_binary("solutions/forward_linear.bin");
  Matrix output = forward_linear(a);
  TEST(matrix_within_eps(gt, output, EPS));
}

void test_forward_logistic() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix gt = load_binary("solutions/forward_logistic.bin");
  Matrix output = forward_logistic(a);
  TEST(matrix_within_eps(gt, output, EPS));
}

void test_forward_tanh() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix gt = load_binary("solutions/forward_tanh.bin");
  Matrix output = forward_tanh(a);
  TEST(matrix_within_eps(gt, output, EPS));
}

void test_forward_relu() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix gt = load_binary("solutions/forward_relu.bin");
  Matrix output = forward_relu(a);
  TEST(matrix_within_eps(gt, output, EPS));
}

void test_forward_lrelu() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix gt = load_binary("solutions/forward_lrelu.bin");
  Matrix output = forward_lrelu(a);
  TEST(matrix_within_eps(gt, output, EPS));
}

void test_forward_softmax() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix gt = load_binary("solutions/forward_softmax.bin");
  Matrix output = forward_softmax(a);
  TEST(matrix_within_eps(gt, output, EPS));
}


void test_backward_linear() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix b = load_binary("solutions/b.bin");
  Matrix gt = load_binary("solutions/backward_linear.bin");
  Matrix output = backward_linear(a, b);
  TEST(matrix_within_eps(gt, output, EPS));
}

void test_backward_logistic() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix b = load_binary("solutions/b.bin");
  Matrix gt = load_binary("solutions/backward_logistic.bin");
  Matrix output = backward_logistic(a, b);
  TEST(matrix_within_eps(gt, output, EPS));
}

void test_backward_tanh() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix b = load_binary("solutions/b.bin");
  Matrix gt = load_binary("solutions/backward_tanh.bin");
  Matrix output = backward_tanh(a, b);
  TEST(matrix_within_eps(gt, output, EPS));
}

void test_backward_relu() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix b = load_binary("solutions/b.bin");
  Matrix gt = load_binary("solutions/backward_relu.bin");
  Matrix output = backward_relu(a, b);
  TEST(matrix_within_eps(gt, output, EPS));
}

void test_backward_lrelu() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix b = load_binary("solutions/b.bin");
  Matrix gt = load_binary("solutions/backward_lrelu.bin");
  Matrix output = backward_lrelu(a, b);
  TEST(matrix_within_eps(gt, output, EPS));
}

void test_backward_softmax() {
  Matrix a = load_binary("solutions/a.bin");
  Matrix b = load_binary("solutions/b.bin");
  Matrix gt = load_binary("solutions/backward_softmax.bin");
  Matrix output = backward_softmax(a, b);
  TEST(matrix_within_eps(gt, output, EPS));
}

void run_tests() {
  test_forward_linear();
  test_forward_logistic();
  test_forward_tanh();
  test_forward_relu();
  test_forward_lrelu();
  test_forward_softmax();

  test_backward_linear();
  test_backward_logistic();
  test_backward_tanh();
  test_backward_relu();
  test_backward_lrelu();
  test_backward_softmax();

  printf("%d tests, %d passed, %d failed\n", tests_total, tests_total - tests_fail, tests_fail);
}

int main(int argc, char **argv) {
  run_tests();
  return 0;
}


