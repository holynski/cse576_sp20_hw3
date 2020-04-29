#pragma once

#include <cmath>
#include <algorithm>

#include "matrix.h"
#include "neural.h"

Matrix forward_linear(const Matrix &matrix);
Matrix backward_linear(const Matrix &out, const Matrix &prev_grad);
Matrix forward_logistic(const Matrix &matrix);
Matrix backward_logistic(const Matrix &out, const Matrix &prev_grad);
Matrix forward_tanh(const Matrix &matrix);
Matrix backward_tanh(const Matrix &out, const Matrix &prev_grad);
Matrix forward_relu(const Matrix &matrix);
Matrix backward_relu(const Matrix &out, const Matrix &prev_grad);
Matrix forward_lrelu(const Matrix &matrix);
Matrix backward_lrelu(const Matrix &out, const Matrix &prev_grad);
Matrix forward_softmax(const Matrix &matrix);
Matrix softmax_jacobian(const Matrix &out_row);
Matrix backward_softmax(const Matrix &out, const Matrix &prev_grad);
Matrix forward_activate_matrix(const Matrix &matrix, Activation a);
Matrix backward_activate_matrix(const Matrix &out, const Matrix &grad, Activation a);
