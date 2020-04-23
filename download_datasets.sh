#!/bin/bash

mkdir mnist
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  -O mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz  -O mnist/t10k-labels-idx1-ubyte.gz

gunzip -f mnist/train-images-idx3-ubyte.gz
gunzip -f mnist/train-labels-idx1-ubyte.gz
gunzip -f mnist/t10k-images-idx3-ubyte.gz
gunzip -f mnist/t10k-labels-idx1-ubyte.gz

mkdir cifar
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -O cifar/cifar-10-binary.tar.gz
wget https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -O cifar/cifar-100-binary.tar.gz
tar -C cifar -xf cifar/cifar-10-binary.tar.gz
tar -C cifar -xf cifar/cifar-100-binary.tar.gz
