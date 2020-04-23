#include "matrix.h"
#include "neural.h"
#include "utils.h"

Data Data::random_batch(int batch_size) const {
  Data res(batch_size, X.cols, y.cols);

  for (int q1 = 0; q1 < batch_size; q1++) {
    int c1 = mt() % unsigned(X.rows);

    for (int q2 = 0; q2 < X.cols; q2++)res.X(q1, q2) = X(c1, q2);
    for (int q2 = 0; q2 < y.cols; q2++)res.y(q1, q2) = y(c1, q2);
  }

  return res;
}

bool file_exists(const std::string &file)
  {
  FILE *fn = fopen(file.c_str(), "r");
  if(fn==nullptr)return false;
  fclose(fn);
  return true;
  }

size_t file_size(const std::string &file) {
  FILE *fn = fopen(file.c_str(), "r");
  fseek(fn, 0, SEEK_END);
  size_t sz = ftell(fn);
  fclose(fn);
  return sz;
}

std::vector<unsigned char> read_file(const std::string &file) {
  if(!(file_exists(file)))
    {
    printf("Input file: \"%s\" missing. Download the dataset first\n",file.c_str());
    exit(-1);
    }
  std::vector<unsigned char> data(file_size(file));
  FILE *fn = fopen(file.c_str(), "rb");
  size_t bytes = fread(data.data(), data.size(), 1, fn);
  fclose(fn);
  return data;
};

Data read_mnist(const std::string &image_file, const std::string &label_file) {

  std::vector<unsigned char> image_data = read_file(image_file);
  std::vector<unsigned char> label_data = read_file(label_file);

  for (int q1 = 0; q1 < 4; q1++)
    for (int q2 = 0; q2 < 2; q2++)
      std::swap(image_data[q1 * 4 + q2],
                image_data[q1 * 4 + 3 - q2]);
  for (int q1 = 0; q1 < 2; q1++)
    for (int q2 = 0; q2 < 2; q2++)
      std::swap(label_data[q1 * 4 + q2],
                label_data[q1 * 4 + 3 - q2]);

  int num_images = ((int *) image_data.data())[1];

  std::vector<T> floatdata(image_data.size() - 16);
  std::vector<int> labels(label_data.size() - 8);

  //OMP5for2(8,16,image_data.size()-16,q1,{floatdata[q1-16]=(T)image_data[q1]/255.f;});
  //OMP5for2(8,8,label_data.size()-8,q1,{labels[q1-8]=(int)label_data[q1];});


  for (size_t q1 = 16; q1 < image_data.size(); q1++)floatdata[q1 - 16] = (T) image_data[q1] / T(255);
  for (size_t q1 = 8; q1 < label_data.size(); q1++)labels[q1 - 8] = (int) label_data[q1];

  int N = (int) labels.size();

  assert(N == num_images);

  Data data(N, 28 * 28, 10);

  for (int q1 = 0; q1 < N; q1++)memcpy(data.X[q1], &(floatdata[q1 * 28 * 28]), 28 * 28 * sizeof(T));
  for (int q1 = 0; q1 < N; q1++)data.y(q1, labels[q1]) = T(1);

  return data;
}

// READS CIFAR dataset. Valid combos are:
// dataset=10, labels=10   // reads CIFAR10
// dataset=100, labels=20  // reads CIFAR100 coarse labels
// dataset=100, labels=100 // reads CIFAR100 fine labels
Data read_cifar(const std::string &file, int dataset, int labels) {
  assert((dataset == 10 && labels == 10) ||
      (dataset == 100 && labels == 20) ||
      (dataset == 100 && labels == 100));

  int skip = dataset == 10 ? 3073 : 3074;
  int offset = dataset == 10 ? 1 : 2;
  int label_offset = offset - 1 - (dataset != labels);

  std::vector<unsigned char> data = read_file(file);
  int N = (int) data.size() / skip;

  Data d(N, 3072, labels);

  for (int q1 = 0; q1 < N; q1++) {
    unsigned char *ptr = &data[q1 * skip];
    d.y(q1, ptr[label_offset]) = T(1);
    ptr += offset;
    for (int q2 = 0; q2 < 3072; q2++)d.X(q1, q2) = ptr[q2] / T(255);
  }

  return d;
}
