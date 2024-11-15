#include "mlp.h"

class MLP_CPU : public MLP {
public:
    MLP_CPU(int n, vector<int> layers)
        : MLP(n, layers) { }
    MLP_CPU(int n, int batch_size, vector<int> layers) 
        : MLP(n, batch_size, layers) { }
    
    double* forward(double *data, int n);

private:
    void matmul(const double *A, const double *B, double *C, const int M, const int K, const int N);
    void bias_addition(int n, double *idata, double *bias, double *odata);
    void relu_activation(int n, double *idata, double *odata);
    void broadcast_sub(int n, double *idata, double *odata, double sub);
    void elementwise_exp(int n, double *idata, double *odata);
    void reduce_max(int n, double *idata, double *odata);
    void reduce_sum(int n, double *idata, double *odata);
    void broadcast_div(int n, double *idata, double *odata, double exp_sum);
};

/*
class Net {
public:
  Net(int n, vector<int> layers);
  double* forward(double *data, int n);
  void load_weights(string path);
private:
  double *data;
  double *y;
  double *exp_data;

  vector<double *> w;
  vector<double *> b;
  vector<double *> z; // z = w * x + b
  vector<double *> a; //
  // parameters
  Params params;
  void fill_rand(double *A, int size, double std);
  void matmul(const double *A, const double *B, double *C, const int M, const int K, const int N);
  void bias_addition(int n, double *idata, double *bias, double *odata);
  void relu_activation(int n, double *idata, double *odata);
  void broadcast_sub(int n, double *idata, double *odata, double sub);
  void elementwise_exp(int n, double *idata, double *odata);
  void reduce_max(int n, double *idata, double *odata);
  void reduce_sum(int n, double *idata, double *odata);
  void broadcast_div(int n, double *idata, double *odata, double exp_sum);
};
*/