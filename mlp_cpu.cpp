#include "mlp_cpu.h"


void MLP_CPU::matmul(const double *A, const double *B, double *C, const int M, const int K, const int N) {
    for (int m=0; m<M; m++) {
        for (int n=0; n<N; n++) {
            int c_index = n * M + m;
            C[c_index] = 0.0;
            for (int k=0; k<K; k++) {
                int a_index = k * M + m;
                int b_index = n * K + k;
                C[c_index] += A[a_index] * B[b_index];
            }
        }
    }
}

void MLP_CPU::bias_addition(int n, double *idata, double *bias, double *odata) {
    for (int i=0; i<n; i++) {
        odata[i] = idata[i] + bias[i];
    }
}

void MLP_CPU::relu_activation(int n, double *idata, double *odata) {
    for (int i=0; i<n; i++) {
        odata[i] = idata[i] > 0.0 ? idata[i] : 0.0;
    }
}

void MLP_CPU::broadcast_sub(int n, double *idata, double *odata, double sub) {
    for (int i=0; i<n; i++) {
        odata[i] = idata[i] - sub;
    }
}

void MLP_CPU::elementwise_exp(int n, double *idata, double *odata) {
    for (int i=0; i<n; i++) {
        odata[i] = std::exp(idata[i]);
    }
}

void MLP_CPU::reduce_max(int n, double *idata, double *odata) {
    for (int i=0; i<n; i++) {
        *odata = *odata > idata[i] ? *odata : idata[i];
    }
}

void MLP_CPU::reduce_sum(int n, double *idata, double *odata) {
    for (int i=0; i<n; i++) {
        *odata += idata[i];
    }
}

void MLP_CPU::broadcast_div(int n, double *idata, double *odata, double exp_sum) {
    for (int i=0; i<n; i++) {
        odata[i] = idata[i] / exp_sum;
    }
}

double* MLP_CPU::forward(double *data, int n) {
    timer().startCpuTimer();
    double exp_sum;
    double out_max;
    double *res = new double[params.output_size]();

    for (int i = 0; i < params.layer_count; i++) {
        // matrix multiplication
        if (!i) { // first iteration, so a[i-1] hasn't been set yet
            matmul(w[i], data, z[i], params.layer_sizes[i], params.input_size, 1);
        }
        else {
            matmul(w[i], a[i - 1], z[i], params.layer_sizes[i], params.layer_sizes[i - 1], 1);
        }

        // bias addition
        bias_addition(params.layer_sizes[i], z[i], b[i], z[i]);
        if (i != params.layer_count - 1) {
            // relu activation
            relu_activation(params.layer_sizes[i], z[i], a[i]);
        }
        else {
            out_max = z[i][0];
            reduce_max(params.layer_sizes[i], z[i], &out_max);
            broadcast_sub(params.layer_sizes[i], z[i], z[i], out_max);
            elementwise_exp(params.layer_sizes[i], z[i], exp_data);
            exp_sum = 0;
            reduce_sum(params.layer_sizes[i], exp_data, &exp_sum);
            broadcast_div(params.layer_sizes[i], exp_data, a[i], exp_sum);
        }
    }
    memcpy(res, a[params.layer_count - 1], params.layer_sizes[params.layer_count - 1] * sizeof(double));
    timer().endCpuTimer();
    return res;
}
