#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include "common.h"
using namespace std;

Common::PerformanceTimer& timer();

struct Params {
    vector<int> layer_sizes; // layer_sizes[i] = # of perceptrons in layer i
    int layer_count;         // # of total layers in MLP
    int input_size;          // dimension of input vector
    int output_size;         // dimension of output vector = layer size of last layer.
    int batch_size;          // batch size - number of inputs to process at once.
};

class MLP {
public:
    MLP(int n, vector<int> layers);
    MLP(int n, int batch_size, vector<int> layers) {
        MLP(n, layers);
        params.batch_size = batch_size;
    }

    void load_weights(string path);
    
    // virtual double* forward(double *data) = 0;                       // Single input
    // virtual double* forward(double *data, int num_data) = 0;         // Multiple input
    // virtual double* forward(vector<double*> data, int num_data) = 0; // Multiple input
    
    static void match(double *res, double *ans, int output_size, int &val); // Single match
    static void match(double *res, vector<double*> &ans, int output_size, int num_data, int &val); // Multiple match

    int set_batch_size(int new_batch_size);
    int get_batch_size();

protected:
    Params params;  // MLP Parameters

    double *data;   // 
    double *y;
    double *exp_data;

    vector<double *> w; // Weights of Perceptron Layer
    vector<double *> b;
    vector<double *> z; // z = w * x + b
    vector<double *> a; //
};

