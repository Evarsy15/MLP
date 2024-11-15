#include "mlp.h"

Common::PerformanceTimer& timer() {
    static Common::PerformanceTimer timer;
    return timer;
}

void fill_rand(double *A, int size, double std) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, std);
    for (int i=0; i<size; i++) {
        A[i] = distribution(generator);
    }
}

MLP::MLP(int n, vector<int> layers) {
    params.layer_count = layers.size();
    params.input_size  = n;
    params.output_size = layers[params.layer_count - 1];
    params.layer_sizes = layers;

    data = (double*)(malloc(n * sizeof(double)));
    y = (double*)(malloc(n * sizeof(double)));
    exp_data = (double*)(malloc(params.output_size * sizeof(double)));
 
    // add input layer to front
    layers.insert(layers.begin(), n);
    double *layer_w, *layer_b, *layer_z, *layer_a;
    for (int i = 0; i < params.layer_count; i++) {
        layer_w = (double*)(malloc(layers[i] * layers[i + 1] * sizeof(double)));
        layer_b = (double*)(malloc(layers[i+1] * sizeof(double)));

        // initilize w, b using gaussian distribution
        fill_rand(layer_w, layers[i] * layers[i + 1], 2.0 / (layers[i])); // uniform random initilization
        for (int j=0; j<layers[i + 1]; j++){
            layer_b[j] = 0.1;
        }
        w.push_back(layer_w);
        b.push_back(layer_b);

        // intermediate results arrays
        layer_z = (double*)(malloc(layers[i + 1] * sizeof(double)));
        layer_a = (double*)(malloc(layers[i + 1] * sizeof(double)));
        z.push_back(layer_z);
        a.push_back(layer_a);
    }
}

void MLP::load_weights(string path) {
    std::ifstream infile(path);
    if (!infile.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        return;
    }

    std::string line;
    for (int i = 0; i < params.layer_count; i++) {
        int lmi;
        if (i != 0)
            lmi = params.layer_sizes[i - 1];
        else
            lmi = params.input_size;

        int n = params.layer_sizes[i] * lmi;
        double *temp_weights = new double[n];

        // Find the weight header for this layer
        std::getline(infile, line);  // W[i] header
        std::getline(infile, line);  // "-----" separator

        // Read weights
        for (int j = 0; j < n && std::getline(infile, line); j++) {
            temp_weights[j] = std::stod(line);
        }
        memcpy(w[i], temp_weights, n * sizeof(double));
        delete[] temp_weights;
    }

    for (int i = 0; i < params.layer_count; i++) {
        int n = params.layer_sizes[i];
        double *temp_biases = new double[n];

        // Find the bias header for this layer
        std::getline(infile, line);  // b[i] header
        std::getline(infile, line);  // "-----" separator

        // Read biases
        for (int j = 0; j < n && std::getline(infile, line); j++) {
            temp_biases[j] = std::stod(line);
        }
        memcpy(b[i], temp_biases, n * sizeof(double));
        delete[] temp_biases;
    }

    infile.close();
}

static void MLP::match(double *res, double *ans, int output_size, int &val) {
    int pos1 = distance(res, max_element(res, res + output_size));
    int pos2 = distance(ans, max_element(ans, ans + output_size));
    val += (pos1 == pos2);
}

static void MLP::match(double *res, vector<double*> &ans, int output_size, int num_data, int &val) {
    for (int i = 0; i < num_data; i++)
        match(res + (i * output_size), ans[i], output_size, val);
}

int MLP::set_batch_size(int new_batch_size) {
    int old_batch_size = this->params.batch_size;
    this->params.batch_size = new_batch_size;
    return old_batch_size;
}

int MLP::get_batch_size() {
    return this->params.batch_size;
}