#include <string>
#include <iostream>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "mlp.h"
#include "mlp_cpu.h"
#include "mlp_cuda.h"

#define classes 52
#define in_dim 15
#define epochs 400
#define inputs in_dim*in_dim

#define batch 64

using namespace std;

float total_time_forward = 0;

int image_read(string path, vector<double *> &data) {
    cv::Mat image = cv::imread(path.c_str(), cv::IMREAD_GRAYSCALE);
    if (!image.data) {
        cout << "Could not open or find the image" << std::endl;
        return 0;
    }
    data.push_back(new double[inputs]);
    cv::resize(image, image, cv::Size(in_dim, in_dim));
    for (int i = 0; i < inputs; i++)
        data[data.size() - 1][i] = (double)image.data[i];
    return 1;
}

void lable_read(string name, vector<double *> &data) {
    int value = stoi(name.substr(0, 2));
    data.push_back(new double[classes]);
    memset(data[data.size() - 1], 0, classes * sizeof(double));
    data[data.size() - 1][value - 1] = 1;
}

void read_directory(const std::string& name, vector<string>& v) {
    for (const auto& entry : std::filesystem::directory_iterator(name)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bmp") {
            v.push_back(entry.path().filename().string());
        }
    }
}

int main(int argc, char* argv[]) {
    string path = "./data-set/";
    vector<string> files;
    vector<double *> input_data;
    vector<double *> output_data;

    read_directory(path, files);
    for (auto x : files) {
        if (image_read(path + x, input_data)) {
            lable_read(x, output_data);
        }
    }

    /* 
        Run Forward-pass with CPU-Only MLP
    */
    MLP_CPU NN_CPU(inputs, {98, 65, 50, 30, 25, 40, classes});
    NN_CPU.load_weights("./weights.csv");
    int cpu_val = 0;
    for (int e = 0; e < epochs; e++) {
        total_time_forward += timer().getCpuElapsedTimeForPreviousOperation();
        for (int i = 0; i < classes; i++) {
            double* x = NN_CPU.forward(input_data[i], inputs);
            // MLP::match(x, output_data[i], classes, cpu_val);
            int pos1 = distance(x, max_element(x, x + classes));
            int pos2 = distance(output_data[i], max_element(output_data[i], output_data[i] + classes));
            val += pos1 == pos2;
            delete[] x;
        }
    }
    cout << "Inference total forward = " << total_time_forward << " ms, avg forward = " << total_time_forward / epochs << " ms" << endl;
    cout << "Accuracy : " << (float) cpu_val / (epochs * classes) * 100.0 << "%" << endl;


    /* 
        Run Forward-pass with MLP w/ CUDA
    */
    MLP_CUDA NN_CUDA(inputs, {98, 65, 50, 30, 25, 40, classes});
    NN_CUDA.set_batch_size(batch);
    NN_CUDA.allocate_gpu_memory();
    float CUDA_malloc_overhead = timer().getCpuElapsedTimeForPreviousOperation();

    // Copy MLP layers (weights and biases) into GPU & measure data transfer time.
    NN_CUDA.load_weights("./weights.csv");
    NN_CUDA.copy_mlp_into_gpu();
    float MLP_transfer_overhead = timer().getCpuElapsedTimeForPreviousOperation();

    // MLP execution
    float MLP_execution_time = 0.0f;
    int num_data = input_data.size();
    int output_size = classes;
    int gpu_val = 0; // match success counter

    for (int e = 0; e < epochs; e++) {
        double *res = NN_CUDA.forward(input_data, num_data);
        MLP_execution_time += NN_CUDA.get_total_time_forward();
        MLP::match(res, output_data, output_size, num_data, gpu_val);
        delete[] res;
    }

    cout << "Inference total forward : " << MLP_execution_time << " ms, avg forward = " << MLP_execution_time / epochs << " ms" << endl;
    cout << "Accuracy : " << (float) gpu_val / (epochs * classes) * 100.0 << "%" << endl;

    NN_CUDA.free_gpu_memory();
}