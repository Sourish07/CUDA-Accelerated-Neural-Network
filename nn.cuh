#include <stdlib.h>
#include <math.h>
#include <random>
#include <algorithm>


#include <cublas_v2.h>
#include <curand.h>

#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "curand.lib")

#include "mnist.cuh"
#include "timer.h"

using namespace std;

// #define SIZE_OF_TRAIN 60000
// #define SIZE_OF_TEST 10000

#define BLOCK_SIZE 32


__global__ 
void grad_desc_weights(double* weights, double* gradient, double learning_rate, int r, int c, int batch_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < r && col < c)
    {
        weights[row * c + col] -= learning_rate * gradient[row * c + col];
    }    
}

__global__ 
void grad_desc_bias(double* bias, double* error, double learning_rate, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size)
    {
        bias[index] -= learning_rate * error[index];
    }
}

__global__
void calc_avgs(double* error, double* avgs, int num, int batch_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < num)
    {
        auto sum = 0.;
        for (int b = 0; b < batch_size; b++)
        {
            sum += error[index * batch_size + b];
        }
        avgs[index] = sum / double(batch_size);
    }
}

__global__ void sigmoid_derivative(double* y_truth, double* output_layer, double* output_layer_error, int batch_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 10 && y < batch_size) {
        double error_prime = -1 * (y_truth[x * batch_size + y] - output_layer[x * batch_size + y]);
        double sigmoid_derivative = output_layer[x * batch_size + y] * (1 - output_layer[x * batch_size + y]);
        output_layer_error[x * batch_size + y] = error_prime * sigmoid_derivative;
    }

}

__global__ void tanh_derivative(double* output_layer_error, double* b, double* hidden_layer, double* hidden_layer_error, int batch_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 16 && y < batch_size)
    {
        double error_prime = 0;
        for (int j = 0; j < 10; j++)
        {
            error_prime += output_layer_error[j * batch_size + y] * b[j * 16 + x]; // b: 10 * 16
        }
        double tanh_derivative = 1. - (hidden_layer[x * batch_size + y] * hidden_layer[x * batch_size + y]);
        hidden_layer_error[x * batch_size + y] = error_prime * tanh_derivative;
    }
}

__global__
void set_up_y_truth(double* dataset, double* dest, int* indices, int batch_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 10 && y < batch_size)
    {
        int label = dataset[indices[y]];

        if (x == label) { dest[x * batch_size + y] = 1; }
        else { dest[x * batch_size + y] = 0; }
    }
}

__global__
void sigmoid(double* layers, int num, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < num * batch_size; i += stride)    {
        auto exp_x = exp(layers[i]);
        layers[i] = exp_x / (exp_x + 1);
    }
}

__global__ 
void tanh(double* layers, int num, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < num * batch_size; i += stride)    {
        auto exp_2x = exp(2 * layers[i]);
        layers[i] = (exp_2x - 1) / (exp_2x + 1);
    }
}

// Layer is 16 by batch_size (or 10 by batch_size)
__global__ 
void add_bias(double* layer, double* bias, int num, int batch_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int row = x; row < num; row += stride_x)
    {
        for (int col = y; col < batch_size; col += stride_y)
        {
            layer[row * batch_size + col] += bias[row];
        }
    }
}

__global__
void transpose(double* mat, double* mat_t, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int row = x; row < rows; row += stride_x)
    {
        for (int col = y; col < cols; col += stride_y) {
            mat_t[col * rows + row] = mat[row * cols + col];
        }
    }
}

/*
    m - leading dimension of first matrix
    n - Shared dimension of matrices
    o - Trailing dimension of second matrix
*/
__global__
void mat_mul(double* a, double* b, double* c, int m, int n, int o)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int row = x; row < m; row += stride_x)
    {
        for (int col = y; col < o; col += stride_y)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += a[row * n + i] * b[i * o + col];
            }
            c[row * o + col] = sum;
        }
    }
}


// Each column is a data point
__global__ 
void copy_data(double* dataset, double* dest, int* indices, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 784 && b < batch_size) {
        int data_index = indices[b];
        dest[i * batch_size + b] = dataset[data_index * 784 + i];
    }    
}

__global__ 
void init_bias_cuda(double* w, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index != 0) return;
    
    for (int i = 0; i < size; i++) {
        w[i] = 0;
    }
}

__global__ 
void init_weights(double* w, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index != 0) return;

    //auto epsilon = pow(6, 0.5) / pow(size, 0.5);

    for (int i = 0; i < size; i++)
    {
        w[i] = (w[i] * 0.2) - 0.1;
        //w[i] = w[i] * epsilon * 2 - epsilon;
    }
}

// 784 * 16 * 10
class MNISTNeuralNetwork
{
public:
    MNISTNeuralNetwork()
    {
        mnist = MNIST();
        mnist.read_data();
        mnist.normalize();

        batch_size = 256;
        learning_rate = 0.01;
        input_layer_size = 784;
        hidden_layer_size = 16;
        output_layer_size = 10;

        curandGenerator_t prng;
        curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

        // Weight matrix for second layer
        cudaMallocManaged(&a, hidden_layer_size * input_layer_size * sizeof(double));
        curandGenerateUniformDouble(prng, a, hidden_layer_size * input_layer_size);
        init_weights<<<1, 1>>>(a, hidden_layer_size * input_layer_size);

        cudaMallocManaged(&a_bias, hidden_layer_size * sizeof(double));
        init_bias_cuda<<<1, 1>>>(a_bias, hidden_layer_size);
        

        // Weight matrix for output layer
        cudaMallocManaged(&b, output_layer_size * hidden_layer_size * sizeof(double));
        curandGenerateUniformDouble(prng, b, output_layer_size * hidden_layer_size);
        init_weights << <1, 1 >> > (b, output_layer_size * hidden_layer_size);

        cudaMallocManaged(&b_bias, output_layer_size * sizeof(double));
        init_bias_cuda << <1, 1 >> > (b_bias, output_layer_size);
        cudaDeviceSynchronize();

        cudaMallocManaged(&input_layer, input_layer_size * SIZE_OF_TRAIN * sizeof(double));

        cudaMallocManaged(&hidden_layer, hidden_layer_size * SIZE_OF_TRAIN * sizeof(double));
        cudaMallocManaged(&hidden_layer_error, hidden_layer_size * SIZE_OF_TRAIN * sizeof(double));
        cudaMallocManaged(&hidden_layer_error_avgs, hidden_layer_size * sizeof(double));

        cudaMallocManaged(&output_layer, output_layer_size * SIZE_OF_TRAIN * sizeof(double));
        cudaMallocManaged(&output_layer_error, output_layer_size * SIZE_OF_TRAIN * sizeof(double));
        cudaMallocManaged(&output_layer_error_avgs, output_layer_size * sizeof(double));

        cudaMallocManaged(&batch_indices, batch_size * sizeof(int));

        cudaMallocManaged(&y_truth, output_layer_size * SIZE_OF_TRAIN * sizeof(double));

        cudaMallocManaged(&input_layer_t, SIZE_OF_TRAIN * input_layer_size * sizeof(double));
        cudaMallocManaged(&hidden_layer_t, SIZE_OF_TRAIN * hidden_layer_size * sizeof(double));
        cudaMallocManaged(&gradient_a, hidden_layer_size * input_layer_size * sizeof(double));
        cudaMallocManaged(&gradient_b, output_layer_size * hidden_layer_size * sizeof(double));

        cudaGetDevice(&deviceId);
        cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    }

    

    void learn(int num_of_epochs)
    {
        random_device rd;
        mt19937 g(rd());

        vector<int> indices;
        for (int i = 0; i < SIZE_OF_TRAIN; i++)
            indices.push_back(i);

        for (int i = 0; i < num_of_epochs; i++)
        {
            //printf("Epoch: %d\n", i);
            shuffle(indices.begin(), indices.end(), g);

            for (int j = 0; j < indices.size() - batch_size + 1; j += batch_size)
            {
                
                if (j % (10 * batch_size) == 0 || indices.size() - j > batch_size)
                {
                    std::cerr << "\rEpoch: " << i << " | Progress: " << (j + batch_size) * 100 / ((indices.size() / batch_size) * batch_size) << "% " << std::flush;
                }

                cudaMemPrefetchAsync(batch_indices, batch_size * sizeof(int), cudaCpuDeviceId);
                for (int k = 0; k < batch_size; k++)
                {
                    batch_indices[k] = indices[j + k];
                }

                cudaMemPrefetchAsync(batch_indices, batch_size * sizeof(int), deviceId);

                copy_data << < dim3(calc_dim3(784), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (mnist.x_train, input_layer, batch_indices, batch_size);
                
                forward();

                set_up_y_truth << <dim3(calc_dim3(10), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (mnist.y_train, y_truth, batch_indices, batch_size);

                back_prop();
                adjust_weights();

                cudaDeviceSynchronize();
            }
            start_timer();
            printf("| Train accuracy: %f ", calc_train_accuracy());
            end_timer();
            start_timer();
            printf("| Test accuracy: %f\n", calc_test_accuracy());
            end_timer();
        }
    }

    void forward()
    {
        int yeet = 16;
        // a: 16 * 784, input_layer: 784 * batch_size
        //start_time();
        mat_mul<<<dim3(calc_dim3(hidden_layer_size), calc_dim3(yeet)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> >(a, input_layer, hidden_layer, hidden_layer_size, input_layer_size, batch_size);
        add_bias << <dim3(calc_dim3(hidden_layer_size), calc_dim3(yeet)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (hidden_layer, a_bias, hidden_layer_size, batch_size);
        tanh<<<calc_dim3(hidden_layer_size * yeet), BLOCK_SIZE >> >(hidden_layer, hidden_layer_size, batch_size);
        
        // b: 10 * 16, hidden_layer: 16 * 1
        mat_mul << <dim3(calc_dim3(10), calc_dim3(yeet)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (b, hidden_layer, output_layer, 10, 16, batch_size);
        add_bias << <dim3(calc_dim3(10), calc_dim3(yeet)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (output_layer, b_bias, 10, batch_size);
        sigmoid<<<calc_dim3(output_layer_size * yeet), BLOCK_SIZE >>>(output_layer, 10, batch_size);
        //end_time();
    }

    int calc_dim3(int num) {
        return (num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    void back_prop()
    {
        sigmoid_derivative << <dim3(calc_dim3(10), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (y_truth, output_layer, output_layer_error, batch_size);
        tanh_derivative << <dim3(calc_dim3(16), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (output_layer_error, b, hidden_layer, hidden_layer_error, batch_size);

        calc_avgs << <calc_dim3(10), calc_dim3(batch_size), BLOCK_SIZE >> > (output_layer_error, output_layer_error_avgs, 10, batch_size);
        calc_avgs << <calc_dim3(16), calc_dim3(batch_size), BLOCK_SIZE >> > (hidden_layer_error, hidden_layer_error_avgs, 16, batch_size);

    }

    void adjust_weights()
    {        
        transpose << <dim3(calc_dim3(784), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (input_layer, input_layer_t, 784, batch_size);
        transpose << <dim3(calc_dim3(16), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (hidden_layer, hidden_layer_t, 16, batch_size);

        // a: 16 * 784
        grad_desc_bias << <calc_dim3(16), BLOCK_SIZE >> > (a_bias, hidden_layer_error_avgs, learning_rate, 16);
        mat_mul << <dim3(calc_dim3(16), calc_dim3(784)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (hidden_layer_error, input_layer_t, gradient_a, 16, batch_size, 784);
        grad_desc_weights << <dim3(calc_dim3(16), calc_dim3(784)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (a, gradient_a, learning_rate, 16, 784, batch_size);
        
        // b: 10 * 16
        grad_desc_bias << <calc_dim3(10), BLOCK_SIZE >> > (b_bias, output_layer_error_avgs, learning_rate, 10);
        mat_mul << <dim3(calc_dim3(10), calc_dim3(16)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (output_layer_error, hidden_layer_t, gradient_b, 10, batch_size, 16);
        grad_desc_weights << <dim3(calc_dim3(10), calc_dim3(16)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (b, gradient_b, learning_rate, 10, 16, batch_size);
    }

    int predict(double* input_layer)
    {
        forward();

        auto prediction = 0;
        double max_value = 0;
        for (int i = 0; i < 10; i++)
        {
            if (output_layer[i] > max_value)
            {
                prediction = i;
                max_value = output_layer[i];
            }
        }

        return prediction;
    }

    float calc_train_accuracy()
    {
        
        int num_correct = 0;

        transpose<<<dim3(calc_dim3(784), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(mnist.x_train, input_layer, 784, SIZE_OF_TRAIN);
        cudaDeviceSynchronize();

        int old_batch_size = batch_size;
        batch_size = SIZE_OF_TRAIN;

        forward();
        cudaDeviceSynchronize();

        for (int j = 0; j < SIZE_OF_TRAIN; j++)
        {
            auto prediction = 0;
            double max_value = 0;
            for (int i = 0; i < 10; i++)
            {
                if (output_layer[i * batch_size + j] > max_value)
                {
                    prediction = i;
                    max_value = output_layer[i * batch_size + j];
                }
            }

            if (prediction == mnist.y_train[j])
            {
                num_correct++;
            }
        }

        batch_size = old_batch_size;

        return num_correct / (double)SIZE_OF_TRAIN;
    }

    float calc_test_accuracy()
    {
        int num_correct = 0;
        int num_counted = 0;

        for (int i = 0; i < SIZE_OF_TEST - batch_size + 1; i += batch_size)
        {
            for (int j = 0; j < batch_size; j++)
            {
                batch_indices[j] = i + j;
            }

            copy_data << <dim3(calc_dim3(784), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (mnist.x_test, input_layer, batch_indices, batch_size);
            cudaDeviceSynchronize();

            int true_labels[1000];

            for (int j = 0; j < batch_size; j++)
            {
                true_labels[j] = (int) mnist.y_test[i + j];
            }

            forward();
            cudaDeviceSynchronize();

            for (int j = 0; j < batch_size; j++)
            {
                auto prediction = 0;
                double max_value = 0;
                for (int i = 0; i < 10; i++)
                {
                    if (output_layer[i * batch_size + j] > max_value)
                    {
                        prediction = i;
                        max_value = output_layer[i * batch_size + j];
                    }
                }

                if (prediction == true_labels[j])
                {
                    num_correct++;
                }
                num_counted++;
            }
        }

        return num_correct / (float) num_counted * 100;
    }

public:
    MNIST mnist;

    double* a;
    double* b;

    double* a_bias;
    double* b_bias;

    double* input_layer;
    double* hidden_layer;
    double* output_layer;

    double* hidden_layer_error;
    double* output_layer_error;

    double* hidden_layer_error_avgs;
    double* output_layer_error_avgs;

    int* batch_indices;

    double* y_truth;

    double learning_rate;

    int input_layer_size;
    int hidden_layer_size;
    int output_layer_size;
    int batch_size;
    
    double* input_layer_t;
    double* hidden_layer_t;
    double* gradient_a;
    double* gradient_b;

    int deviceId;
    int numberOfSMs;

    // double* input_layer_train;
    // double* hidden_layer_train;
    // double* output_layer_train;

    // double* input_layer_test;
    // double* hidden_layer_test;
    // double* output_layer_test;
};