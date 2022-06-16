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

#define BLOCK_SIZE 32

__global__ void grad_desc_weights(double *weights, double *gradient, double learning_rate, int rows, int cols)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int i = idx_x; i < rows; i += stride_x)
    {
        for (int j = idx_y; j < cols; j += stride_y)
        {
            weights[i * cols + j] -= learning_rate * gradient[i * cols + j];
        }
    }
}

__global__ void grad_desc_bias(double *bias, double *error, double learning_rate, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < size; i += stride)
    {
        bias[idx] -= learning_rate * error[idx];
    }
}

// Taking the average of each row
__global__ void calc_avgs(double *error, double *avgs, int rows, int cols)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < rows; i += stride)
    {
        auto sum = 0.;
        for (int b = 0; b < cols; b++)
        {
            sum += error[idx * cols + b];
        }
        avgs[idx] = sum / double(cols);
    }
}

__global__ void sigmoid_derivative(double *y_truth, double *output_layer, double *output_layer_error, int rows, int cols)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int i = idx_x; i < rows; i += stride_x)
    {
        for (int j = idx_y; j < cols; j += stride_y)
        {
            double error_prime = -1 * (y_truth[i * cols + j] - output_layer[i * cols + j]);
            double sigmoid_derivative = output_layer[i * cols + j] * (1 - output_layer[i * cols + j]);
            output_layer_error[i * cols + j] = error_prime * sigmoid_derivative;
        }
    }
}

__global__ void tanh_derivative(double *output_layer_error, double *b, double *hidden_layer, double *hidden_layer_error, int output_size, int rows, int cols)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int i = idx_x; i < rows; i += stride_x)
    {
        for (int j = idx_y; j < cols; j += stride_y)
        {
            double error_prime = 0;
            for (int k = 0; k < output_size; k++)
            {
                error_prime += output_layer_error[k * cols + idx_y] * b[k * rows + idx_x]; // b: 10 * 16
            }
            double tanh_derivative = 1. - (hidden_layer[idx_x * cols + idx_y] * hidden_layer[idx_x * cols + idx_y]);
            hidden_layer_error[idx_x * cols + idx_y] = error_prime * tanh_derivative;
        }
    }
}

__global__ void set_up_y_truth(double *dataset, double *dest, int *indices, int rows, int cols)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int i = idx_x; i < rows; i += stride_x)
    {
        for (int j = idx_y; j < cols; j += stride_y)
        {
            int label = dataset[indices[j]];

            if (i == label)
            {
                dest[i * cols + j] = 1;
            }
            else
            {
                dest[i * cols + j] = 0;
            }
        }
    }
}

__global__ void sigmoid(double *layers, int num, int batch_size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num * batch_size; i += stride)
    {
        auto exp_x = exp(layers[i]);
        layers[i] = exp_x / (exp_x + 1);
    }
}

__global__ void tanh(double *layers, int num, int batch_size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num * batch_size; i += stride)
    {
        auto exp_2x = exp(2 * layers[i]);
        layers[i] = (exp_2x - 1) / (exp_2x + 1);
    }
}

// Layer is 16 by batch_size (or output_layer_size by batch_size)
__global__ void add_bias(double *layer, double *bias, int num, int batch_size)
{
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

__global__ void transpose(double *mat, double *mat_t, int rows, int cols)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    for (int i = idx_x; i < rows; i += stride_x)
    {
        for (int j = idx_y; j < cols; j += stride_y)
        {
            mat_t[j * rows + i] = mat[i * cols + j];
        }
    }
}

/*
    m - leading dimension of first matrix
    n - Shared dimension of matrices
    o - Trailing dimension of second matrix
*/
__global__ void mat_mul(double *a, double *b, double *c, int m, int n, int o)
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
__global__ void copy_data(double *dataset, double *dest, int *indices, int rows, int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && b < batch_size)
    {
        int data_index = indices[b];
        dest[i * batch_size + b] = dataset[data_index * rows + i];
    }
}

__global__ void init_bias_cuda(double *w, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index != 0)
        return;

    for (int i = 0; i < size; i++)
    {
        w[i] = 0;
    }
}

__global__ void init_weights(double *w, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    // printf("stride is: %d\n", stride);

    // if (index != 0) return;

    // auto epsilon = pow(6, 0.5) / pow(size, 0.5);

    for (int i = index; i < size; i += stride)
    {
        w[i] = (w[i] * 0.2) - 0.1;
        // w[i] = w[i] * epsilon * 2 - epsilon;
    }
}

void init_weights_cpu(double *w, int rows, int cols)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-0.1, 0.1);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            w[i * cols + j] = dist(gen);
        }
    }
}

class MNISTNeuralNetwork
{
public:
    MNISTNeuralNetwork()
    {
        cudaGetDevice(&deviceId);
        cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

        auto timer = Timer("Network init");
        mnist = MNIST();
        mnist.read_data();
        mnist.normalize();

        batch_size = 256;
        learning_rate = 0.01;

        input_layer_size = 784;
        hidden_layer_size = 64;
        output_layer_size = 10;

        // Weight matrix for second layer
        cudaMallocManaged(&a, hidden_layer_size * input_layer_size * sizeof(double));
        cudaMemPrefetchAsync(&a, hidden_layer_size * input_layer_size * sizeof(double), cudaCpuDeviceId);
        init_weights_cpu(a, hidden_layer_size, input_layer_size);

        // Weight matrix for output layer
        cudaMallocManaged(&b, output_layer_size * hidden_layer_size * sizeof(double));
        cudaMemPrefetchAsync(&b, output_layer_size * hidden_layer_size * sizeof(double), cudaCpuDeviceId);
        init_weights_cpu(b, output_layer_size, hidden_layer_size);

        // Bias for hidden layer
        cudaMallocManaged(&a_bias, hidden_layer_size * sizeof(double));
        init_bias_cuda<<<1, 1>>>(a_bias, hidden_layer_size);

        // Bias for output layer
        cudaMallocManaged(&b_bias, output_layer_size * sizeof(double));
        init_bias_cuda<<<1, 1>>>(b_bias, output_layer_size);

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

        timer.stop();
    }

    enum Dataset {train, test};

    void learn(int num_of_epochs)
    {
        random_device rd;
        mt19937 g(rd());

        vector<int> indices;
        for (int i = 0; i < SIZE_OF_TRAIN; i++)
            indices.push_back(i);

        for (int i = 0; i < num_of_epochs; i++)
        {
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

                copy_data<<<dim3(calc_dim3(input_layer_size), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(mnist.x_train, input_layer, batch_indices, input_layer_size, batch_size);

                forward();

                set_up_y_truth<<<dim3(calc_dim3(output_layer_size), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(mnist.y_train, y_truth, batch_indices, output_layer_size, batch_size);

                back_prop();
                adjust_weights();

                cudaDeviceSynchronize();
            }
            // start_timer();
            printf("| Train accuracy: %f%% ", calc_accuracy(train));
            // end_timer();
            // start_timer();
            printf("| Test accuracy: %f%%\n", calc_accuracy(test));
            // end_timer();
        }
    }

    void forward()
    {
        int grid_col_dims = (batch_size > 512) ? 512 : batch_size;
        // a: 16 * input_layer_size, input_layer: input_layer_size * batch_size
        // start_time();
        mat_mul<<<dim3(calc_dim3(hidden_layer_size), calc_dim3(grid_col_dims)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(a, input_layer, hidden_layer, hidden_layer_size, input_layer_size, batch_size);
        add_bias<<<dim3(calc_dim3(hidden_layer_size), calc_dim3(grid_col_dims)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(hidden_layer, a_bias, hidden_layer_size, batch_size);
        tanh<<<calc_dim3(hidden_layer_size * grid_col_dims), BLOCK_SIZE>>>(hidden_layer, hidden_layer_size, batch_size);

        // b: output_layer_size * 16, hidden_layer: 16 * 1
        mat_mul<<<dim3(calc_dim3(output_layer_size), calc_dim3(grid_col_dims)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(b, hidden_layer, output_layer, output_layer_size, hidden_layer_size, batch_size);
        add_bias<<<dim3(calc_dim3(output_layer_size), calc_dim3(grid_col_dims)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(output_layer, b_bias, output_layer_size, batch_size);
        sigmoid<<<calc_dim3(output_layer_size * grid_col_dims), BLOCK_SIZE>>>(output_layer, output_layer_size, batch_size);
        // end_time();
    }

    int calc_dim3(int num)
    {
        return (num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    void back_prop()
    {
        sigmoid_derivative<<<dim3(calc_dim3(output_layer_size), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(y_truth, output_layer, output_layer_error, output_layer_size, batch_size);
        tanh_derivative<<<dim3(calc_dim3(hidden_layer_size), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(output_layer_error, b, hidden_layer, hidden_layer_error, output_layer_size, hidden_layer_size, batch_size);

        calc_avgs<<<calc_dim3(output_layer_size), calc_dim3(batch_size), BLOCK_SIZE>>>(output_layer_error, output_layer_error_avgs, output_layer_size, batch_size);
        calc_avgs<<<calc_dim3(hidden_layer_size), calc_dim3(batch_size), BLOCK_SIZE>>>(hidden_layer_error, hidden_layer_error_avgs, hidden_layer_size, batch_size);
    }

    void adjust_weights()
    {
        transpose<<<dim3(calc_dim3(input_layer_size), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(input_layer, input_layer_t, input_layer_size, batch_size);
        grad_desc_bias<<<calc_dim3(hidden_layer_size), BLOCK_SIZE>>>(a_bias, hidden_layer_error_avgs, learning_rate, hidden_layer_size);
        mat_mul<<<dim3(calc_dim3(hidden_layer_size), calc_dim3(input_layer_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(hidden_layer_error, input_layer_t, gradient_a, hidden_layer_size, batch_size, input_layer_size);
        grad_desc_weights<<<dim3(calc_dim3(hidden_layer_size), calc_dim3(input_layer_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(a, gradient_a, learning_rate, hidden_layer_size, input_layer_size);

        transpose<<<dim3(calc_dim3(hidden_layer_size), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(hidden_layer, hidden_layer_t, hidden_layer_size, batch_size);
        grad_desc_bias<<<calc_dim3(output_layer_size), BLOCK_SIZE>>>(b_bias, output_layer_error_avgs, learning_rate, output_layer_size);
        mat_mul<<<dim3(calc_dim3(output_layer_size), calc_dim3(hidden_layer_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(output_layer_error, hidden_layer_t, gradient_b, output_layer_size, batch_size, hidden_layer_size);
        grad_desc_weights<<<dim3(calc_dim3(output_layer_size), calc_dim3(hidden_layer_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(b, gradient_b, learning_rate, output_layer_size, hidden_layer_size);
    }

    int predict(double *input_layer)
    {
        forward();

        auto prediction = 0;
        double max_value = 0;
        for (int i = 0; i < output_layer_size; i++)
        {
            if (output_layer[i] > max_value)
            {
                prediction = i;
                max_value = output_layer[i];
            }
        }

        return prediction;
    }

    float calc_accuracy(Dataset data)
    {
        int num_correct = 0;
        int num_counted = 0;

        int data_count = (data == train) ? SIZE_OF_TRAIN : SIZE_OF_TEST;        

        for (int i = 0; i < data_count - batch_size + 1; i += batch_size)
        {
            for (int j = 0; j < batch_size; j++)
            {
                batch_indices[j] = i + j;
            }

            if (data == train)
            {
                copy_data<<<dim3(calc_dim3(input_layer_size), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(mnist.x_train, input_layer, batch_indices, input_layer_size, batch_size);
            } else {
                copy_data<<<dim3(calc_dim3(input_layer_size), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(mnist.x_test, input_layer, batch_indices, input_layer_size, batch_size);
            }
            //cudaDeviceSynchronize();

            forward();
            cudaDeviceSynchronize();

            for (int j = 0; j < batch_size; j++)
            {
                auto prediction = 0;
                double max_value = 0;
                for (int i = 0; i < output_layer_size; i++)
                {
                    if (output_layer[i * batch_size + j] > max_value)
                    {
                        prediction = i;
                        max_value = output_layer[i * batch_size + j];
                    }
                }

                if ((data == train && prediction == mnist.y_train[num_counted]) || (data == test && prediction == mnist.y_test[num_counted]))
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
    

    double *a; // Weight matrix for hidden layer
    double *b; // Weight matrix for output layer

    double *a_bias;
    double *b_bias;

    double *input_layer;
    double *hidden_layer;
    double *output_layer;

    double *hidden_layer_error;
    double *output_layer_error;

    double *hidden_layer_error_avgs;
    double *output_layer_error_avgs;

    int *batch_indices;

    double *y_truth;

    double learning_rate;

    int input_layer_size;
    int hidden_layer_size;
    int output_layer_size;
    int batch_size;

    double *input_layer_t;
    double *hidden_layer_t;
    double *gradient_a;
    double *gradient_b;

    int deviceId;
    int numberOfSMs;

    
};