#include <stdlib.h>
#include <math.h>
#include <random>
#include <algorithm>


#include <cublas_v2.h>
#include <curand.h>

#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "curand.lib")

#include "mnist.cuh"

using namespace std;

#define SIZE_OF_TRAIN 60000
#define SIZE_OF_TEST 10000

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
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (index < num * batch_size)
    {
        auto exp_x = exp(layers[index]);
        layers[index] = exp_x / (exp_x + 1);
    }
}

__global__ 
void tan_h(double* layers, int num, int batch_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (index < num * batch_size)
    {
        auto exp_2x = exp(2 * layers[index]);
        layers[index] = (exp_2x - 1) / (exp_2x + 1);
    }
}

// Layer is 16 by batch_size (or 10 by batch_size)
__global__ 
void add_bias(double* layer, double* bias, int num, int batch_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num && col < batch_size) {
        layer[num * batch_size + col] += bias[num];
    }
}

__global__
void transpose(double* mat, double* mat_t, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
    {
        mat_t[col * rows + row] = mat[row * cols + col];
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < o)
    {
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * o + col];
        }
        c[row * o + col] = sum;
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
        hidden_layer_size = 16;

        curandGenerator_t prng;
        curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

        // Weight matrix for second layer
        cudaMallocManaged(&a, 16 * 784 * sizeof(double));
        curandGenerateUniformDouble(prng, a, 16 * 784);
        init_weights<<<1, 1>>>(a, 16 * 784);

        cudaMallocManaged(&a_bias, 16 * sizeof(double));
        init_bias_cuda<<<1, 1>>>(a_bias, 16);
        

        // Weight matrix for output layer
        cudaMallocManaged(&b, 10 * 16 * sizeof(double));
        curandGenerateUniformDouble(prng, b, 10 * 16);
        init_weights << <1, 1 >> > (b, 10 * 16);

        cudaMallocManaged(&b_bias, 10 * sizeof(double));
        init_bias_cuda << <1, 1 >> > (b_bias, 10);
        cudaDeviceSynchronize();

        cudaMallocManaged(&input_layer, 784 * batch_size * sizeof(double));

        cudaMallocManaged(&hidden_layer, 16 * batch_size * sizeof(double));
        cudaMallocManaged(&hidden_layer_error, 16 * batch_size * sizeof(double));
        cudaMallocManaged(&hidden_layer_error_avgs, 16 * sizeof(double));

        cudaMallocManaged(&output_layer, 10 * batch_size * sizeof(double));
        cudaMallocManaged(&output_layer_error, 10 * batch_size * sizeof(double));
        cudaMallocManaged(&output_layer_error_avgs, 10 * sizeof(double));

        cudaMallocManaged(&batch_indices, batch_size * sizeof(int));

        cudaMallocManaged(&y_truth, 10 * batch_size * sizeof(double));

        cudaMallocManaged(&input_layer_t, batch_size * 784 * sizeof(double));
        cudaMallocManaged(&hidden_layer_t, batch_size * 16 * sizeof(double));
        cudaMallocManaged(&gradient_a, 16 * 784 * sizeof(double));
        cudaMallocManaged(&gradient_b, 10 * 16 * sizeof(double));

        cudaGetDevice(&deviceId);
        cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    }

    

    void learn(int num_of_epochs)
    {
        random_device rd;
        mt19937 g(rd());

        vector<int> indices;
        for (int j = 0; j < SIZE_OF_TRAIN; j++)
            indices.push_back(j);

        for (int i = 0; i < num_of_epochs; i++)
        {
            printf("Epoch: %d\n", i);
            shuffle(indices.begin(), indices.end(), g);

            for (int j = 0; j < indices.size() - batch_size; j += batch_size)
            {
                if (j % (10 * batch_size) == 0)
                {
                    printf("Training example: %d\n", j);
                }

                cudaMemPrefetchAsync(batch_indices, batch_size * sizeof(int), cudaCpuDeviceId);
                for (int k = 0; k < batch_size; k++)
                {
                    batch_indices[k] = indices[j + k];
                }

                cudaMemPrefetchAsync(batch_indices, batch_size * sizeof(int), deviceId);

                copy_data << < dim3(calc_dim3(784), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (mnist.x_train, input_layer, batch_indices, batch_size);
                
                forward(input_layer);

                set_up_y_truth << <dim3(calc_dim3(10), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (mnist.y_train, y_truth, batch_indices, batch_size);

                back_prop();
                adjust_weights();

                cudaDeviceSynchronize();
            }
            //calc_train_accuracy();
            calc_test_accuracy();
        }
    }

    void forward(double* input_layer)
    {
        // a: 16 * 784, input_layer: 784 * batch_size
        //start_time();
        mat_mul<<<dim3(calc_dim3(16), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> >(a, input_layer, hidden_layer, 16, 784, batch_size);
        add_bias << <dim3(calc_dim3(16), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (hidden_layer, a_bias, 16, batch_size);
        tan_h<<<4096 / 32, BLOCK_SIZE >> >(hidden_layer, 16, batch_size);

        // b: 10 * 16, hidden_layer: 16 * 1
        mat_mul << <dim3(calc_dim3(10), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (b, hidden_layer, output_layer, 10, 16, batch_size);
        add_bias << <dim3(calc_dim3(10), calc_dim3(batch_size)), dim3(BLOCK_SIZE, BLOCK_SIZE) >> > (output_layer, b_bias, 10, batch_size);
        sigmoid<<<((16 * batch_size + 32 - 1) / 32), BLOCK_SIZE >>>(output_layer, 10, batch_size);//((10 * batch_size + 32 - 1) / 32) * 32
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
        forward(input_layer);

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

    void calc_train_accuracy()
    {
        int num_correct = 0;
        for (int i = 0; i < SIZE_OF_TRAIN; i++)
        {
            printf("%d\n", i);
            double *test_data;
            cudaMallocManaged(&test_data, 784 * sizeof(double));

            for (int k = 0; k < 784; k++)
            {
                test_data[k] = 0;
            }
            for (int j = 0; j < 784; j++)
            {
                test_data[j] = mnist.x_train[i * 784 + j];
            }

            int true_label = (int)mnist.y_train[i];
            int predicted_label = predict(test_data);

            if (predicted_label == true_label)
            {
                num_correct++;
            }
        }
        double percentage = num_correct / (double)SIZE_OF_TRAIN;
        printf("Train accuracy: %f\n", percentage);
    }

    void calc_test_accuracy()
    {
        int num_correct = 0;
        int num_counted = 0;
        for (int i = 0; i < SIZE_OF_TEST - batch_size; i += batch_size)
        {
            printf("%d\n", i);
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

            forward(input_layer);
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

        double percentage = num_correct / (double)num_counted;
        printf("Test accuracy: %f\n", percentage);
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

    int hidden_layer_size;
    int batch_size;
    
    double* input_layer_t;
    double* hidden_layer_t;
    double* gradient_a;
    double* gradient_b;

    int deviceId;
    int numberOfSMs;

    
};