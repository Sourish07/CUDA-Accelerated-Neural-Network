#include <stdlib.h>
#include <math.h>
#include <random>
#include <algorithm>
#include <chrono>
#include "mnist.h"

using namespace std;

#define SIZE_OF_TRAIN 60000
#define SIZE_OF_TEST 10000

// 784 * 16 * 10
class NeuralNetwork
{
public:
    NeuralNetwork(int b_s)
    {
        mnist = MNIST();
        mnist.read_data();
        mnist.normalize();

        batch_size = 1;

        // Weight matrix for second layer
        a = (double *)malloc(16 * 784 * sizeof(double));
        init_weights(a, 16, 784);

        a_bias = (double *)malloc(16 * sizeof(double));
        init_bias(a_bias, 16);
        a_rows = 784;
        a_cols = 16;

        // Weight matrix for output layer
        b = (double *)malloc(10 * 16 * sizeof(double));
        init_weights(b, 10, 16);

        b_bias = (double *)malloc(10 * sizeof(double));
        init_bias(b_bias, 10);

        b_rows = 16;
        b_cols = 10;

        hidden_layer = (double *)malloc(16 * batch_size * sizeof(double));
        hidden_layer_error = (double *)malloc(16 * batch_size * sizeof(double));

        output_layer = (double *)malloc(10 * batch_size * sizeof(double));
        output_layer_error = (double *)malloc(10 * batch_size * sizeof(double));

        learning_rate = 0.01;
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

            for (int j = 0; j < indices.size(); j++)
            {
                auto start = std::chrono::high_resolution_clock::now();
                int index = indices[j];
                if (j % 10000 == 0)
                {
                    printf("Training example: %d\n", j);
                }

                double input_layer[784];
                for (int k = 0; k < 784; k++)
                {
                    input_layer[k] = 0;
                }

                for (int k = 0; k < 784; k++)
                {
                    input_layer[k] = mnist.x_train[index * 784 + k];
                }
                forward(input_layer);

                double y_truth[10];
                for (int k = 0; k < 10; k++)
                {
                    y_truth[k] = 0;
                }

                int label = mnist.y_train[index];
                y_truth[label] = 1;

                back_prop(y_truth);
                adjust_weights(input_layer);

                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                //std::cout << "CPU-func: Time took: " << duration.count() << " microseconds." << std::endl;
            }
            calc_train_accuracy();
            calc_test_accuracy();
        }
    }

    void forward(double *input_layer)
    {
        // a: 16 * 784, input_layer: 784 * 1
        auto start = std::chrono::high_resolution_clock::now();
        mat_mul(a, input_layer, hidden_layer, 16, 784, batch_size);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        //std::cout << "CPU-mat_mul: Time took: " << duration.count() << " microseconds." << std::endl;

        for (int i = 0; i < 16; i++)
        {
            hidden_layer[i] += a_bias[i];
        }

        tanh(hidden_layer, 16);

        // b: 10 * 16, hidden_layer: 16 * 1
        mat_mul(b, hidden_layer, output_layer, 10, 16, batch_size);
        for (int i = 0; i < 10; i++)
        {
            output_layer[i] += b_bias[i];
        }

        sigmoid(output_layer, 10);
    }

    void back_prop(double *y_truth)
    {
        for (int i = 0; i < 10; i++)
        {
            auto error_prime = -1 * (y_truth[i] - output_layer[i]);
            auto sigmoid_derivative = output_layer[i] * (1 - output_layer[i]);
            output_layer_error[i] = error_prime * sigmoid_derivative;
        }

        for (int i = 0; i < 16; i++)
        {
            double error_prime = 0;
            for (int j = 0; j < 10; j++)
            {
                error_prime += output_layer_error[j] * b[j * 16 + i]; // b: 10 * 16
            }
            double tanh_derivative = 1. - (hidden_layer[i] * hidden_layer[i]);
            hidden_layer_error[i] = error_prime * tanh_derivative;
        }
    }

    void adjust_weights(double *x)
    {
        for (int i = 0; i < 16; i++)
        {
            a_bias[i] -= learning_rate * hidden_layer_error[i];
        }

        // a: 16 * 784
        for (int rows = 0; rows < 16; rows++)
        {
            for (int cols = 0; cols < 784; cols++)
            {
                a[rows * 784 + cols] -= x[cols] * learning_rate * hidden_layer_error[rows];
            }
        }

        for (int i = 0; i < 10; i++)
        {
            b_bias[i] -= learning_rate * output_layer_error[i];
        }

        // b: 10 * 16
        for (int rows = 0; rows < 10; rows++)
        {
            for (int cols = 0; cols < 16; cols++)
            {
                b[rows * 16 + cols] -= hidden_layer[cols] * learning_rate * output_layer_error[rows];
            }
        }
    }

    int predict(double *input_layer)
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

    void init_weights(double *w, int rows, int cols)
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

    void init_bias(double *b, int size)
    {
        for (int i = 0; i < size; i++)
        {
            b[i] = 0;
        }
    }

    /*
        m - leading dimension of first matrix
        n - Shared dimension of matrices
        o - Trailing dimension of second matrix
    */
    void mat_mul(double *a, double *b, double *c, int m, int n, int o)
    {
        
        // First row of first matrix
        for (int i = 0; i < m; i++)
        {
            // First column of second matrix
            for (int j = 0; j < o; j++)
            {
                
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += a[i * n + k] * b[k * o + j];
                }
                // c[i * m + j] = sum;
                c[j * o + i] = sum;
            }
        }
    }

    void tanh(double *a, int size)
    {
        for (int i = 0; i < size; i++)
        {
            auto exp_2x = exp(2 * a[i]);
            a[i] = (exp_2x - 1) / (exp_2x + 1);
        }
    }

    void sigmoid(double *a, int size)
    {
        for (int i = 0; i < size; i++)
        {
            auto exp_x = exp(a[i]);
            a[i] = exp_x / (exp_x + 1);
        }
    }

    void calc_train_accuracy()
    {
        int num_correct = 0;
        for (int i = 0; i < SIZE_OF_TRAIN; i++)
        {
            double test_data[784];
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
        for (int i = 0; i < SIZE_OF_TEST; i++)
        {
            double test_data[784];
            for (int k = 0; k < 784; k++)
            {
                test_data[k] = 0;
            }
            for (int j = 0; j < 784; j++)
            {
                test_data[j] = mnist.x_test[i * 784 + j];
            }

            int true_label = (int)mnist.y_test[i];
            int predicted_label = predict(test_data);

            if (predicted_label == true_label)
            {
                num_correct++;
            }
        }

        double percentage = num_correct / (double)SIZE_OF_TEST;
        printf("Test accuracy: %f\n", percentage);
    }

    void debug(double *a, int size)
    {
        double v[size];
        for (int i = 0; i < size; i++)
        {
            v[i] = a[i];
        }
        int x = 0;
    }

public:
    MNIST mnist;

    double *a;
    double *b;

    double *a_bias;
    double *b_bias;

    double *hidden_layer;
    double *output_layer;

    double *hidden_layer_error;
    double *output_layer_error;

    double learning_rate;

    int a_rows;
    int a_cols;

    int b_rows;
    int b_cols;

    int batch_size;
};