#include <stdlib.h>
#include <math.h>
#include <random>
#include <cmath>

#include <fstream>
#include <string>
#include <sstream>

using namespace std;

#define SIZE_OF_TRAIN 60000
#define SIZE_OF_TEST 10000

class MNIST
{
public:
    MNIST()
    {
        double train_mean = 0;
        double train_std = 0;
    }

    void read_data() {
        x_train = (double*) malloc(SIZE_OF_TRAIN * 784 * sizeof(double));
        y_train = (double*) malloc(SIZE_OF_TRAIN * sizeof(double));

        x_test = (double*) malloc(SIZE_OF_TEST * 784 * sizeof(double));
        y_test = (double*) malloc(SIZE_OF_TEST * sizeof(double));

        read_x_data("../data/x_train.csv", x_train);
        read_y_data("../data/y_train.csv", y_train);

        read_x_data("../data/x_test.csv", x_test);
        read_y_data("../data/y_test.csv", y_test);
    }

    void read_x_data(string file_name, double* var) {
        printf("Reading %s\n", file_name.c_str());
        ifstream file(file_name);
        string line = "";
        
        int counter = 0;
        getline(file, line); // skip first line
        while(getline(file, line)) {
            stringstream input_line(line);
            
            string tempString = "";
            getline(input_line, tempString, ','); // skip first column
            while(getline(input_line, tempString, ',')) {
                var[counter] = (double) atof(tempString.c_str());
                counter++;
            }
            line = "";
        }
    }

    void read_y_data(string file_name, double* var) {
        printf("Reading %s\n", file_name.c_str());
        ifstream file(file_name);
        string line = "";
        
        int counter = 0;
        getline(file, line); // skip first line
        while(getline(file, line)) {
            stringstream input_line(line);
            
            string tempString = "";
            getline(input_line, tempString, ','); // skip first column
            
            while(getline(input_line, tempString, ',')) {
                var[counter] = (double) atof(tempString.c_str());
                counter++;
            }
            line = "";
        }
    }

    void normalize() {
        for (int i = 0; i < 784 * SIZE_OF_TRAIN; i++)
        {
            train_mean += x_train[i];
        }
        train_mean /= 784 * SIZE_OF_TRAIN;

        for (int i = 0; i < 784 * SIZE_OF_TRAIN; i++)
        {
            train_std += pow(x_train[i] - train_mean, 2);
        }
        train_std /= (784 * SIZE_OF_TRAIN);
        train_std = sqrt(train_std);
        
        // Standardizing the values
        for (int i = 0; i < 784 * SIZE_OF_TRAIN; i++)
        {
            x_train[i] = (x_train[i] - train_mean) / train_std;
        }

        for (int i = 0; i < 784 * SIZE_OF_TEST; i++)
        {
            x_test[i] = (x_test[i] - train_mean) / train_std;
        }
        
    }

public:
    double *x_train;
    double *y_train;
    double *x_test;
    double *y_test;

    double train_mean;
    double train_std;
};
