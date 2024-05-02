#define MAX_EPOCHS 100
#define SAMPLES 200
#define VLD_SPLIT 0.3
#define NR_FEATURES 5
#define LEARNING_RT 0.03
#define RANDOM_INIT_W 1
#define INPUT_SCL_FCT 1000

#define FRAND_NRM() (((double) rand() / RAND_MAX) * 2.0 - 1.0)
#define SIGMOID(x) (1 / (1 + exp(-x)))
#define TRN_SAMPLES (SAMPLES * (1.0 - VLD_SPLIT))

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// Declarations
double dot_product(const double *a, const double *b, const int vector_size);
double vec_norm(const double *a, const int vector_size);
double cross_entropy_loss_vanilla(double y_true, double y_pred);
double cross_entropy_loss_mod(double y_true, double w_sum);
double test_model(const int *feat_index, const double *weights, const int data[][SAMPLES], const int *y, const int *order);
void shuffle_rows(int *randomOrder);

double linearClassifier(const int *feat_index, const int data[][SAMPLES], const int *y)
{
    double weights[NR_FEATURES+1]= {0};
    double gradient[NR_FEATURES+1];
    double loss = 0.0;
    double best_loss = TRN_SAMPLES;
    static int randomOrder[SAMPLES];
    int epoch = 0;
    static double best = 0.0;
    int trn_error;
    int tst_error;
    double X[NR_FEATURES+1] = {0};
    double w_sum;
    double fitness = 0.0;
    double best_fitness = 0.0;
    /* assign random or zero initial weight values depending on RANDOM_INIT_W flag */
    if (RANDOM_INIT_W) {
        for (int i = 0; i < NR_FEATURES+1; i++) {
            weights[i] = FRAND_NRM();
        }
    }
    /* shuffling rows once to split samples in training and test set*/
    shuffle_rows(randomOrder);
    /* outer loop for every epoch */
    while (epoch++ < MAX_EPOCHS) {
        for (int i = 0; i < NR_FEATURES+1; i++) {
            gradient[i] = 0;
        }
        loss = 0.0;
        trn_error = 0;
        tst_error = 0;
        /* inner loop for every training sample */
        for (int j = 0; j < TRN_SAMPLES; j++) {
            int row = randomOrder[j];
            for (int i = 0; i < NR_FEATURES; i++) {
                X[i] = (double)data[feat_index[i]][row] / INPUT_SCL_FCT;
            }
            X[NR_FEATURES] = 1.0;
            int y_true = y[row];
            w_sum = dot_product(X, weights, NR_FEATURES+1);
            double output = SIGMOID(w_sum);
            int y_pred = output > 0.5 ? 1 : -1;
            loss += cross_entropy_loss_mod(y_true, w_sum);
            for (int i = 0; i < NR_FEATURES+1; i++) {
                gradient[i] += X[i] * (output - (y_true + 1) / 2);
            }
        }

        // Update weights (once per epoch)
        for (int i = 0; i < NR_FEATURES+1; i++) {
            weights[i] -= LEARNING_RT * gradient[i];
        }

        fitness = test_model(feat_index, weights, data, y, randomOrder);
        if (fitness > best_fitness) {
            best_fitness = fitness;
        }
    }
    return best_fitness;
}

// Function to compute dot product of two vectors
double dot_product(const double *a, const double *b, const int vector_size) {
    double result = 0.0;
    for (int i = 0; i < vector_size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Function to compute norm of a vector
double vec_norm(const double *a, const int vector_size) {
    double result = 0.0;
    for (int i = 0; i < vector_size; i++) {
        result += pow(a[i], 2);
    }
    return sqrt(result);
}


// Cross entropy loss function for {0, 1} labels
double cross_entropy_loss_vanilla(double y_true, double y_pred) {
    return - (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred));
}

// Cross entropy loss function for {-1, 1} labels
double cross_entropy_loss_mod(double y_true, double w_sum) {
    return log(1 + exp(-y_true * w_sum));
}

// Function to shuffle row indices
void shuffle_rows(int *randomOrder) {

    // make list of numbers 0 - (SAMPLES - 1) in random order
    int rnd, tmp;
    for (int i = 0; i < SAMPLES; i++) {
        randomOrder[i] = i;
    }
    for (int i = SAMPLES - 1; i >= 0; i--) {
        rnd = rand() % (i + 1);
        tmp = randomOrder[rnd];
        randomOrder[rnd] = randomOrder[i];
        randomOrder[i] = tmp;
    }
}

// Test function
double test_model(const int *feat_index, const double *weights, const int data[][SAMPLES], const int *y, const int *order) {
    int fp = 0, tp = 0, tn = 0, fn = 0;
    static double best = 0;
    double error = 0;
    double X[NR_FEATURES+1] = {0};
    for (int j = TRN_SAMPLES; j < SAMPLES; j++) {
            int row = order[j];
            for (int i = 0; i < NR_FEATURES; i++) {
                X[i] = (double)data[feat_index[i]][row] / 1000;
            }
            X[NR_FEATURES] = 1.0;
            int y_true = y[row];
            double w_sum = dot_product(X, weights, NR_FEATURES+1);
            double output = SIGMOID(w_sum);
            int y_pred = SIGMOID(w_sum) > 0.5 ? 1 : -1;
            if (y_pred != y_true) {
                error++;
                if (y_true == 1) {
                    fn++;
                } else {
                    fp++;
                }
            } else {
                if (y_true == 1) {
                    tp++;
                } else {
                    tn++;
                }
            }
    }
    double recall = (double)tp / (tp + fn);
    double precision = (double)tp / (tp + fp);
    double accuracy = (double) (tp + tn) / (tp + tn + fp + fn);
    double f_measure = (2 * precision * recall) / (precision + recall);
    if (f_measure > best) {
        best = f_measure;
        printf("[%d\t%d]\n[%d\t%d]\n", tp, fn, fp, tn);
    }
    return f_measure;
}