#define MAX_EPOCHS 50
#define SAMPLES 200
#define SPLIT_RATIO 0.8
#define TRN_SAMPLES (SAMPLES * SPLIT_RATIO)
#define TST_SAMPLES (SAMPLES - TRN_SAMPLES)
#define NR_FEATURES 5
#define LEARNING_RT 6
#define RANDOM_INIT_W 1
#define INPUT_SCL_FCT 1000

#define FRAND_NRM() (((double) rand() / RAND_MAX) * 2.0 - 1.0)
#define SIGMOID(x) (1 / (1 + exp(-x)))

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// Declarations
double dot_product(const double *a, const double *b, const int vector_size);
double vec_norm(const double *a, const int vector_size);
double cross_entropy_loss(double y_true, double y_pred);
double test_model(const int *feat_index, const double *weights, const int data[][SAMPLES], const int *y, const int *order, int start, int end);
void shuffle_rows(int *randomOrder);

double linearClassifier(const int *feat_index, const int data[][SAMPLES], const int *y)
{
    double X[NR_FEATURES+1] = {0};  // X vector init to all zeroes
    X[NR_FEATURES] = 1.0;           // Initialize input to bias to one
    double weights[NR_FEATURES+1] = {0};
    double gradient[NR_FEATURES+1];
    double trn_loss = 0.0;
    static int randomOrder[SAMPLES];
    int epoch = 0;
    double fitness = 0.0;

    /* assign random [-1,1] or zero initial weight values depending on RANDOM_INIT_W flag */
    if (RANDOM_INIT_W) {
        for (int i = 0; i < NR_FEATURES+1; i++) {
            weights[i] = FRAND_NRM();
        }
    }

    /* shuffle rows once to split samples in training and test set*/
    shuffle_rows(randomOrder);

    /* outer loop for every epoch */
    while (epoch++ < MAX_EPOCHS) {

        /* initializations to zero */
        for (int i = 0; i < NR_FEATURES+1; i++) {
            gradient[i] = 0;
        }
        trn_loss = 0.0;

        /* inner loop for every training sample */
        for (int j = 0; j < TRN_SAMPLES; j++) {

            int row = randomOrder[j];

            /* Load X vector with apropriate values */
            for (int i = 0; i < NR_FEATURES; i++) {
                X[i] = (double)data[feat_index[i]][row] / INPUT_SCL_FCT;
            }

            /* load label with mapping -1 to 0 and 1 to 1 */
            int y_true = (y[row] + 1) / 2;

            /* perform dot product of <w,x> */
            double w_sum = dot_product(weights, X, NR_FEATURES + 1);

            /* apply activation function */
            double y_pred = SIGMOID(w_sum);

            /* Calculate loss for this sample */
            trn_loss += cross_entropy_loss(y_true, y_pred);

            /* Calulate gradient for this sample */
            for (int i = 0; i < NR_FEATURES+1; i++) {
                gradient[i] += X[i] * (y_pred - y_true) / TRN_SAMPLES;
            }
        }

        // Update weights (once per epoch - batch gradient descend)
        for (int i = 0; i < NR_FEATURES+1; i++) {
            weights[i] -= LEARNING_RT * gradient[i];
        }
    }

    /* Evaluate model after training on all 200 data samples (for direct comparison)  */
    fitness = test_model(feat_index, weights, data, y, randomOrder, 0, SAMPLES);

    /* Evaluate model after training only on test samples
    fitness = test_model(feat_index, weights, data, y, randomOrder, TRN_SAMPLES, SAMPLES);
    */

    return fitness;
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
double cross_entropy_loss(double y_true, double y_pred) {
    return -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred));
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
double test_model(const int *feat_index, const double *weights, const int data[][SAMPLES], const int *y, const int *order, int start, int end) {
    int fp = 0, tp = 0, tn = 0, fn = 0;
    double X[NR_FEATURES+1] = {0};
    X[NR_FEATURES] = 1.0;
    double tst_loss = 0.0;
    for (int j = start; j < end; j++) {
            int row = order[j];
            for (int i = 0; i < NR_FEATURES; i++) {
                X[i] = (double)data[feat_index[i]][row] / INPUT_SCL_FCT;
            }
            int y_true = (y[row] + 1) / 2;
            double w_sum = dot_product(X, weights, NR_FEATURES+1);
            double y_pred = SIGMOID(w_sum);
            tst_loss += cross_entropy_loss(y_true, y_pred);
            int y_pred_label = y_pred > 0.5 ? 1 : 0;
            if (y_pred_label != y_true) {
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

    double fitness;

    /* Metric is Mathews Correlation Coefficient
    double mcc;
    double mcc_denom = sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
    if (mcc_denom == 0.0)
        mcc = (tp * tn - fp * fn);
    else
        mcc = (tp * tn - fp * fn) / mcc_denom;
    
    fitness = mcc;
    */

    /* Metric is Accuracy (for direct comparison with previous classifier) */
    double accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
    fitness = accuracy;
  
    return fitness;
}