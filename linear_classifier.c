/*                      LINEAR CLASSIFIER                         */
/*															      */
#define SAMPLES 200
#define TRAINING_SAMPLES 160
#define VALIDATION_SAMPLES 40
#define MAX_EPOCHS 100
#define INPUT_SIZE 5
#define INPUT_SCL_FCT 1000
#define OUTPT_SCL_FCT 2
#define LEARNING_RT 0.5
#define LEARNING_RT_DCR 0.0049
#define OVRTRN_MAX_EPOCHS 5

#include <math.h>
#include <stdlib.h>

double linearClassifier(int X[INPUT_SIZE], int data[][SAMPLES], int *labels)
{
	int epochs = 0;						// number of training steps
	double learning_rate = LEARNING_RT; // starting learning rate(reduced by LEARNING_RT_DCR every loop)
	static int randomOrder[SAMPLES];	// list of numbers 0-199 in random order
	double weights[INPUT_SIZE];			// perceptron weights
	double bias;						// perceptron bias
	int i, j;							// counters
	int training_correct = 0;			// correct matches in training set
	int validation_correct = 0;			// correct matches in validation set
	int overtraining = 0;				// overtraining counter/flag
	double fitnessValue;				// value of fitness

	/* choose random initial weights  */
	for (i = 0; i < INPUT_SIZE; i++)
	{
		weights[i] = (double)rand() / RAND_MAX;
	}
	bias = (double)rand() / RAND_MAX;

	while (epochs < MAX_EPOCHS)
	{

		/* make list of numbers from 0 to (SAMPLES - 1) in random order */
		int k;
		for (k = 0; k < SAMPLES; k++)
		{
			randomOrder[k] = k;
		}
		for (k = SAMPLES - 1; k >= 0; k--)
		{
			int j = rand() % (k + 1);
			int temp = randomOrder[j];
			randomOrder[j] = randomOrder[k];
			randomOrder[k] = temp;
		}
		/* reset matching values */
		training_correct = 0;
		validation_correct = 0;
		/* for all members of dataset(patients or healthy) */
		for (i = 0; i < SAMPLES; i++)
		{
			/* check where they are categorized correctly */
			int row = randomOrder[i];
			int output;
			double w_sum = 0.0;
			for (j = 0; j < INPUT_SIZE; j++) {
				w_sum += weights[j] * (double)data[X[j]][row] / INPUT_SCL_FCT;
			}
			w_sum += bias;
			if (w_sum < 0)
			{
				output = -1;
			}
			else
			{
				output = 1;
			}
			int target = labels[row];
			int error = (target - output) / OUTPT_SCL_FCT;
			/* if they are categorized wrongly */
			if (error != 0)
			{
				/* if is part of training set */
				if (i < TRAINING_SAMPLES)
				{
					for (j = 0; j < INPUT_SIZE; j++)
					{
						weights[j] += learning_rate * error * (double)data[X[j]][row] / INPUT_SCL_FCT;
					}
					bias += learning_rate * error;
				}
			}
			/* if they are categorized correctly */
			else
			{
				/* if is part of training set */
				if (i < TRAINING_SAMPLES)
				{
					training_correct++;
				}
				/* if is part of validation set */
				else
				{
					validation_correct++;
				}
			}
		}

		epochs++;
		/* reduce learning rate */
		learning_rate -= LEARNING_RT_DCR;
		double mean_training_correct = (double)training_correct / TRAINING_SAMPLES;
		double mean_validation_correct = (double)validation_correct / VALIDATION_SAMPLES;
		/* if for 5 turns training set is categorized better that validation */
		/* then we have overtraining                                         */
		if (mean_training_correct > mean_validation_correct)
		{
			overtraining++;
			if (overtraining == OVRTRN_MAX_EPOCHS)
			{
				break;
			}
		}
		else
		{
			overtraining = 0;
		}
	}
	/* fitness funtion is: (total correctly categorized)/(total number) */
	fitnessValue = ((double)training_correct + (double)validation_correct) / SAMPLES;
	return fitnessValue;
}