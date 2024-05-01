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
	double weights[INPUT_SIZE + 1];		// perceptron weights
	int i, j;							// counters
	int training_correct = 0;			// correct matches in training set
	int validation_correct = 0;			// correct matches in validation set
	int overtraining = 0;				// overtraining counter/flag
	double fitnessValue;				// value of fitness

	/* choose random initial weights  */
	for (i = 0; i < INPUT_SIZE + 1; i++)
	{
		weights[i] = (double)rand() / RAND_MAX;
	}

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
			if  ((weights[0] * data[X[0]][row] +
			      weights[1] * data[X[1]][row] +
				  weights[2] * data[X[2]][row] +
				  weights[3] * data[X[3]][row] +
				  weights[4] * data[X[4]][row] - weights[5]) < 0)
			{
				output = -1;
			}
			else
			{
				output = 1;
			}
			/* if they are categorized wrongly */
			int target = labels[row];
			if (output != target)
			{
				/* if is part of training set */
				if (i < TRAINING_SAMPLES)
				{
					for (j = 0; j < INPUT_SIZE; j++)
					{
						weights[j] += learning_rate * ((double)(target - output) / OUTPT_SCL_FCT) * (double)data[X[j]][row] / INPUT_SCL_FCT;
					}
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