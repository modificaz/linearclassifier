/*                      LINEAR CLASSIFIER                         */
/*															      */
#define SAMPLES 200
#define INPUT_SIZE 5

#include <math.h>
#include <stdlib.h>

double linearClassifier(int X[INPUT_SIZE], int data[][SAMPLES], int *labels)
{
	int iterations = 0;			 // number of training steps
	double learning_rate = 0.5;	 // starting learning rate(reduced by 0.0049 every loop)
	static int randomOrder[200]; // list of numbers 0-199 in random order
	double weights[6];			 // perceptron weights
	int i, j;					 // counters
	int training_correct = 0;	 // correct matches in training set
	int validation_correct = 0;	 // correct matches in validation set
	int overtraining = 0;		 // overtraining counter/flag
	double fitnessValue;		 // value of fitness

	/* choose random initial weights  */
	for (i = 0; i < 6; i++)
	{
		weights[i] = (double)rand() / RAND_MAX;
	}

	while (iterations < 100)
	{

		/* make list of numbers 0-199 in random order */
		int k;
		for (k = 0; k < 200; k++)
		{
			randomOrder[k] = k;
		}
		for (k = 200 - 1; k >= 0; k--)
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
		for (i = 0; i < 200; i++)
		{
			/* check where they are categorized correctly */
			int y;
			if (((weights[0] * data[X[0]][randomOrder[i]]) + (weights[1] * data[X[1]][randomOrder[i]]) + (weights[2] * data[X[2]][randomOrder[i]]) + (weights[3] * data[X[3]][randomOrder[i]]) + (weights[4] * data[X[4]][randomOrder[i]]) - weights[5]) < 0)
			{
				y = -1;
			}
			else
			{
				y = 1;
			}
			/* if they are categorized wrongly */
			if (y != labels[randomOrder[i]])
			{
				/* if is part of training set */
				if (i < 160)
				{
					for (j = 0; j < 5; j++)
					{
						weights[j] = weights[j] + (double)(learning_rate / 1000) * (double)(labels[randomOrder[i]] - y) * (double)data[X[j]][randomOrder[i]] / 2;
					}
				}
			}
			/* if they are categorized correctly */
			else
			{
				/* if is part of training set */
				if (i < 160)
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

		iterations++;
		/* reduce learning rate */
		learning_rate = learning_rate - 0.0049;
		double mean_training_correct = (double)training_correct / 160;
		double mean_validation_correct = (double)validation_correct / 40;
		/* if for 5 turns training set is categorized better that validation */
		/* then we have overtraining                                         */
		if (mean_training_correct > mean_validation_correct)
		{
			overtraining++;
			if (overtraining == 5)
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
	fitnessValue = ((double)training_correct + (double)validation_correct) / 200;
	return fitnessValue;
}