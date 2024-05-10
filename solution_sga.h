/*
						A Simple Genetic Algorithm (SGA)
						Created by Stratos Georgopoulos

  based on the algorithm presented in the book:
  Michalewicz, Z., "Genetic Algorithms + Data Structures = Evolution Programs",
  Springer-Verlag, 3rd edition, 1996.

			  --------------- SGA.H  Header File ---------------
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define frand_ab() (2 * ((double)rand() / RAND_MAX) - 1)
#define frand() ((double)rand() / RAND_MAX)

/* Input files */

#define DATA_COLUMNS 8192 /* Nr of feature vectors */
#define DATA_ROWS 200	  /* Nr of samples         */

/* Feature selection */
#define INPUT_SIZE 5

/* change any of these parameters to match your needs */

#define POPSIZE 200 						 /* Population size                       */
#define MAXGENS 1000 						 /* Maximum number of generations         */
#define CHROMOSOME_SIZE 13					 /* Chromosome size                       */
#define NVARS (CHROMOSOME_SIZE * INPUT_SIZE) /* Number of problem variables           */
#define PXOVER 0.9							 /* Probability of crossover              */
#define PMUTATION 0.01						 /* Probability of mutation	         	  */
#define DISPLAYFREQ 100						 /* Display frequency of the best fitness */
#define TRUE 1
#define FALSE 0

int Generation, /* Curent Generation number                 */
	Best;		/* The Best genotype in the population      */
FILE *galog;	/* A log file to write results to           */
static int sampleSize = DATA_ROWS;
static int data[DATA_COLUMNS][DATA_ROWS];
static int labels[DATA_ROWS];

struct genotype /* Each genotype is a member of      */
{				/* the population                    */

	double Gene[NVARS],	 /* A string of variables (genotype)  */
		Fitness,		 /* The genotype's fitness            */
		Upper[NVARS],	 /* The genotype's upper bound        */
		Lower[NVARS],	 /* The genotype's lower bound        */
		RFitness,		 /* The relative fitness              */
		CFitness,		 /* The cumulative fitness            */
		Selection_Prob,	 /* The probability of a selection    */
		Cumulative_Prob; /* The cumulative probability        */

	int Survivor, /* Flag for selection procedure      */
		Mate,	  /* Flag for Crossover procedure      */
		Mutate;	  /* Flag for Mutation procedure       */
};

struct genotype Population[POPSIZE + 1]; /* The population of genotypes */
struct genotype Best_Individual;

/* GA function prototypes */
void readDataFiles(void);
int *rpermute(int);
double linearClassifier(int X[INPUT_SIZE], int data[][DATA_ROWS], int *labels);
void initialize(void);
double RandVal(double, double);
void evaluate(void);
void copy_genotypes(struct genotype *, struct genotype *);
void copy_population(struct genotype old_pop[POPSIZE + 1], struct genotype new_pop[POPSIZE + 1]);
void select_ga(void);
void crossover(void);
void mutate(void);
void report(void);
