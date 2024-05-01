/*
						A Simple Genetic Algorithm (SGA)
						Created by Stratos Georgopoulos

  based on the algorithm presented in the book:
  Michalewicz, Z., "Genetic Algorithms + Data Structures = Evolution Programs",
  Springer-Verlag, 3rd edition, 1996.

			  --------------- SGA.C  Source File ---------------
*/
#include "solution_sga.h" /* include the sga header file */

/*              RandVal                                 */
/* returns a pseudorandom value between the supplied    */
/* low and high arguments.                              */
/*                                                      */
double RandVal(double low, double high)
{
	double temp = ((((double)rand()) / RAND_MAX) * (high - low)) + low;
	if (temp < 0.5)
	{
		temp = 0;
	}
	else
	{
		temp = 1;
	}
	return temp;
}

/*                    readDataFiles                        */
/* reads in the dataset and label of each patient from     */
/* a text file and stores them to data[8192][200] and      */
/* labels[200]                                             */
/*                                                         */
void readDataFiles(void)
{
	static const char filename[] = "cancer_data.txt";
	FILE *file = fopen(filename, "r");
	char *token;
	int i;
	int j = 0;
	if (file != NULL)
	{
		static char line[30000]; /* or other suitable maximum line size */

		while (fgets(line, sizeof line, file) != NULL) /* read a line */
		{
			/* get the first token */
			token = strtok(line, "\t");
			data[0][j] = atoi(token);

			/* walk through other tokens */
			for (i = 1; i < 8192; i++)
			{
				token = strtok(NULL, "\t");
				data[i][j] = atoi(token);
			}
			j++;
		}
		fclose(file);
	}
	else
	{
		perror(filename); /* why didn't the file open? */
	}
	static const char filename2[] = "cancer_label.txt";
	file = fopen(filename2, "r");
	j = 0;
	if (file != NULL)
	{
		static char line[30];						   /* or other suitable maximum line size */
		while (fgets(line, sizeof line, file) != NULL) /* read a line */
		{
			/* get the token */
			token = strtok(line, "\t");
			labels[j] = atoi(token);
			j++;
		}
		fclose(file);
	}
	else
	{
		perror(filename2); /* why didn't the file open? */
	}
}

/*              INITIALIZE                                      */
/* reads in the upper and lower bounds of each variable         */
/* from a text file. Then it randomly generates values          */
/* between these bounds for each gene of each genotype          */
/* in the population. Be aware that the random number           */
/* generator is not seeded using the system time.               */
/*                                                              */
/* The format for the input file "gadata.txt" is as follows :   */
/* Var1_lower_bound Var1_upper_bound                            */
/* Var2_lower_bound Var2_upper_bound                            */
/* etc.                                                         */
/*                                                              */
void initialize(void)
{
	FILE *infile;
	int i, j;
	double lbound, ubound;

	/* open the input file        */
	if ((infile = fopen("gadata.txt", "r")) == NULL)
	{ /*can't open the input file*/
		fprintf(galog, "\n Cannot open the input file!\n");
		exit(-1);
	}

	/* Read in the lower and upper bounds for each variable */
	for (i = 0; i < NVARS; i++)
	{
		fscanf(infile, "%lf", &lbound);
		fscanf(infile, "%lf", &ubound);

		for (j = 0; j <= POPSIZE; j++)
		{
			Population[j].Lower[i] = lbound;
			Population[j].Upper[i] = ubound;
		}
	}

	/* Using the lower and upper bounds, randomly         */
	/* assign a value to each gene of each genotype.      */
	for (i = 0; i <= POPSIZE; i++)
		for (j = 0; j < NVARS; j++)
			Population[i].Gene[j] = RandVal(Population[i].Lower[j], Population[i].Upper[j]);

	fclose(infile);
}

/*              EVALUATE                                        */
/* is a user defined function.For each problem you solve        */
/* using this GA, you will modify this function and you         */
/* will recompile the code.The simple function "a^2+b^3+c^4+abc"*/
/* is supplied here for illustration purposes                   */
/*                                                              */
void evaluate(void)
{
	int member = 0, i = 0;
	int feature_vec[INPUT_SIZE];
	Best = 0;
	for (member = 0; member < POPSIZE; member++)
	{
		/* Conversion of binary to decimal in order to find the */
		/* right peptides                                       */
        for (int i = 0; i < INPUT_SIZE; i++) {
			feature_vec[i] = 0;
            for (int j = 0; j < 13; j++) {
                feature_vec[i] += ((int)Population[member].Gene[j + (i * 13)]) << (12 - j);
            }
        }

		// function to change
		Population[member].Fitness = linearClassifier(feature_vec, data, labels);

		/* Keep track of the best member of the population  */
		/* Note that the last member of the population holds*/
		/* a copy of the best member.                       */

		if (Population[member].Fitness > Population[POPSIZE].Fitness)
		{
			Best = member;
			Population[POPSIZE].Fitness = Population[member].Fitness;
			for (i = 0; i < NVARS; i++)
				Population[POPSIZE].Gene[i] = Population[member].Gene[i];
		}
	}
}

/*              COPY_GENOTYPES                                   */
/* 		Copies a genotype to another                             */
/*                                                               */
void copy_genotypes(struct genotype *oldgenotype, struct genotype *newgenotype)
{
	int i = 0; /* some counter variables */

	for (i = 0; i < NVARS; i++)
		newgenotype->Gene[i] = oldgenotype->Gene[i];

	newgenotype->Fitness = oldgenotype->Fitness;

	for (i = 0; i < NVARS; i++)
		newgenotype->Upper[i] = oldgenotype->Upper[i];
	for (i = 0; i < NVARS; i++)
		newgenotype->Lower[i] = oldgenotype->Lower[i];

	newgenotype->RFitness = oldgenotype->RFitness;
	newgenotype->CFitness = oldgenotype->CFitness;
	newgenotype->Selection_Prob = oldgenotype->Selection_Prob;
	newgenotype->Cumulative_Prob = oldgenotype->Cumulative_Prob;

	newgenotype->Survivor = oldgenotype->Survivor;
	newgenotype->Mate = oldgenotype->Mate;
	newgenotype->Mutate = oldgenotype->Mutate;
}

/*              COPY_POPULATION                                 */
/* 		Copies a population to another population               */
/*                                                              */
void copy_population(struct genotype old_pop[POPSIZE + 1], struct genotype new_pop[POPSIZE + 1])
{
	int mem = 0; /* some counter variables */

	for (mem = 0; mem <= POPSIZE; mem++)
		copy_genotypes(&old_pop[mem], &new_pop[mem]);
}

/*              SELECT                                          */
/* This is an implementation of STANDARD PROPORTIONAL SELECTION */
/* (or ROULETTE WHEEL SELECTION) for MAXIMIZATION problems      */
/* It also checks to make sure that the best member survives    */
/* (i.e., elitest selection).                                   */
/*                                                              */
void select_ga(void)
{
	int member, spin_num, mem; /* Some counter variables       */
	double Total_Fitness;	   /* The total population fitness */
	double roulette = 0;
	/* a variable for temporary storing of the population */
	struct genotype Buffered_Pop[POPSIZE + 1];
	int Found; /* A flag */

	/* First, find the total fitness of the population    */
	Total_Fitness = 0;
	for (member = 0; member < POPSIZE; member++)
		Total_Fitness += Population[member].Fitness;

	/* Next, calculate the probability of a selection of each genotype      */
	for (member = 0; member < POPSIZE; member++)
		Population[member].Selection_Prob = Population[member].Fitness / Total_Fitness;

	/* Now, calculate the cumulative probability of each genotype     */
	Population[0].Cumulative_Prob = Population[0].Selection_Prob;

	for (member = 1; member < POPSIZE; member++)
		Population[member].Cumulative_Prob = Population[member - 1].Cumulative_Prob +
											 Population[member].Selection_Prob;

	/* Finally, select the survivors using the cumulative probability */
	/* and create the new Population                                  */
	for (spin_num = 0; spin_num < POPSIZE; spin_num++)
	{
		Found = FALSE;
		roulette = (double)rand() / RAND_MAX;

		if (roulette < Population[0].Cumulative_Prob) /* Select the first member of the Population */
			copy_genotypes(&Population[0], &Buffered_Pop[spin_num]);
		else if (roulette > Population[POPSIZE - 1].Cumulative_Prob) /* select the best member of the population */
			copy_genotypes(&Population[POPSIZE], &Buffered_Pop[spin_num]);
		else
			for (mem = 1; mem < POPSIZE && Found == FALSE; mem++)
				if ((roulette > Population[mem - 1].Cumulative_Prob) &&
					(roulette <= Population[mem].Cumulative_Prob))
				{
					copy_genotypes(&Population[mem], &Buffered_Pop[spin_num]);
					Found = TRUE;
				}
	}

	/* copy the best individual */
	copy_genotypes(&Population[POPSIZE], &Buffered_Pop[POPSIZE]);

	/* Copy the Buffered_Pop to the original Population */
	copy_population(Buffered_Pop, Population);

	/* Population , now is the new population           */
}

/*              CROSSOVER                                               */
/* This is an implementation of STANDARD SINGLE POINT CROSSOVER.        */
/* Many other crossover operators developed specifically for            */
/* real-coded GA's may give better results. For simplicity only         */
/* the single point crossover is shown here.                            */
/*                                                                      */
void crossover(void)
{
	int mem, i,		 /* Counting variables   */
		parent1,	 /* Parent one           */
		parent2,	 /* Parent two           */
		xover_point, /* Crossover point      */
		count = 0,
		lovers = 0, /* number of matting genotypes */
		indiv = 0;
	struct genotype temp_parent;

	for (mem = 0; mem <= POPSIZE; mem++)
		Population[mem].Mate = FALSE;

	/* first find the individuals for the Crossover operation */
	for (mem = 0; mem < POPSIZE; mem++)
		if (frand() < PXOVER)
		{ /* frand returns a random number in the range [0,1] */
			Population[mem].Mate = TRUE;
			lovers++;
		}

	/* We want an even number of "lovers"*/
	if ((lovers % 2) != 0)
	{
		do /* find an non "lover" */
			indiv = rand() % POPSIZE;
		while (Population[indiv].Mate == TRUE);
		/* make it "lover"    */
		Population[indiv].Mate = TRUE;
		lovers++;
	}

	/* second mate the "lovers" */
	mem = 0;
	for (count = 0; count < (lovers / 2); count++)
	{
		while (Population[mem].Mate == FALSE)
			mem++; /* find the first parent */
		parent1 = mem;
		mem++;
		while (Population[mem].Mate == FALSE)
			mem++; /* find the second parent */
		parent2 = mem;
		mem++;

		/* select the crossover point :1...NVARS-1 */
		xover_point = (rand() % (NVARS - 1)) + 1;

		/* Perform the crossover */
		/* copy parent1 to temp_parent */
		copy_genotypes(&Population[parent1], &temp_parent);

		for (i = xover_point; i < NVARS; i++)
			Population[parent1].Gene[i] = Population[parent2].Gene[i];
		for (i = xover_point; i < NVARS; i++)
			Population[parent2].Gene[i] = temp_parent.Gene[i];
	}
	/* set Mate flag to False */
	for (mem = 0; mem <= POPSIZE; mem++)
		Population[mem].Mate = FALSE;
}

/*              MUTATION                                                */
/* This is an implementation of random mutation. A value selected       */
/* for mutation is replaced by a randomly generated number between      */
/* that variable's lower and upper bounds.                              */
/* As is the case with crossover, other mutations developed for         */
/* real-coded GA's can be added.                                        */
/*                                                                      */
void mutate(void)
{
	double lbound, hbound;
	int member, /* The member to be mutated                 */
		var;	/* The variable to be mutated               */

	for (member = 0; member < POPSIZE; member++) /* for every member */
		for (var = 0; var < NVARS; var++)		 /* for every gene   */
			if (frand() < PMUTATION)
			{
				lbound = Population[member].Lower[var];
				hbound = Population[member].Upper[var];

				/* Generate a new random value between the bounds */
				Population[member].Gene[var] = RandVal(lbound, hbound);
			}
}

/*              REPORT                                          */
/* Report progress of the simulation. Data is dumped to a log   */
/* file in comma separated value format which can be imported   */
/* and graphed using any commercial spreadsheet package.        */
/*                                                              */
void report(void)
{
	double best_val, /* Best population fitness                      */
		avg,		 /* Average population fitness                   */
		stddev,		 /* Std. deviation of population fitness         */
		sum_square,	 /* Sum of the squares for std. dev calc.        */
		square_sum,	 /* Square of the sums for std. dev. calc.       */
		sum;		 /* The summed population fitness                */
	int i = 0;		 /* counter */

	sum = 0.0;
	sum_square = 0.0;

	/* Calculate the summed population fitness and the sum        */
	/* of the squared individual fitnesses.                       */
	for (i = 0; i < POPSIZE; i++)
	{
		sum += (Population[i].Fitness);
		sum_square += pow(Population[i].Fitness, 2);
	}

	/* Calculate the average and standard deviations of the       */
	/* population's fitness.                                      */
	avg = sum / (double)POPSIZE;
	square_sum = sum * sum / (double)POPSIZE;
	stddev = sqrt((1.0 / (double)(POPSIZE - 1)) * (sum_square - square_sum));
	best_val = Population[POPSIZE].Fitness;

	/* Print the generation, best, avg, and std. deviation to a   */
	/* file in csv format.                                        */
	fprintf(galog, "\n%10d  %15.4f  %15.4f  %15.4f", Generation, best_val, avg, stddev);
	/* Print the Best Genotype */
	fprintf(galog, "  (");
	for (i = 0; i < NVARS; i++)
		fprintf(galog, " %5.3f ", Population[POPSIZE].Gene[i]);
	fprintf(galog, ") ");
}

/*                     MAIN                                     */
/* This is the main function. It loops for the specified number */
/* of generations. Each generation involves selecting survivors,*/
/* performing crossover and mutation, and then evaluating the   */
/* resulting population.                                        */
/*                                                              */
int main(void)
{
	int i, j;

	srand(time(0));

	if ((galog = fopen("galog.txt", "w")) == NULL)
	{
		printf("Can't open galog.txt \n");
		exit(1);
	}
	fprintf(galog, "%10s  %15s  %15s  %15s  %20s\n", "Generation", "Best Value", "Average", "StdDev", "Best Genotype");

	Generation = 0;
	readDataFiles();
	initialize();
	printf("Initial Population\n");
	for (j = 0; j < NVARS; j++)
		printf("  Gene%d   ", j);
	printf("\n");
	for (i = 0; i < POPSIZE; i++)
	{
		for (j = 0; j < NVARS; j++)
			printf("%g ", Population[i].Gene[j]);
		printf("\n");
	}

	evaluate();
	while (Generation < MAXGENS)
	{
		Generation++;
		select_ga();
		crossover();
		mutate();
		evaluate();

		if (Generation % DISPLAYFREQ == 0)
			printf("Generation : %d/%d \t Best Fitness: %5f\n", Generation, MAXGENS, Population[POPSIZE].Fitness);

		report();
	}
	/* print final result to screen */
	printf("\n\nSimulation completed\n");
	printf("\n   Best Member:");
	for (i = 0; i < NVARS; i++)
		printf(" %f ", Population[POPSIZE].Gene[i]);
	printf("  Fitness: %5f\n", Population[POPSIZE].Fitness);

	fprintf(galog, "\n\nSimulation completed\n");
	fprintf(galog, "\n   Best member :\n");

	for (i = 0; i < NVARS; i++)
		fprintf(galog, "\n Var(%d) = %3.3f", i, Population[POPSIZE].Gene[i]);
	fprintf(galog, "  Fitness: %5f\n", Population[POPSIZE].Fitness);

	fclose(galog);

	return 0;
}

/* ------------------------- THE END ---------------------------        */
