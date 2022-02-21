/* Author: Zachery Creech
 * COSC462 Fall 2021
 * PA2 Problem 2: integration.c
 * 10/28/2021 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void main(int ac, char** av)
{
    int rank, nprocs, i, nsteps, nsteps_perproc, end_step;
    double my_sum, sum, pi, x, step;

    MPI_Init(&ac, &av);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(ac < 2)
	{
		fprintf(stderr, "usage: ./integration <nsteps>\n");
		exit(1);
	}

	my_sum = 0.0;
	nsteps = atoi(av[1]);
    step = 1.0 / nsteps;

    nsteps_perproc = nsteps / nprocs;

    //each processor will compute the partial sum of nsteps / nprocs
    //if the nsteps cannot be divided evenly among processors, the last processor will get a partition plus the remainder
    if(rank == nprocs - 1 && nsteps % nprocs != 0)
        end_step = ((rank + 1) * nsteps_perproc) + nsteps % nprocs;
    else
        end_step = (rank + 1) * nsteps_perproc;
	
	//compute each partial sum on each processor
    for(i = rank * nsteps_perproc; i < end_step; i++)
	{
		x = (i + 0.5) * step;
		my_sum += 4.0 / (1.0 + x*x);
	}
	
	//sum them all up on processor 0
    MPI_Reduce(&my_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        pi = sum * step;
	    printf("The value of pi = %f\n", pi);
    }

    MPI_Finalize();
}
