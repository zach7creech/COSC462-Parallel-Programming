/* Author: Zachery Creech
 * COSC462: Fall 2021
 * PA2 Problem 1: prefix_polynomial.c
 * 10/28/2021 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void main(int ac, char** av)
{
    int rank, nprocs, i, coef_perproc, exp;
    double *send_coef, *recv_coef, sum, x, my_sum;

    MPI_Init(&ac, &av);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //divide the coefficients evenly among all processors
    coef_perproc = 64 / nprocs;
    
    //generate the coefficients on processor 0
    if(rank == 0)
    {
        send_coef = (double *)malloc(64 * sizeof(double));

        srand(0);

        for(i = 0; i < 64; i++)
            send_coef[i] = 1.0 / (rand() % 100 + 1);
    }

    //send partitions of coefficient array to each processor with MPI_Scatter
    recv_coef = (double *)malloc(coef_perproc * sizeof(double));

    MPI_Scatter(send_coef, coef_perproc, MPI_DOUBLE, recv_coef, coef_perproc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    x = 0.5;
    my_sum = 0;
    exp = rank * coef_perproc;

    //on each processor, compute the partial sum
    for(i = 0; i < coef_perproc; i++)
    {
        my_sum += recv_coef[i] * pow(x, exp);
        exp++;
    }

    //use parallel prefix to get total sum on last processor
    MPI_Scan(&my_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(rank == nprocs - 1)
        printf("Rank %d: Received %f\n", nprocs - 1, sum); 

    if(rank == 0)
        free(send_coef);
    
    free(recv_coef);

    MPI_Finalize(); 
}
