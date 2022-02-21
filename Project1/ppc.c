/* Author: Zachery Creech
 * COSC462: Fall 2021
 * Problem 1: Point-to-Point Communication
 * 09/27/2021 
 * Question 2: Based on the graph, tau is roughly 0.091 and mu is roughly 1.22 x 10^-6. Pretty sure that isn't right,
 * but trying to run the program with a message size larger than 10000 never seemed to finish. */

#include <stdio.h>
#include <mpi.h>

int main(int ac, char **av)
{
  int rank, nprocs, my_integer, recv_integer, send_rank, recv_rank;
  MPI_Init(&ac, &av);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  my_integer = rank + 10;

  if(rank != nprocs - 1)
    send_rank = rank + 1;
  else
    send_rank = 0;

  if(rank != 0)
    recv_rank = rank - 1;
  else
    recv_rank = nprocs - 1;
    
  MPI_Send(&my_integer, 1, MPI_INT, send_rank, 0, MPI_COMM_WORLD);
  MPI_Recv(&recv_integer, 1, MPI_INT, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  printf("Rank: %d Received %d from rank %d.\n", rank, recv_integer, recv_rank);
  
  if(rank != 0)
    send_rank = rank - 1;
  else
    send_rank = nprocs - 1;

  if(rank != nprocs - 1)
    recv_rank = rank + 1;
  else
    recv_rank = 0;

  MPI_Send(&my_integer, 1, MPI_INT, send_rank, 0, MPI_COMM_WORLD);
  MPI_Recv(&recv_integer, 1, MPI_INT, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  printf("Rank: %d Received %d from rank %d.\n", rank, recv_integer, recv_rank);
  
  MPI_Finalize(); 
}
