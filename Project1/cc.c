/* Author: Zachery Creech
 * COSC462: Fall 2021
 * Problem 2: Collective Communication
 * 09/27/2021 
 * Note about both programs: running either may print output in a weird order since
 * it's executing in parallel. */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int ac, char **av)
{
  int rank, nprocs, *all_ints, *recv_buf, my_integer, i;
  MPI_Init(&ac, &av);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  my_integer = rank + 10;

  //Question 1
  if(rank == 0)
  {
    printf("Question 1\n");
    all_ints = (int *)malloc(nprocs*sizeof(int));
    recv_buf = all_ints;
  }

  MPI_Gather(&my_integer, 1, MPI_INT, recv_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(rank == 0)
  {
    printf("Rank 0: Received ");
    for(i = 0; i < nprocs; i++)
    {
      printf("%d ", *recv_buf);
      recv_buf++;
    }
    printf("\n");
  }

  if(rank == 0)
    free(all_ints);
  
  //Question 2
  if(rank == 0)
    printf("Question 2\n");

  all_ints = (int *)malloc(nprocs*sizeof(int));
  recv_buf = all_ints;

  MPI_Allgather(&my_integer, 1, MPI_INT, recv_buf, 1, MPI_INT, MPI_COMM_WORLD);

  printf("Rank %d: Received ", rank);
  for(i = 0; i < nprocs; i++)
  {
    printf("%d ", *recv_buf);
    recv_buf++;
  }
  printf("\n");

  free(all_ints);

  //Question 3
  if(rank == 0)
  {
    printf("Question 3\n");
    all_ints = (int *)malloc(nprocs*sizeof(int));
    recv_buf = all_ints;
  }

  MPI_Reduce(&my_integer, recv_buf, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if(rank == 0)
    printf("Rank 0: Sum is %d.\n", *recv_buf);

  MPI_Reduce(&my_integer, recv_buf, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);

  if(rank == 0)
    printf("Rank 0: Product is %d.\n", *recv_buf);  

  if(rank == 0)
    free(all_ints);

  //Question 4
  if(rank == 0)
    printf("Question 4\n");

  all_ints = (int *)malloc(nprocs*sizeof(int));
  recv_buf = all_ints;

  MPI_Allreduce(&my_integer, recv_buf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  printf("Rank %d: Sum is %d.\n", rank, *recv_buf);

  MPI_Allreduce(&my_integer, recv_buf, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);

  printf("Rank %d: Product is %d.\n", rank, *recv_buf);  

  free(all_ints);

  MPI_Finalize(); 
}
