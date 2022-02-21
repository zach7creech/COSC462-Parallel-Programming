/* Author: Zachery Creech
 * COSC462: Fall 2021
 * PA3 Problem 1: cannons.c
 * 12/2/2021 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void main(int ac, char** av)
{
    MPI_Comm cart_comm;
    
    int rank, nprocs, i, j, k, r, n, p, c, success;
    int nper_proc, ndims, reorder, send_rank, recv_rank, init_i, start_row, row_bound, start_col, col_bound, jump_point;
    int dim_size[2], periods[2], coord[2], send_coord[2], recv_coord[2];
    int *my_a, *my_b, *my_c, *recv_all;
    int **final_c;

    double total_comp_time, start_comp_time, total_comm_time, start_comm_time;

    
    if(ac != 3)
    {
        printf("usage: ./cannons <n> <p>\n");
        return;
    }
    
    n = atoi(av[1]);
    p = atoi(av[2]);
 
    MPI_Init(&ac, &av);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //set up cartesian shape of processors p x p
    ndims = 2;
    dim_size[0] = sqrt(nprocs);
    dim_size[1] = sqrt(nprocs);
    periods[0] = 1;
    periods[1] = 1;
    reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dim_size, periods, reorder, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coord); 
        
    nper_proc = n / sqrt(nprocs);
    
    //set all initial values on all submatrices on every processor
    
    my_a = (int *)malloc(nper_proc * nper_proc * sizeof(int));
    my_b = (int *)malloc(nper_proc * nper_proc * sizeof(int));
    my_c = (int *)malloc(nper_proc * nper_proc * sizeof(int));

    for(i = 0; i < nper_proc; i++)
    {
        for(j = 0; j < nper_proc; j++)
        {
            my_a[i * nper_proc + j] = 1;
            my_b[i * nper_proc + j] = 1;
            my_c[i * nper_proc + j] = 0;
        }
    }

    total_comm_time = 0;

    start_comm_time = MPI_Wtime();
    //initial alignment
    
    //shift my_a left coord[0] columns on all procs
    send_coord[0] = coord[0];
    send_coord[1] = coord[1] - coord[0];
    MPI_Cart_rank(cart_comm, send_coord, &send_rank);

    recv_coord[0] = coord[0];
    recv_coord[1] = coord[1] + coord[0];
    MPI_Cart_rank(cart_comm, recv_coord, &recv_rank);

    MPI_Sendrecv_replace(my_a, nper_proc * nper_proc, MPI_INT, send_rank, 0, recv_rank, 0, cart_comm, MPI_STATUS_IGNORE);

    //shift my_b up coord[1] rows on all procs
    send_coord[0] = coord[0] - coord[1];
    send_coord[1] = coord[1];
    MPI_Cart_rank(cart_comm, send_coord, &send_rank);
 
    recv_coord[0] = coord[0] + coord[1];
    recv_coord[1] = coord[1];
    MPI_Cart_rank(cart_comm, recv_coord, &recv_rank);
 
    MPI_Sendrecv_replace(my_b, nper_proc * nper_proc, MPI_INT, send_rank, 0, recv_rank, 0, cart_comm, MPI_STATUS_IGNORE);

    total_comm_time += MPI_Wtime() - start_comm_time;

    total_comp_time = 0;
    
    //for every round of communication, calculate the partial sum on each processors current submatrices a and b
    for(r = 0; r < sqrt(nprocs); r++)
    {
        start_comp_time = MPI_Wtime();
        //partial sum (dot product on current submatrix)
        for(k = 0; k < nper_proc * nper_proc; k++)
        {
            init_i = k - (k % nper_proc);
            j = k % nper_proc;
            for(i = init_i; i < init_i + nper_proc; i++)
            {
                my_c[k] += my_a[i] * my_b[j];
                j += nper_proc;
            }
        }
        total_comp_time += MPI_Wtime() - start_comp_time;

        start_comm_time = MPI_Wtime();
        
        //shift my_a left one step on all procs
        send_coord[0] = coord[0];
        send_coord[1] = coord[1] - 1;
        MPI_Cart_rank(cart_comm, send_coord, &send_rank);

        recv_coord[0] = coord[0];
        recv_coord[1] = coord[1] + 1;
        MPI_Cart_rank(cart_comm, recv_coord, &recv_rank);

        MPI_Sendrecv_replace(my_a, nper_proc * nper_proc, MPI_INT, send_rank, 0, recv_rank, 0, cart_comm, MPI_STATUS_IGNORE);
        
        //shift my_b up one step on all procs
        send_coord[0] = coord[0] - 1;
        send_coord[1] = coord[1];
        MPI_Cart_rank(cart_comm, send_coord, &send_rank);

        recv_coord[0] = coord[0] + 1;
        recv_coord[1] = coord[1];
        MPI_Cart_rank(cart_comm, recv_coord, &recv_rank);

        MPI_Sendrecv_replace(my_b, nper_proc * nper_proc, MPI_INT, send_rank, 0, recv_rank, 0, cart_comm, MPI_STATUS_IGNORE);
        
        total_comm_time += MPI_Wtime() - start_comm_time;
    }
    
    //gather all submatrices into recv_all on processor rank 0
    recv_all = (int *)malloc(n * n * sizeof(int));

    start_comm_time = MPI_Wtime();

    MPI_Gather(my_c, nper_proc * nper_proc, MPI_INT, recv_all, nper_proc * nper_proc, MPI_INT, 0, cart_comm);

    total_comm_time += MPI_Wtime() - start_comm_time;
    
    if(rank == 0)
    {
        //on rank 0, fix the final matrix into a more useable shape (2D n x n array rather than 1D list of each submatrix)
        final_c = (int **)malloc(n * sizeof(int *));
        
        for(i = 0; i < n; i++)
            final_c[i] = (int *)malloc(n * sizeof(int));
        
        c = 0;
        start_row = 0;
        row_bound = nper_proc;
        start_col = 0;
        col_bound = nper_proc;
        jump_point = sqrt(nprocs);

        //extract each element from submatrix k and place it in correct location in final_c
        //this is done because recv_all is just one long list of each element in sub-block order, for example if p = 4 and n = 4:
        //recv_all looks like 0 1 4 5 2 3 6 7 8 9 12 13 10 11 14 15 because the submatrices were stored like this:
        //0 1 | 0 1
        //2 3 | 2 3
        //---------
        //0 1 | 0 1
        //2 3 | 2 3
        for(k = 0; k < nprocs; k++)
        {
            //starting row needs to jump down to th next sub-block when the layer of sub-blocks above it has been extracted
            //in the picture above, each sub-block is extracted starting at row = 0. After the second block is finished, jump down to row = 2
            if(k == jump_point)
            {
                start_row += nper_proc;
                row_bound += nper_proc;
                jump_point += sqrt(nprocs);
            }
            //after every sub-block is printed, this is incremented to move to the next sub-block. When the row jumps, this has to go back to 0
            if(start_col == n)
            {
                start_col = 0;
                col_bound = nper_proc;
            }

            //extract all elements from current sub-block
            for(i = start_row; i < row_bound; i++)
            {
                for(j = start_col; j < col_bound; j++)
                {
                    final_c[i][j] = recv_all[c];
                    c++;
                }
            }
            start_col += nper_proc;
            col_bound += nper_proc;
        }
        
        //check that every element in the final matrix == n, since the result of matrix-matrix multiplication when both
        //matrices are filled with ones is a matrix filled with n

        success = 1;
        
        for(i = 0; i < n; i++)
        {
            if(!success)
                break;

            for(j = 0; j < n; j++)
            {
                if(final_c[i][j] != n)
                {
                    success = 0;
                    break;
                }
            }
        }
        if(success)
            printf("success\n");
        else
            fprintf(stderr, "failed on %d\n", nprocs);
    }

    printf("rank %d comp_time = %f\n", rank, total_comp_time);
    printf("rank %d comm_time = %f\n", rank, total_comm_time);

    MPI_Finalize(); 
}
