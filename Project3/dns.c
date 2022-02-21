/* Author: Zachery Creech
 * COSC462: Fall 2021
 * PA3 Problem 2: dns.c
 * Running this with the shell script and P = 512 will likely cause a vmem error, even on monster partition. Not sure why. Runs fine locally.
 * 12/2/2021 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void main(int ac, char** av)
{
    int n, p;
    
    if(ac != 3)
    {
        printf("usage: ./dns <n> <p>\n");
        return;
    }
    
    n = atoi(av[1]);
    p = atoi(av[2]);

    MPI_Comm cart_comm, col_comm, row_comm, reduce_comm, gather_comm;
    MPI_Group world_group, col_group, row_group, reduce_group;
    
    int i, j, k, r, c, rank, nprocs, success;
    int nper_proc, ndims, reorder, send_rank, recv_rank, broad_root, reduce_root, init_i, start_row, row_bound, start_col, col_bound, jump_point;
    int dim_size[3], periods[3], coord[3], send_coord[3], recv_coord[3], col_broad_coord[3], row_broad_coord[3], reduce_coord[3];
    int col_remain_dims[3], row_remain_dims[3], reduce_remain_dims[3], gather_remain_dims[3];
    int world_ranks[p], col_ranks[p], row_ranks[p], reduce_ranks[p];
    int *my_a, *my_b, *my_c, *reduce_c, *recv_all;
    int **final_c;

    double total_comp_time, start_comp_time, total_comm_time, start_comm_time;

    total_comp_time = 0;
    total_comm_time = 0;
    
    MPI_Init(&ac, &av);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //set up cartesian shape of processors p x p x p
    ndims = 3;
    dim_size[0] = cbrt(nprocs);
    dim_size[1] = cbrt(nprocs);
    dim_size[2] = cbrt(nprocs);
    periods[0] = 1;
    periods[1] = 1;
    periods[2] = 1;
    reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dim_size, periods, reorder, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 3, coord); 
        
    nper_proc = n / cbrt(nprocs);
    
    //set all initial values on all submatrices on every processor
    //set 1s on k = 0 and -1s on k > 0 to ensure that communication is actually happening

    my_a = (int *)malloc(nper_proc * nper_proc * sizeof(int));
    my_b = (int *)malloc(nper_proc * nper_proc * sizeof(int));
    my_c = (int *)malloc(nper_proc * nper_proc * sizeof(int));

    for(i = 0; i < nper_proc; i++)
    {
        for(j = 0; j < nper_proc; j++)
        {
            if(coord[0] == 0)
            {
                my_a[i * nper_proc + j] = 1;
                my_b[i * nper_proc + j] = 1;
            }
            else
            {
                my_a[i * nper_proc + j] = -1;
                my_b[i * nper_proc + j] = -1;
            }
            my_c[i * nper_proc + j] = 0;
        }
    }
    
    //on processors k = 0, send columns to k = j and rows k = i
    if(coord[0] == 0)
    {
        reduce_c = (int *)malloc(nper_proc * nper_proc * sizeof(int));
        
        start_comm_time = MPI_Wtime();
        
        //send a_matrix from one processor k == 0 to another processor, except for j = 0
        if(coord[2] != 0)
        {
            send_coord[0] = coord[2];
            send_coord[1] = coord[1];
            send_coord[2] = coord[2];
            MPI_Cart_rank(cart_comm, send_coord, &send_rank);
            MPI_Send(my_a, nper_proc * nper_proc, MPI_INT, send_rank, 0, cart_comm);
        }
        //send b_matrix from one processor k == 0 to another processor, except for i = 0
        if(coord[1] != 0)
        {
            send_coord[0] = coord[1];
            send_coord[1] = coord[1];
            send_coord[2] = coord[2];
            MPI_Cart_rank(cart_comm, send_coord, &send_rank);
            MPI_Send(my_b, nper_proc * nper_proc, MPI_INT, send_rank, 0, cart_comm);
        }

        total_comm_time += MPI_Wtime() - start_comm_time;
    }
    //receive a and/or b matrices on all rows/columns k > 0
    else if(coord[0] == coord[2] || coord[0] == coord[1])
    {
        start_comm_time = MPI_Wtime();

        recv_coord[0] = 0;
        recv_coord[1] = coord[1];
        recv_coord[2] = coord[2];
        MPI_Cart_rank(cart_comm, recv_coord, &recv_rank);
        
        //receive a_matrix on all k == j, receive b_matrix on all k == i
        if(coord[0] == coord[2])
            MPI_Recv(my_a, nper_proc * nper_proc, MPI_INT, recv_rank, 0, cart_comm, MPI_STATUS_IGNORE);
        if(coord[0] == coord[1])
            MPI_Recv(my_b, nper_proc * nper_proc, MPI_INT, recv_rank, 0, cart_comm, MPI_STATUS_IGNORE);
    
        total_comm_time = MPI_Wtime() - start_comm_time;
    }

    start_comm_time = MPI_Wtime();

    //create new communication along all columns (j dimension)
    col_remain_dims[0] = 0;
    col_remain_dims[1] = 0;
    col_remain_dims[2] = 1;
    MPI_Cart_sub(cart_comm, col_remain_dims, &col_comm);
    
    //create new communication along all rows (i dimension)
    row_remain_dims[0] = 0;
    row_remain_dims[1] = 1;
    row_remain_dims[2] = 0;
    MPI_Cart_sub(cart_comm, row_remain_dims, &row_comm);
    
    for(i = 0; i < nprocs; i++)
        world_ranks[i] = i;

    MPI_Comm_group(cart_comm, &world_group);
   
    //broadcast a_matrix from processor (this_k, this_i, this_k) to all processors in this column
    col_broad_coord[0] = coord[0];
    col_broad_coord[1] = coord[1];
    col_broad_coord[2] = coord[0];
    MPI_Cart_rank(cart_comm, col_broad_coord, &broad_root);
    MPI_Comm_group(col_comm, &col_group);
    MPI_Group_translate_ranks(world_group, nprocs, world_ranks, col_group, col_ranks);
    MPI_Bcast(my_a, nper_proc * nper_proc, MPI_INT, col_ranks[broad_root], col_comm);
 
    //broadcast b_matrix from processor (this_k, this_k, this_j) to all processors in this row
    row_broad_coord[0] = coord[0];
    row_broad_coord[1] = coord[0];
    row_broad_coord[2] = coord[2];
    MPI_Cart_rank(cart_comm, row_broad_coord, &broad_root);
    MPI_Comm_group(row_comm, &row_group);
    MPI_Group_translate_ranks(world_group, nprocs, world_ranks, row_group, row_ranks);
    MPI_Bcast(my_b, nper_proc * nper_proc, MPI_INT, row_ranks[broad_root], row_comm);

    total_comm_time += MPI_Wtime() - start_comm_time;
    
    start_comp_time = MPI_Wtime();

    //all processors should now have their correct copies of a_matrix and b_matrix
    
    //calculate partial sum for each element of this submatrix
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
        
    total_comp_time = MPI_Wtime() - start_comp_time;
    
    start_comm_time = MPI_Wtime();

    //create new communication along each k-column (k-dimension)
    reduce_remain_dims[0] = 1;
    reduce_remain_dims[1] = 0;
    reduce_remain_dims[2] = 0;
    MPI_Cart_sub(cart_comm, reduce_remain_dims, &reduce_comm);
 
    //do a reduction that sums all elements in k-column submatrices and sends resulting matrix to (0, this_i, this_j)
    reduce_coord[0] = 0;
    reduce_coord[1] = coord[1];
    reduce_coord[2] = coord[2];
    MPI_Cart_rank(cart_comm, reduce_coord, &reduce_root);
    MPI_Comm_group(reduce_comm, &reduce_group);
    MPI_Group_translate_ranks(world_group, nprocs, world_ranks, reduce_group, reduce_ranks);
    
    MPI_Reduce(my_c, reduce_c, nper_proc * nper_proc, MPI_INT, MPI_SUM, reduce_ranks[reduce_root], reduce_comm);
    
    //create new communication along k = 0 (keeping just layer k = 0) to gather all reduced submatrices on rank 0
    gather_remain_dims[0] = 0;
    gather_remain_dims[1] = 1;
    gather_remain_dims[2] = 1;
            
    MPI_Cart_sub(cart_comm, gather_remain_dims, &gather_comm);

    total_comm_time += MPI_Wtime() - start_comm_time;

    if(coord[0] == 0)
    {
        //gather all reduced submatrices into recv_all on processor rank 0
        recv_all = (int *)malloc(n * n * sizeof(int));

        start_comm_time = MPI_Wtime();
        
        MPI_Gather(reduce_c, nper_proc * nper_proc, MPI_INT, recv_all, nper_proc * nper_proc, MPI_INT, 0, gather_comm);

        total_comm_time += MPI_Wtime() - start_comm_time;
        
        if(rank == 0)
        {
            //on rank 0, fix the final matrix into a more useable shape (2D n x n array rather than 1D list of each submatrix)
            final_c = (int **)malloc(n * sizeof(int *));
        
            for(i = 0; i < n; i++)
                final_c[i] = (int *)malloc(n * sizeof(int));
       
            //set nprocs to be what it would be if only processors on k = 0 existed, then fix the final matrix as done in cannons.c
            nprocs = cbrt(nprocs) * cbrt(nprocs);
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
                //starting row needs to jump down to the next sub-block when the layer of sub-blocks above it has been extracted
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
    }

    printf("rank %d comp_time = %f\n", rank, total_comp_time);
    printf("rank %d comm_time = %f\n", rank, total_comm_time);

    MPI_Finalize(); 
}
