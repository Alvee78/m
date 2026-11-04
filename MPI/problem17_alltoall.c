/*
 * Problem 17: All-to-All Communication with MPI_Alltoall
 * Each process generates data for every other process. After MPI_Alltoall,
 * each process holds the j-th element from every other process.
 *
 * Compile: mpicc -o problem17 problem17_alltoall.c
 * Run:     mpirun -np 4 ./problem17
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Each process creates one value per destination process */
    int *send_buf = (int *)malloc(size * sizeof(int));
    int *recv_buf = (int *)malloc(size * sizeof(int));

    for (int j = 0; j < size; j++)
        send_buf[j] = rank * 10 + j;   /* P0: {0,1,2,3}, P1:{10,11,12,13}, etc. */

    printf("P%d sends:", rank);
    for (int j = 0; j < size; j++) printf(" %d", send_buf[j]);
    printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);

    double t_start = MPI_Wtime();
    MPI_Alltoall(send_buf, 1, MPI_INT, recv_buf, 1, MPI_INT, MPI_COMM_WORLD);
    double t_end   = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    printf("P%d received:", rank);
    for (int j = 0; j < size; j++) printf(" %d", recv_buf[j]);
    printf("\n");

    if (rank == 0)
        printf("\nMPI_Alltoall with %d processes took %.6f seconds\n",
               size, t_end - t_start);

    free(send_buf);
    free(recv_buf);
    MPI_Finalize();
    return 0;
}
