/*
 * Problem 1: Boundary Element Exchange
 * Divide a large integer array among MPI processes. Each process exchanges
 * its boundary elements with neighboring processes using point-to-point
 * communication and computes a local result.
 *
 * Compile: mpicc -o problem1 problem1_boundary_exchange.c
 * Run:     mpirun -np 4 ./problem1
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 12;
    int global_array[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    int chunk = N / size;

    /* Each process gets its local chunk */
    int *local = (int *)malloc(chunk * sizeof(int));
    for (int i = 0; i < chunk; i++)
        local[i] = global_array[rank * chunk + i];

    printf("P%d: local data =", rank);
    for (int i = 0; i < chunk; i++) printf(" %d", local[i]);
    printf("\n");
    MPI_Barrier(MPI_COMM_WORLD);

    int left_boundary  = local[0];          /* leftmost element  */
    int right_boundary = local[chunk - 1];  /* rightmost element */

    int recv_left = -1, recv_right = -1;
    MPI_Status status;

    /* Send right boundary to right neighbor, receive left boundary from left neighbor */
    if (rank < size - 1) {
        MPI_Send(&right_boundary, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        printf("P%d: Right boundary sent = %d\n", rank, right_boundary);
    }
    if (rank > 0) {
        MPI_Recv(&recv_left, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        printf("P%d: Left boundary received = %d\n", rank, recv_left);
    }

    /* Send left boundary to left neighbor, receive right boundary from right neighbor */
    if (rank > 0) {
        MPI_Send(&left_boundary, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD);
    }
    if (rank < size - 1) {
        MPI_Recv(&recv_right, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("\nBoundary exchange completed successfully.\n");

    free(local);
    MPI_Finalize();
    return 0;
}
