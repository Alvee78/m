/*
 * Problem 6: Halo Exchange and Stencil Computation
 * Divide a 1D array among processes. Exchange halo/boundary values with
 * neighbors using MPI_Send/MPI_Recv, then perform a stencil computation.
 * Stencil: out[i] = left[i] + current[i] + right[i]
 *
 * Compile: mpicc -o problem6 problem6_halo_stencil.c
 * Run:     mpirun -np 4 ./problem6
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int global_array[] = {10, 20, 30, 40, 50, 60, 70, 80};
    int N = 8;
    int chunk = N / size;   /* elements per process */

    /* Local data */
    int *local = (int *)malloc(chunk * sizeof(int));
    for (int i = 0; i < chunk; i++)
        local[i] = global_array[rank * chunk + i];

    int halo_left  = 0;   /* received from left  neighbor */
    int halo_right = 0;   /* received from right neighbor */
    MPI_Status status;

    /* ---- Halo exchange ---- */
    /* Send right boundary, receive from right neighbor */
    if (rank < size - 1) {
        MPI_Send(&local[chunk - 1], 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
    if (rank > 0) {
        MPI_Recv(&halo_left, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
    }

    /* Send left boundary, receive from left neighbor */
    if (rank > 0) {
        MPI_Send(&local[0], 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD);
    }
    if (rank < size - 1) {
        MPI_Recv(&halo_right, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Print halo exchange results */
    if (rank == 0)
        printf("After halo exchange:\n");
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        printf("P0: [%d %d] + right halo %d\n", local[0], local[1], halo_right);
    else if (rank == size - 1)
        printf("P%d: left halo %d, [%d %d]\n", rank, halo_left, local[0], local[1]);
    else
        printf("P%d: left halo %d, [%d %d], right halo %d\n",
               rank, halo_left, local[0], local[1], halo_right);

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Stencil computation ---- */
    int *result = (int *)malloc(chunk * sizeof(int));
    for (int i = 0; i < chunk; i++) {
        int left  = (i == 0)          ? halo_left  : local[i - 1];
        int right = (i == chunk - 1)  ? halo_right : local[i + 1];
        /* Boundary processes: missing side treated as 0 */
        if (rank == 0 && i == 0)          left  = 0;
        if (rank == size - 1 && i == chunk - 1) right = 0;
        result[i] = left + local[i] + right;
    }

    /* Gather all results at root */
    int *global_result = NULL;
    if (rank == 0)
        global_result = (int *)malloc(N * sizeof(int));

    MPI_Gather(result, chunk, MPI_INT, global_result, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* Expected: 30 60 90 120 150 180 210 230 (boundary elements get 0 on missing side) */
        printf("\nStencil result:\n");
        for (int i = 0; i < N; i++) printf("%d ", global_result[i]);
        printf("\n");
        free(global_result);
    }

    free(local);
    free(result);
    MPI_Finalize();
    return 0;
}
