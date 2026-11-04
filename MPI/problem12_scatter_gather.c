/*
 * Problem 12: MPI_Scatter and MPI_Gather — Element-wise Power
 * Distribute a vector with MPI_Scatter. Each process squares its elements.
 * MPI_Gather collects all results at root.
 *
 * Compile: mpicc -o problem12 problem12_scatter_gather.c -lm
 * Run:     mpirun -np 4 ./problem12
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 8;
    int chunk = N / size;

    int *global_input  = NULL;
    int *global_output = NULL;
    int *local_input   = (int *)malloc(chunk * sizeof(int));
    int *local_output  = (int *)malloc(chunk * sizeof(int));

    if (rank == 0) {
        global_input  = (int *)malloc(N * sizeof(int));
        global_output = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) global_input[i] = i + 1;  /* 1..8 */

        printf("Input vector:");
        for (int i = 0; i < N; i++) printf(" %d", global_input[i]);
        printf("\n");
    }

    /* Distribute N/size elements to each process */
    MPI_Scatter(global_input, chunk, MPI_INT,
                local_input,  chunk, MPI_INT, 0, MPI_COMM_WORLD);

    /* Each process squares its elements */
    printf("P%d computing squares of:", rank);
    for (int i = 0; i < chunk; i++) {
        printf(" %d", local_input[i]);
        local_output[i] = local_input[i] * local_input[i];
    }
    printf("\n");

    printf("P%d result:", rank);
    for (int i = 0; i < chunk; i++) printf(" %d", local_output[i]);
    printf("\n");

    /* Collect results at root */
    MPI_Gather(local_output, chunk, MPI_INT,
               global_output, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nGathered result:");
        for (int i = 0; i < N; i++) printf(" %d", global_output[i]);
        printf("\n");
        free(global_input);
        free(global_output);
    }

    free(local_input);
    free(local_output);
    MPI_Finalize();
    return 0;
}
