/*
 * Problem 18: Distributed Histogram with MPI_Allreduce
 * Each process builds a local histogram of its data. MPI_Allreduce
 * combines all local histograms into a global histogram.
 *
 * Compile: mpicc -o problem18 problem18_histogram.c
 * Run:     mpirun -np 4 ./problem18
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define BINS    10   /* values 0..9 */

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Predefined data per process */
    int p_data[4][4] = {
        {1, 2, 2, 3},
        {2, 3, 4, 4},
        {1, 4, 5, 5},
        {0, 2, 5, 9}
    };
    int local_n    = 4;
    int *local_data = p_data[rank];

    /* Build local histogram */
    int local_hist[BINS] = {0};
    for (int i = 0; i < local_n; i++) {
        if (local_data[i] >= 0 && local_data[i] < BINS)
            local_hist[local_data[i]]++;
    }

    printf("P%d local histogram:", rank);
    for (int b = 0; b < BINS; b++) {
        if (local_hist[b] > 0) printf("  [%d]=%d", b, local_hist[b]);
    }
    printf("\n");

    /* Combine all local histograms with MPI_Allreduce */
    int global_hist[BINS] = {0};
    MPI_Allreduce(local_hist, global_hist, BINS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    /* All processes now have the global histogram */
    if (rank == 0) {
        int total = 0;
        printf("\nFinal histogram after MPI_Allreduce:\n");
        printf("%-8s %-10s\n", "Value", "Frequency");
        printf("------------------\n");
        for (int b = 0; b < BINS; b++) {
            printf("%-8d %-10d\n", b, global_hist[b]);
            total += global_hist[b];
        }
        printf("\nTotal number of elements = %d\n", total);
    }

    MPI_Finalize();
    return 0;
}
