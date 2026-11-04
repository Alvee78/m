/*
 * Problem 14: Global Reduction — Sum, Max, Min, Average
 * Each process generates numerical values. MPI_Reduce computes global
 * sum, maximum, minimum, and average.
 *
 * Compile: mpicc -o problem14 problem14_reduce.c
 * Run:     mpirun -np 4 ./problem14
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Predefined data per process */
    int data[4][3] = {
        { 10,  20,  30},
        { 40,  50,  60},
        { 70,  80,  90},
        {100, 110, 120}
    };
    int local_count = 3;
    int *local_data = data[rank];

    /* Compute local sum */
    int local_sum = 0;
    int local_max = local_data[0];
    int local_min = local_data[0];
    for (int i = 0; i < local_count; i++) {
        local_sum += local_data[i];
        if (local_data[i] > local_max) local_max = local_data[i];
        if (local_data[i] < local_min) local_min = local_data[i];
    }

    printf("P%d: local sum = %d, max = %d, min = %d\n",
           rank, local_sum, local_max, local_min);

    /* Global reductions */
    int global_sum, global_max, global_min;
    int total_elements = local_count * size;

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nGlobal Sum     = %d\n", global_sum);
        printf("Global Maximum = %d\n", global_max);
        printf("Global Minimum = %d\n", global_min);
        printf("Average        = %.2f\n", (double)global_sum / total_elements);
    }

    MPI_Finalize();
    return 0;
}
