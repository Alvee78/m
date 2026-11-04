/*
 * Problem 14 (OpenMP): Parallel Reduction — Sum, Max, Min, Average
 * Each thread works on its local data partition. OpenMP reduction
 * clauses compute global sum, max, min, and average.
 *
 * Compile: gcc -fopenmp -o problem14 problem14_reduce.c
 * Run:     ./problem14
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(void) {
    int data[4][3] = {
        { 10,  20,  30},
        { 40,  50,  60},
        { 70,  80,  90},
        {100, 110, 120}
    };
    int num_threads  = 4;
    int local_count  = 3;
    int total_count  = num_threads * local_count;

    int global_sum = 0;
    int global_max = INT_MIN;
    int global_min = INT_MAX;

    /* Compute local sums first for display */
    for (int t = 0; t < num_threads; t++) {
        int ls = 0;
        for (int j = 0; j < local_count; j++) ls += data[t][j];
        printf("P%d: local sum = %d\n", t, ls);
    }
    printf("\n");

    /* OpenMP parallel reduction */
    #pragma omp parallel num_threads(num_threads) \
        reduction(+:global_sum) \
        reduction(max:global_max) \
        reduction(min:global_min)
    {
        int tid = omp_get_thread_num();
        for (int j = 0; j < local_count; j++) {
            global_sum += data[tid][j];
            if (data[tid][j] > global_max) global_max = data[tid][j];
            if (data[tid][j] < global_min) global_min = data[tid][j];
        }
    }

    printf("Global Sum     = %d\n", global_sum);
    printf("Global Maximum = %d\n", global_max);
    printf("Global Minimum = %d\n", global_min);
    printf("Average        = %.2f\n", (double)global_sum / total_count);

    return 0;
}
