/*
 * Problem 16 (OpenMP): Prefix Sum (Parallel Scan)
 * Compute inclusive prefix sum in parallel using OpenMP.
 * Also implements a manual sequential version for comparison.
 *
 * Compile: gcc -fopenmp -o problem16 problem16_prefix_scan.c
 * Run:     ./problem16
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_THREADS 4

int main(void) {
    int local_vals[NUM_THREADS] = {10, 20, 30, 40};
    int prefix_parallel[NUM_THREADS];
    int prefix_sequential[NUM_THREADS];

    /* ============ Sequential prefix sum ============ */
    prefix_sequential[0] = local_vals[0];
    for (int i = 1; i < NUM_THREADS; i++)
        prefix_sequential[i] = prefix_sequential[i - 1] + local_vals[i];

    printf("Sequential prefix sum:\n");
    for (int i = 0; i < NUM_THREADS; i++)
        printf("P%d: Local value = %d, Prefix sum = %d\n",
               i, local_vals[i], prefix_sequential[i]);

    /* ============ Parallel prefix sum (Hillis-Steele algorithm) ============ */
    memcpy(prefix_parallel, local_vals, NUM_THREADS * sizeof(int));
    int temp[NUM_THREADS];

    for (int step = 1; step < NUM_THREADS; step *= 2) {
        memcpy(temp, prefix_parallel, NUM_THREADS * sizeof(int));
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = step; i < NUM_THREADS; i++)
            prefix_parallel[i] = temp[i] + temp[i - step];
    }

    printf("\nParallel (Hillis-Steele) prefix sum:\n");
    for (int i = 0; i < NUM_THREADS; i++)
        printf("P%d: Local value = %d, Prefix sum = %d\n",
               i, local_vals[i], prefix_parallel[i]);

    printf("\nPrefix result:");
    for (int i = 0; i < NUM_THREADS; i++) printf(" %d", prefix_parallel[i]);
    printf("\n");

    return 0;
}
