/*
 * Problem 18 (OpenMP): Parallel Histogram with Reduction
 * Each thread builds a local histogram of its data partition.
 * Global histogram is computed via atomic updates or critical sections.
 * Equivalent to MPI_Allreduce on a histogram.
 *
 * Compile: gcc -fopenmp -o problem18 problem18_histogram.c
 * Run:     ./problem18
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define BINS        10
#define NUM_THREADS  4

int main(void) {
    /* Data per thread */
    int thread_data[NUM_THREADS][4] = {
        {1, 2, 2, 3},
        {2, 3, 4, 4},
        {1, 4, 5, 5},
        {0, 2, 5, 9}
    };
    int local_n = 4;

    int global_hist[BINS] = {0};

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();

        /* Build local histogram */
        int local_hist[BINS] = {0};
        for (int i = 0; i < local_n; i++) {
            int val = thread_data[tid][i];
            if (val >= 0 && val < BINS)
                local_hist[val]++;
        }

        #pragma omp critical
        {
            printf("P%d local histogram:", tid);
            for (int b = 0; b < BINS; b++)
                if (local_hist[b] > 0) printf("  [%d]=%d", b, local_hist[b]);
            printf("\n");
        }

        /* Merge local into global histogram using atomic (equivalent to Allreduce) */
        for (int b = 0; b < BINS; b++) {
            #pragma omp atomic
            global_hist[b] += local_hist[b];
        }
    }

    /* All threads have correct global_hist after atomic updates */
    int total = 0;
    printf("\nFinal histogram after parallel reduction:\n");
    printf("%-8s %-10s\n", "Value", "Frequency");
    printf("------------------\n");
    for (int b = 0; b < BINS; b++) {
        printf("%-8d %-10d\n", b, global_hist[b]);
        total += global_hist[b];
    }
    printf("\nTotal number of elements = %d\n", total);

    return 0;
}
