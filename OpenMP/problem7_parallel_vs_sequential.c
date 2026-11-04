/*
 * Problem 7 (OpenMP): Parallel vs. Sequential Vector Exchange Timing
 * Compare parallel (OpenMP) vector processing vs. sequential processing.
 * Equivalent to the MPI nonblocking vs. blocking comparison.
 *
 * Compile: gcc -fopenmp -o problem7 problem7_parallel_vs_sequential.c
 * Run:     ./problem7
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_SIZE 8
#define REPEAT     100000

int main(void) {
    int p0_data[] = {10, 20, 30, 40, 50, 60, 70, 80};
    int p1_data[] = { 1,  2,  3,  4,  5,  6,  7,  8};

    int recv_by_p0[ARRAY_SIZE];   /* will hold p1_data */
    int recv_by_p1[ARRAY_SIZE];   /* will hold p0_data */
    int temp0[ARRAY_SIZE], temp1[ARRAY_SIZE];

    /* ======== Sequential (blocking equivalent) ======== */
    double t_seq_start = omp_get_wtime();
    for (int r = 0; r < REPEAT; r++) {
        memcpy(temp0, p0_data, ARRAY_SIZE * sizeof(int));
        memcpy(temp1, p1_data, ARRAY_SIZE * sizeof(int));
        memcpy(recv_by_p0, temp1, ARRAY_SIZE * sizeof(int));
        memcpy(recv_by_p1, temp0, ARRAY_SIZE * sizeof(int));
    }
    double t_seq_end = omp_get_wtime();

    printf("Sequential (blocking equivalent):\n");
    printf("  P0 received:");
    for (int i = 0; i < ARRAY_SIZE; i++) printf(" %d", recv_by_p0[i]);
    printf("\n");
    printf("  P1 received:");
    for (int i = 0; i < ARRAY_SIZE; i++) printf(" %d", recv_by_p1[i]);
    printf("\n");

    /* ======== Parallel (nonblocking equivalent) ======== */
    double t_par_start = omp_get_wtime();
    for (int r = 0; r < REPEAT; r++) {
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            { memcpy(recv_by_p0, p1_data, ARRAY_SIZE * sizeof(int)); }

            #pragma omp section
            { memcpy(recv_by_p1, p0_data, ARRAY_SIZE * sizeof(int)); }
        }
    }
    double t_par_end = omp_get_wtime();

    printf("\nParallel (nonblocking equivalent):\n");
    printf("  P0 received:");
    for (int i = 0; i < ARRAY_SIZE; i++) printf(" %d", recv_by_p0[i]);
    printf("\n");
    printf("  P1 received:");
    for (int i = 0; i < ARRAY_SIZE; i++) printf(" %d", recv_by_p1[i]);
    printf("\n\nNonblocking communication completed.\n\n");

    printf("Sample performance (averaged over %d iterations):\n", REPEAT);
    printf("Sequential (blocking) time    = %.6f s\n",
           (t_seq_end - t_seq_start) / REPEAT);
    printf("Parallel (nonblocking) time   = %.6f s\n",
           (t_par_end - t_par_start) / REPEAT);

    return 0;
}
