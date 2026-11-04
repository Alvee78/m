/*
 * Problem 1 (OpenMP): Boundary Element Exchange
 * Divide a large integer array among threads. Each thread exchanges its
 * boundary elements with neighboring threads and computes a local result.
 *
 * Compile: gcc -fopenmp -o problem1 problem1_boundary_exchange.c
 * Run:     ./problem1
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    int N          = 12;
    int num_threads = 4;
    int global_array[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    int chunk = N / num_threads;

    /* Shared boundary arrays: boundary_right[t] = right boundary of thread t */
    int *boundary_right = (int *)calloc(num_threads, sizeof(int));
    int *boundary_left  = (int *)calloc(num_threads, sizeof(int));

    /* Phase 1: Each thread records its boundaries */
    #pragma omp parallel num_threads(num_threads)
    {
        int tid   = omp_get_thread_num();
        int start = tid * chunk;
        int end   = start + chunk - 1;

        boundary_right[tid] = global_array[end];
        boundary_left[tid]  = global_array[start];

        #pragma omp barrier  /* wait until all boundaries are set */

        /* Phase 2: Exchange with neighbors */
        int recv_left  = (tid > 0)              ? boundary_right[tid - 1] : -1;
        int recv_right = (tid < num_threads - 1) ? boundary_left[tid + 1]  : -1;

        /* Print local data */
        #pragma omp critical
        {
            printf("Thread %d: local data =", tid);
            for (int i = start; i <= end; i++) printf(" %d", global_array[i]);
            printf("\n");
        }

        #pragma omp barrier

        if (tid < num_threads - 1) {
            #pragma omp critical
            printf("Thread %d: Right boundary sent = %d\n",
                   tid, boundary_right[tid]);
        }
        if (tid > 0) {
            #pragma omp critical
            printf("Thread %d: Left boundary received = %d\n",
                   tid, recv_left);
        }
    }

    printf("\nBoundary exchange completed successfully.\n");

    free(boundary_right);
    free(boundary_left);
    return 0;
}
