/*
 * Problem 12 (OpenMP): Scatter-Gather — Element-wise Squaring
 * Distribute vector chunks among threads (scatter), each thread squares
 * its elements, then master collects results (gather).
 *
 * Compile: gcc -fopenmp -o problem12 problem12_scatter_gather.c
 * Run:     ./problem12
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int N           = 8;
    int num_threads = 4;
    int chunk       = N / num_threads;

    int input[]  = {1, 2, 3, 4, 5, 6, 7, 8};
    int output[8] = {0};

    printf("Input vector:");
    for (int i = 0; i < N; i++) printf(" %d", input[i]);
    printf("\n\n");

    #pragma omp parallel num_threads(num_threads)
    {
        int tid   = omp_get_thread_num();
        int start = tid * chunk;

        /* Each thread prints its "scattered" portion and squares elements */
        #pragma omp critical
        {
            printf("P%d computing squares of:", tid);
            for (int i = start; i < start + chunk; i++) printf(" %d", input[i]);
            printf("\n");
        }

        for (int i = start; i < start + chunk; i++)
            output[i] = input[i] * input[i];

        #pragma omp critical
        {
            printf("P%d result:", tid);
            for (int i = start; i < start + chunk; i++) printf(" %d", output[i]);
            printf("\n");
        }
    }

    /* Master prints gathered result */
    printf("\nGathered result:");
    for (int i = 0; i < N; i++) printf(" %d", output[i]);
    printf("\n");

    return 0;
}
