/*
 * Problem 6 (OpenMP): Halo Exchange and Stencil Computation
 * Divide a 1D array among threads. Each thread reads halo values from
 * neighboring threads' portions and performs a stencil computation.
 * Stencil: out[i] = left[i] + current[i] + right[i]
 *
 * Compile: gcc -fopenmp -o problem6 problem6_halo_stencil.c
 * Run:     ./problem6
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 4

int main(void) {
    int global_array[] = {10, 20, 30, 40, 50, 60, 70, 80};
    int N     = 8;
    int chunk = N / NUM_THREADS;

    int result[8] = {0};

    /* All threads share global_array; halo is just reading neighbor's elements */
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid   = omp_get_thread_num();
        int start = tid * chunk;
        int end   = start + chunk - 1;

        /* Halo values */
        int halo_left  = (tid > 0)               ? global_array[start - 1] : 0;
        int halo_right = (tid < NUM_THREADS - 1) ? global_array[end + 1]   : 0;

        #pragma omp critical
        {
            if (tid == 0)
                printf("P%d: [%d %d] + right halo %d\n",
                       tid, global_array[start], global_array[end], halo_right);
            else if (tid == NUM_THREADS - 1)
                printf("P%d: left halo %d, [%d %d]\n",
                       tid, halo_left, global_array[start], global_array[end]);
            else
                printf("P%d: left halo %d, [%d %d], right halo %d\n",
                       tid, halo_left, global_array[start], global_array[end], halo_right);
        }

        /* Stencil computation */
        for (int i = start; i <= end; i++) {
            int left  = (i == start) ? halo_left  : global_array[i - 1];
            int right = (i == end)   ? halo_right : global_array[i + 1];
            result[i] = left + global_array[i] + right;
        }
    }

    printf("\nStencil result:\n");
    for (int i = 0; i < N; i++) printf("%d ", result[i]);
    printf("\n");

    return 0;
}
