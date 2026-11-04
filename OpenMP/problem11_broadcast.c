/*
 * Problem 11 (OpenMP): Broadcast Configuration Parameters
 * Master thread sets configuration; all threads use it.
 * Equivalent to MPI_Bcast but via shared memory.
 *
 * Compile: gcc -fopenmp -o problem11 problem11_broadcast.c
 * Run:     ./problem11
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int iterations;
    int threshold;
    int block_size;
} Config;

int main(void) {
    Config cfg;       /* Shared among all threads */
    int num_threads = 4;

    /* Master initializes (equivalent to root in MPI_Bcast) */
    cfg.iterations = 1000;
    cfg.threshold  = 50;
    cfg.block_size = 256;

    printf("Root broadcasting configuration...\n\n");

    #pragma omp parallel num_threads(num_threads) shared(cfg)
    {
        int tid = omp_get_thread_num();

        /* All threads see the same cfg (shared memory broadcast) */
        #pragma omp barrier   /* ensure cfg is fully initialized */

        #pragma omp critical
        printf("P%d received: Iterations = %d, Threshold = %d, Block size = %d\n",
               tid, cfg.iterations, cfg.threshold, cfg.block_size);

        /* Each thread performs computation using cfg */
        long long local_result = 0;
        for (int i = tid; i < cfg.iterations; i += num_threads) {
            if ((i % cfg.block_size) < cfg.threshold)
                local_result += i;
        }

        #pragma omp critical
        printf("P%d: local computation result = %lld\n", tid, local_result);
    }

    return 0;
}
