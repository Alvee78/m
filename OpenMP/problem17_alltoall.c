/*
 * Problem 17 (OpenMP): All-to-All Data Exchange
 * Each thread generates data for every other thread. After parallel
 * exchange, each thread holds one value from every other thread.
 * Equivalent to MPI_Alltoall.
 *
 * Compile: gcc -fopenmp -o problem17 problem17_alltoall.c
 * Run:     ./problem17
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 4

int main(void) {
    /*
     * send_data[i][j] = value that thread i wants to send to thread j
     * recv_data[i][j] = value that thread i receives from thread j
     */
    int send_data[NUM_THREADS][NUM_THREADS];
    int recv_data[NUM_THREADS][NUM_THREADS];

    double t_start, t_end;

    /* Initialize send data */
    for (int i = 0; i < NUM_THREADS; i++)
        for (int j = 0; j < NUM_THREADS; j++)
            send_data[i][j] = i * 10 + j;

    printf("Send data:\n");
    for (int i = 0; i < NUM_THREADS; i++) {
        printf("P%d sends:", i);
        for (int j = 0; j < NUM_THREADS; j++) printf(" %d", send_data[i][j]);
        printf("\n");
    }

    t_start = omp_get_wtime();

    /* All-to-all: thread i reads send_data[j][i] from thread j */
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        #pragma omp barrier   /* ensure send_data fully initialized */

        for (int src = 0; src < NUM_THREADS; src++)
            recv_data[tid][src] = send_data[src][tid];
    }

    t_end = omp_get_wtime();

    printf("\nAfter All-to-All exchange:\n");
    for (int i = 0; i < NUM_THREADS; i++) {
        printf("P%d received:", i);
        for (int j = 0; j < NUM_THREADS; j++) printf(" %d", recv_data[i][j]);
        printf("\n");
    }

    printf("\nAll-to-All with %d threads took %.6f seconds\n",
           NUM_THREADS, t_end - t_start);

    return 0;
}
