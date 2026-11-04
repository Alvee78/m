/*
 * Problem 3 (OpenMP): Ring Topology Communication
 * Simulate a ring topology where each thread passes a value to the next.
 * Uses shared arrays and barriers to mimic ring communication.
 *
 * Compile: gcc -fopenmp -o problem3 problem3_ring_topology.c
 * Run:     ./problem3
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_THREADS 4

int main(void) {
    int mailbox[NUM_THREADS];   /* mailbox[t] = value ready to be received by thread t */
    int received[NUM_THREADS];
    int initial_value = 100;
    double t_start, t_end;

    /* Initialize mailboxes */
    for (int i = 0; i < NUM_THREADS; i++) mailbox[i] = -1;

    t_start = omp_get_wtime();

    /* Phase 1: Thread 0 puts initial value into thread 1's mailbox */
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid  = omp_get_thread_num();
        int next = (tid + 1) % NUM_THREADS;
        int prev = (tid - 1 + NUM_THREADS) % NUM_THREADS;

        /* Thread 0 seeds the ring */
        if (tid == 0) {
            mailbox[next] = initial_value;
            printf("Thread %d sends %d to Thread %d\n", tid, initial_value, next);
        }
        #pragma omp barrier

        /* Each thread (except 0) reads from its mailbox, prints, and forwards */
        if (tid != 0) {
            /* Wait until value is available */
            while (mailbox[tid] == -1) { /* spin wait – fine for demo */ }

            received[tid] = mailbox[tid];
            printf("Thread %d receives %d from Thread %d\n", tid, received[tid], prev);

            if (next != 0) {
                mailbox[next] = received[tid];
                printf("Thread %d sends %d to Thread %d\n", tid, received[tid], next);
            } else {
                /* Return to thread 0 */
                mailbox[0] = received[tid];
                printf("Thread %d sends %d to Thread %d\n", tid, received[tid], 0);
            }
        }

        #pragma omp barrier

        /* Thread 0 receives the value back */
        if (tid == 0) {
            received[0] = mailbox[0];
            printf("Thread %d receives %d from Thread %d\n", tid, received[0], prev);
        }
    }

    t_end = omp_get_wtime();
    printf("\nRing communication completed.\n");
    printf("Total ring time: %.6f seconds\n", t_end - t_start);

    return 0;
}
