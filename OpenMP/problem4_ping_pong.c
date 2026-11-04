/*
 * Problem 4 (OpenMP): Ping-Pong Latency Simulation
 * Simulate ping-pong between two threads using shared memory.
 * Measure round-trip time for various "message sizes" (buffer copies).
 *
 * Compile: gcc -fopenmp -o problem4 problem4_ping_pong.c
 * Run:     ./problem4
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITERATIONS  5

int sizes[] = {64, 256, 1024, 4096, 16384, 65536, 1048576};
int num_sizes = 7;

int main(void) {
    /* Shared buffers */
    char *buf0 = NULL;   /* thread 0 → thread 1 */
    char *buf1 = NULL;   /* thread 1 → thread 0 */
    int   flag = 0;      /* 0 = thread 0's turn, 1 = thread 1's turn */

    for (int s = 0; s < num_sizes; s++) {
        int msg_size = sizes[s];
        buf0 = (char *)calloc(msg_size, 1);
        buf1 = (char *)calloc(msg_size, 1);
        flag = 0;

        double t_start = 0, t_end = 0;

        #pragma omp parallel num_threads(2)
        {
            int tid = omp_get_thread_num();

            if (tid == 0) {
                t_start = omp_get_wtime();
                for (int i = 0; i < ITERATIONS; i++) {
                    /* Send: fill buf0 */
                    memset(buf0, (char)i, msg_size);
                    #pragma omp atomic write
                    flag = 1;                   /* signal thread 1 */

                    /* Wait for pong */
                    int f;
                    do { #pragma omp atomic read
                         f = flag; } while (f != 0);
                }
                t_end = omp_get_wtime();
            } else {
                for (int i = 0; i < ITERATIONS; i++) {
                    /* Wait for ping */
                    int f;
                    do { #pragma omp atomic read
                         f = flag; } while (f != 1);

                    /* Pong: copy back */
                    memcpy(buf1, buf0, msg_size);
                    #pragma omp atomic write
                    flag = 0;
                }
            }
        }

        double elapsed    = t_end - t_start;
        double avg_rtt    = elapsed / ITERATIONS;
        double avg_oneway = avg_rtt / 2.0;
        double bandwidth  = (msg_size * 2.0 * ITERATIONS) / elapsed / (1024.0 * 1024.0);

        printf("Message size  : %d bytes\n", msg_size);
        printf("Iterations    : %d\n", ITERATIONS);
        printf("Total round trips: %d\n", ITERATIONS);
        printf("Average round-trip: %.6f seconds\n", avg_rtt);
        printf("Average one-way  : %.6f seconds\n", avg_oneway);
        printf("Effective bandwidth: %.2f MB/s\n", bandwidth);
        printf("------------------------------------------\n");

        free(buf0); free(buf1);
    }

    return 0;
}
