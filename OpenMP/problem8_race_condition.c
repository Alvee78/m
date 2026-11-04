/*
 * Problem 8 (OpenMP): Race Condition and Safe Synchronization
 * Demonstrates a race condition (unsafe version) and a corrected version
 * using OpenMP critical sections and atomic operations.
 *
 * Compile: gcc -fopenmp -o problem8 problem8_race_condition.c
 * Run:     ./problem8
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define ITERATIONS 1000000

int main(void) {
    /* ============================================================
     * UNSAFE VERSION: Two threads increment a shared counter without
     * synchronization — results in a race condition (data corruption).
     * ============================================================ */
    long unsafe_counter = 0;

    printf("Unsafe version (race condition demo):\n");
    printf("Thread 0: Waiting to write shared counter...\n");
    printf("Thread 1: Waiting to write shared counter...\n");

    #pragma omp parallel num_threads(2) shared(unsafe_counter)
    {
        for (int i = 0; i < ITERATIONS; i++)
            unsafe_counter++;   /* DATA RACE — no synchronization! */
    }

    printf("Expected: %d  |  Got (race): %ld  <-- Potential data corruption!\n\n",
           2 * ITERATIONS, unsafe_counter);

    /* ============================================================
     * CORRECTED VERSION: Using atomic or critical sections
     * ============================================================ */
    long safe_counter_atomic   = 0;
    long safe_counter_critical = 0;

    printf("Corrected version (atomic):\n");
    #pragma omp parallel num_threads(2) shared(safe_counter_atomic)
    {
        int tid = omp_get_thread_num();
        for (int i = 0; i < ITERATIONS; i++) {
            #pragma omp atomic
            safe_counter_atomic++;
        }
        #pragma omp barrier
        #pragma omp master
        printf("Thread 0 accumulated, Thread 1 accumulated safely.\n");
    }
    printf("Atomic result: %ld (expected %d)\n\n",
           safe_counter_atomic, 2 * ITERATIONS);

    printf("Corrected version (critical section):\n");
    #pragma omp parallel num_threads(2) shared(safe_counter_critical)
    {
        int tid = omp_get_thread_num();
        for (int i = 0; i < ITERATIONS; i++) {
            #pragma omp critical
            safe_counter_critical++;
        }
    }
    printf("Critical result: %ld (expected %d)\n",
           safe_counter_critical, 2 * ITERATIONS);
    printf("\nCommunication completed without race condition.\n");

    return 0;
}
