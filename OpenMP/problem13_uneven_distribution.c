/*
 * Problem 13 (OpenMP): Uneven Row Distribution
 * Distribute matrix rows unevenly among threads (equivalent to MPI_Scatterv).
 * Each thread multiplies its rows by 2. Results are gathered in shared memory.
 *
 * Compile: gcc -fopenmp -o problem13 problem13_uneven_distribution.c
 * Run:     ./problem13
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define TOTAL_ROWS 5
#define COLS       3
#define NUM_THREADS 3

void print_rows(int *data, int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++) printf("%4d", data[i * COLS + j]);
        printf("\n");
    }
}

int main(void) {
    int matrix[TOTAL_ROWS][COLS] = {
        { 1,  2,  3},
        { 4,  5,  6},
        { 7,  8,  9},
        {10, 11, 12},
        {13, 14, 15}
    };
    int result[TOTAL_ROWS][COLS] = {0};

    /* Uneven distribution: P0→rows 0-1, P1→rows 2-3, P2→row 4 */
    int row_start[NUM_THREADS] = {0, 2, 4};
    int row_count[NUM_THREADS] = {2, 2, 1};

    printf("Original matrix:\n");
    print_rows(&matrix[0][0], TOTAL_ROWS);
    printf("\n");

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid   = omp_get_thread_num();
        int start = row_start[tid];
        int count = row_count[tid];

        /* Multiply each element by 2 */
        for (int i = start; i < start + count; i++)
            for (int j = 0; j < COLS; j++)
                result[i][j] = matrix[i][j] * 2;

        #pragma omp critical
        {
            printf("P%d result (%d rows):\n", tid, count);
            print_rows(&result[start][0], count);
        }
    }

    printf("\nFinal matrix (all rows × 2):\n");
    print_rows(&result[0][0], TOTAL_ROWS);

    return 0;
}
