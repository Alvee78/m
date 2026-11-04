/*
 * Problem 2 (OpenMP): Matrix Row Distribution and Sum
 * Distribute matrix rows among threads. Each thread computes its row sum.
 * The master thread collects and prints the total.
 *
 * Compile: gcc -fopenmp -o problem2 problem2_matrix_row_sum.c
 * Run:     ./problem2
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 4
#define COLS 4

int main(void) {
    int matrix[ROWS][COLS] = {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    int row_sums[ROWS] = {0};
    int total_sum = 0;

    #pragma omp parallel for num_threads(ROWS) schedule(static, 1)
    for (int i = 0; i < ROWS; i++) {
        int tid = omp_get_thread_num();
        int sum = 0;
        for (int j = 0; j < COLS; j++)
            sum += matrix[i][j];
        row_sums[i] = sum;

        #pragma omp critical
        {
            printf("P%d: Row =", tid);
            for (int j = 0; j < COLS; j++) printf(" %d", matrix[i][j]);
            printf(" Sum = %d\n", sum);
        }
    }

    /* Total sum */
    for (int i = 0; i < ROWS; i++) total_sum += row_sums[i];
    printf("\nTotal matrix sum = %d\n", total_sum);

    return 0;
}
