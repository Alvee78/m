/*
 * Problem 5 (OpenMP): Parallel Matrix Multiplication
 * Distribute rows of matrix A among threads. Each thread computes its
 * portion of the result matrix C = A x B.
 *
 * Compile: gcc -fopenmp -o problem5 problem5_matrix_multiply.c
 * Run:     ./problem5
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 2   /* Matrix dimension */

void print_matrix(const char *label, int mat[N][N]) {
    printf("%s\n", label);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%4d", mat[i][j]);
        printf("\n");
    }
}

int main(void) {
    int A[N][N] = {{1, 2}, {3, 4}};
    int B[N][N] = {{5, 6}, {7, 8}};
    int C[N][N] = {{0, 0}, {0, 0}};

    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf("%4d", A[i][j]); printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf("%4d", B[i][j]); printf("\n");
    }

    /* Parallel matrix multiplication — each thread computes a row of C */
    #pragma omp parallel for num_threads(N) schedule(static, 1)
    for (int i = 0; i < N; i++) {
        int tid = omp_get_thread_num();
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
        #pragma omp critical
        printf("Thread %d computed row %d of C\n", tid, i);
    }

    printf("\nMatrix C = A x B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf("%4d", C[i][j]); printf("\n");
    }

    printf("\nDetailed computation:\n");
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            printf("C[%d][%d] = ", i, j);
            for (int k = 0; k < N; k++) {
                printf("%d×%d", A[i][k], B[k][j]);
                if (k < N - 1) printf(" + ");
            }
            printf(" = %d\n", C[i][j]);
        }

    return 0;
}
