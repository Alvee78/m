/*
 * Problem 5: Distributed Matrix Multiplication
 * Root distributes rows of matrix A to worker processes. Each process
 * multiplies its rows with the full matrix B and sends results back.
 *
 * Compile: mpicc -o problem5 problem5_matrix_multiply.c
 * Run:     mpirun -np 2 ./problem5
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 2  /* Matrix dimension NxN */

void print_matrix(const char *label, int mat[N][N]) {
    printf("%s\n", label);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%4d", mat[i][j]);
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int A[N][N] = {{1, 2}, {3, 4}};
    int B[N][N] = {{5, 6}, {7, 8}};
    int C[N][N] = {0};

    int rows_per_proc = N / size;
    int local_A[N][N];  /* local rows of A */
    int local_C[N][N];  /* local rows of C */
    MPI_Status status;

    if (rank == 0) {
        /* Root sends rows of A to each process */
        for (int p = 1; p < size; p++) {
            int start_row = p * rows_per_proc;
            MPI_Send(&A[start_row][0], rows_per_proc * N, MPI_INT, p, 0, MPI_COMM_WORLD);
            MPI_Send(&B[0][0],         N * N,             MPI_INT, p, 1, MPI_COMM_WORLD);
        }

        /* Root computes its own rows */
        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < N; j++) {
                local_C[i][j] = 0;
                for (int k = 0; k < N; k++)
                    local_C[i][j] += A[i][k] * B[k][j];
                C[i][j] = local_C[i][j];
            }
        }

        /* Receive results from workers */
        for (int p = 1; p < size; p++) {
            int start_row = p * rows_per_proc;
            MPI_Recv(&C[start_row][0], rows_per_proc * N, MPI_INT, p, 2, MPI_COMM_WORLD, &status);
        }

        print_matrix("Matrix A:", A);
        print_matrix("Matrix B:", B);
        print_matrix("Matrix C = A x B:", C);

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
    } else {
        int local_B[N][N];
        int local_rows[N][N];
        int result[N][N];

        MPI_Recv(&local_rows[0][0], rows_per_proc * N, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&local_B[0][0],    N * N,             MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < N; j++) {
                result[i][j] = 0;
                for (int k = 0; k < N; k++)
                    result[i][j] += local_rows[i][k] * local_B[k][j];
            }
        }

        MPI_Send(&result[0][0], rows_per_proc * N, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
