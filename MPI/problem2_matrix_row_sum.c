/*
 * Problem 2: Matrix Row Distribution and Sum
 * Root distributes different rows to worker processes using MPI_Send.
 * Workers calculate the row sum and return results via MPI_Recv.
 *
 * Compile: mpicc -o problem2 problem2_matrix_row_sum.c
 * Run:     mpirun -np 4 ./problem2
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define COLS 4
#define ROWS 4

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int matrix[ROWS][COLS] = {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    int row[COLS];
    int row_sum = 0;
    MPI_Status status;

    if (rank == 0) {
        /* Root: send each row to corresponding process (including itself) */
        for (int p = 0; p < size; p++) {
            if (p == 0) {
                /* Root handles its own row */
                for (int j = 0; j < COLS; j++) row[j] = matrix[0][j];
            } else {
                MPI_Send(matrix[p], COLS, MPI_INT, p, 0, MPI_COMM_WORLD);
            }
        }

        /* Compute own sum */
        for (int j = 0; j < COLS; j++) row_sum += row[j];
        printf("P0: Row =");
        for (int j = 0; j < COLS; j++) printf(" %d", row[j]);
        printf(" Sum = %d\n", row_sum);

        /* Collect sums from workers */
        int total = row_sum;
        for (int p = 1; p < size; p++) {
            int recv_sum;
            MPI_Recv(&recv_sum, 1, MPI_INT, p, 1, MPI_COMM_WORLD, &status);
            total += recv_sum;
        }
        printf("\nTotal matrix sum = %d\n", total);

    } else {
        /* Worker: receive row, compute sum, send back */
        MPI_Recv(row, COLS, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        for (int j = 0; j < COLS; j++) row_sum += row[j];

        printf("P%d: Row =", rank);
        for (int j = 0; j < COLS; j++) printf(" %d", row[j]);
        printf(" Sum = %d\n", row_sum);

        MPI_Send(&row_sum, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
