/*
 * Problem 13: Uneven Row Distribution with MPI_Scatterv / MPI_Gatherv
 * Distribute matrix rows unevenly (5 rows, 3 processes). Each process
 * multiplies every element by 2. MPI_Gatherv collects results.
 *
 * Compile: mpicc -o problem13 problem13_scatterv_gatherv.c
 * Run:     mpirun -np 3 ./problem13
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TOTAL_ROWS 5
#define COLS       3

void print_matrix_chunk(int *data, int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < COLS; j++)
            printf("%4d", data[i * COLS + j]);
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0)
            fprintf(stderr, "This program requires exactly 3 processes.\n");
        MPI_Finalize();
        return 1;
    }

    /* 5x3 matrix (flattened) */
    int matrix[TOTAL_ROWS * COLS];
    if (rank == 0) {
        int val = 1;
        for (int i = 0; i < TOTAL_ROWS * COLS; i++) matrix[i] = val++;
        printf("Original matrix:\n");
        print_matrix_chunk(matrix, TOTAL_ROWS);
    }

    /*
     * Distribution: P0→2 rows, P1→2 rows, P2→1 row
     * sendcounts and displs are in units of int elements
     */
    int rows_per_proc[3] = {2, 2, 1};
    int sendcounts[3], displs[3];
    int recvcounts[3], rdispls[3];

    int offset = 0;
    for (int p = 0; p < size; p++) {
        sendcounts[p] = rows_per_proc[p] * COLS;
        recvcounts[p] = rows_per_proc[p] * COLS;
        displs[p]     = offset;
        rdispls[p]    = offset;
        offset       += sendcounts[p];
    }

    int my_rows  = rows_per_proc[rank];
    int *local   = (int *)malloc(my_rows * COLS * sizeof(int));
    int *result  = (int *)malloc(my_rows * COLS * sizeof(int));

    MPI_Scatterv(matrix, sendcounts, displs, MPI_INT,
                 local,  my_rows * COLS,      MPI_INT,
                 0, MPI_COMM_WORLD);

    /* Multiply every element by 2 */
    for (int i = 0; i < my_rows * COLS; i++)
        result[i] = local[i] * 2;

    printf("P%d result (%d rows):\n", rank, my_rows);
    print_matrix_chunk(result, my_rows);

    /* Gather results back at root */
    int *gathered = NULL;
    if (rank == 0)
        gathered = (int *)malloc(TOTAL_ROWS * COLS * sizeof(int));

    MPI_Gatherv(result,   my_rows * COLS, MPI_INT,
                gathered, recvcounts, rdispls, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nFinal matrix (all rows × 2):\n");
        print_matrix_chunk(gathered, TOTAL_ROWS);
        free(gathered);
    }

    free(local);
    free(result);
    MPI_Finalize();
    return 0;
}
