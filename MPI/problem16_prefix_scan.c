/*
 * Problem 16: Prefix Sum using MPI_Scan
 * Each process stores a local value. MPI_Scan computes inclusive prefix sum.
 * Also implements the equivalent point-to-point version for comparison.
 *
 * Compile: mpicc -o problem16 problem16_prefix_scan.c
 * Run:     mpirun -np 4 ./problem16
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Local values: P0=10, P1=20, P2=30, P3=40 */
    int local_val = (rank + 1) * 10;

    /* ============ MPI_Scan approach ============ */
    int prefix_scan;
    MPI_Scan(&local_val, &prefix_scan, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("[MPI_Scan] P%d: Local value = %d, Prefix sum = %d\n",
           rank, local_val, prefix_scan);

    MPI_Barrier(MPI_COMM_WORLD);

    /* ============ Point-to-point approach ============ */
    int prefix_p2p = local_val;
    MPI_Status status;

    for (int step = 1; step < size; step *= 2) {
        if (rank >= step) {
            int recv_val;
            MPI_Sendrecv(&prefix_p2p, 1, MPI_INT, rank - step, 0,
                         &recv_val,   1, MPI_INT, rank - step, 0,
                         MPI_COMM_WORLD, &status);
            prefix_p2p += recv_val;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Print results */
    if (rank == 0)
        printf("\nPrefix result (MPI_Scan): ");
    MPI_Barrier(MPI_COMM_WORLD);

    /* Gather all prefix sums to root for display */
    int *all_prefix = NULL;
    if (rank == 0) all_prefix = (int *)malloc(size * sizeof(int));
    MPI_Gather(&prefix_scan, 1, MPI_INT, all_prefix, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < size; i++) printf("%d ", all_prefix[i]);
        printf("\n");
        free(all_prefix);
    }

    MPI_Finalize();
    return 0;
}
