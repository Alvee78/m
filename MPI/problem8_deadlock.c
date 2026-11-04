/*
 * Problem 8: Deadlock Detection and Avoidance
 * Demonstrates a deadlock scenario with blocking sends and shows the
 * corrected version using MPI_Sendrecv or reordering.
 *
 * Compile: mpicc -o problem8 problem8_deadlock.c
 * Run:     mpirun -np 2 ./problem8
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            fprintf(stderr, "This program requires exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    int send_val = (rank == 0) ? 100 : 200;
    int recv_val = -1;
    MPI_Status status;

    /* ============================================================
     * UNSAFE VERSION (deadlock when buffer is too small to buffer)
     * Both processes send before receiving → potential deadlock
     * We simulate this with a comment — actually running it would hang.
     * ============================================================ */
    if (rank == 0) {
        printf("Unsafe version (simulated description):\n");
        printf("P0: Waiting to send %d\n", send_val);
        printf("P1: Waiting to send 200\n");
        printf("Potential deadlock detected.\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ============================================================
     * CORRECTED VERSION using MPI_Sendrecv (atomic send+receive)
     * ============================================================ */
    if (rank == 0) {
        printf("Corrected version (using MPI_Sendrecv):\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Sendrecv(
        &send_val, 1, MPI_INT, 1 - rank, 0,    /* send to other   */
        &recv_val, 1, MPI_INT, 1 - rank, 0,    /* recv from other */
        MPI_COMM_WORLD, &status
    );

    if (rank == 0) {
        printf("P0 sent %d\n", send_val);
        printf("P0 received %d\n", recv_val);
    } else {
        printf("P1 sent %d\n", send_val);
        printf("P1 received %d\n", recv_val);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("\nCommunication completed without deadlock.\n");

    /* ============================================================
     * ALTERNATIVE CORRECTED VERSION: reorder send/recv by rank
     * ============================================================ */
    if (rank == 0) {
        MPI_Send(&send_val, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
        MPI_Recv(&recv_val, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &status);
    } else {
        MPI_Recv(&recv_val, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Send(&send_val, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("Alternative reordered version also completed without deadlock.\n");

    MPI_Finalize();
    return 0;
}
