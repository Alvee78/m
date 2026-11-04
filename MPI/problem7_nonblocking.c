/*
 * Problem 7: Nonblocking vs. Blocking Communication
 * Implement distributed vector exchange using MPI_Isend/MPI_Irecv and compare
 * execution time with blocking MPI_Send/MPI_Recv.
 *
 * Compile: mpicc -o problem7 problem7_nonblocking.c
 * Run:     mpirun -np 2 ./problem7
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 8
#define REPEAT     1000   /* repeat for timing accuracy */

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

    int other = 1 - rank;

    /* Setup arrays */
    int send_buf[ARRAY_SIZE];
    int recv_buf_blocking[ARRAY_SIZE];
    int recv_buf_nonblocking[ARRAY_SIZE];

    int p0_data[] = {10, 20, 30, 40, 50, 60, 70, 80};
    int p1_data[] = { 1,  2,  3,  4,  5,  6,  7,  8};

    if (rank == 0) for (int i = 0; i < ARRAY_SIZE; i++) send_buf[i] = p0_data[i];
    else           for (int i = 0; i < ARRAY_SIZE; i++) send_buf[i] = p1_data[i];

    MPI_Status  status;
    MPI_Request req[2];

    /* ======== Blocking communication ======== */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_block_start = MPI_Wtime();

    for (int r = 0; r < REPEAT; r++) {
        if (rank == 0) {
            MPI_Send(send_buf, ARRAY_SIZE, MPI_INT, other, 0, MPI_COMM_WORLD);
            MPI_Recv(recv_buf_blocking, ARRAY_SIZE, MPI_INT, other, 0, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(recv_buf_blocking, ARRAY_SIZE, MPI_INT, other, 0, MPI_COMM_WORLD, &status);
            MPI_Send(send_buf, ARRAY_SIZE, MPI_INT, other, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_block_end = MPI_Wtime();

    /* ======== Nonblocking communication ======== */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_nonblock_start = MPI_Wtime();

    for (int r = 0; r < REPEAT; r++) {
        MPI_Irecv(recv_buf_nonblocking, ARRAY_SIZE, MPI_INT, other, 1, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(send_buf,             ARRAY_SIZE, MPI_INT, other, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_nonblock_end = MPI_Wtime();

    /* ======== Output ======== */
    if (rank == 0) {
        printf("P0 received (blocking): ");
        for (int i = 0; i < ARRAY_SIZE; i++) printf("%d ", recv_buf_blocking[i]);
        printf("\n");

        printf("P0 received (nonblocking): ");
        for (int i = 0; i < ARRAY_SIZE; i++) printf("%d ", recv_buf_nonblocking[i]);
        printf("\n\nNonblocking communication completed.\n\n");

        printf("Sample performance (averaged over %d iterations):\n", REPEAT);
        printf("Blocking communication time    = %.6f s\n",
               (t_block_end    - t_block_start)    / REPEAT);
        printf("Nonblocking communication time = %.6f s\n",
               (t_nonblock_end - t_nonblock_start) / REPEAT);
    } else {
        printf("P1 received (blocking): ");
        for (int i = 0; i < ARRAY_SIZE; i++) printf("%d ", recv_buf_blocking[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
