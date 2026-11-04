/*
 * Problem 4: Ping-Pong Communication (Latency & Bandwidth)
 * Implement ping-pong between two MPI processes using MPI_Send/MPI_Recv.
 * Test multiple message sizes and analyze latency and bandwidth.
 *
 * Compile: mpicc -o problem4 problem4_ping_pong.c
 * Run:     mpirun -np 2 ./problem4
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITERATIONS 5

/* Message sizes to test (bytes) */
int sizes[] = {64, 256, 1024, 4096, 16384, 65536, 1048576};
int num_sizes = 7;

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

    if (rank == 0) {
        printf("%-15s %-12s %-20s %-20s %-18s\n",
               "Message Size", "Iterations", "Total Round-trip(s)",
               "Avg Round-trip(s)", "Avg One-way(s)");
        printf("-------------------------------------------------------------------------------------\n");
    }

    for (int s = 0; s < num_sizes; s++) {
        int msg_size = sizes[s];
        char *buf = (char *)calloc(msg_size, sizeof(char));
        MPI_Status status;
        double t_start, t_end, elapsed;

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            t_start = MPI_Wtime();
            for (int i = 0; i < ITERATIONS; i++) {
                MPI_Send(buf, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &status);
            }
            t_end = MPI_Wtime();
            elapsed = t_end - t_start;

            double avg_rtt     = elapsed / ITERATIONS;
            double avg_oneway  = avg_rtt / 2.0;
            double bandwidth   = (msg_size * 2.0 * ITERATIONS) / elapsed / (1024.0 * 1024.0);

            printf("\nMessage size  : %d bytes\n", msg_size);
            printf("Iterations    : %d\n", ITERATIONS);
            printf("Total round trips: %d\n", ITERATIONS);
            printf("Average round-trip: %.6f seconds\n", avg_rtt);
            printf("Average one-way  : %.6f seconds\n", avg_oneway);
            printf("Effective bandwidth: %.2f MB/s\n", bandwidth);
            printf("------------------------------------------\n");
        } else {
            for (int i = 0; i < ITERATIONS; i++) {
                MPI_Recv(buf, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Send(buf, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }

        free(buf);
    }

    MPI_Finalize();
    return 0;
}
