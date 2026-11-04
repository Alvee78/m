/*
 * Problem 3: Ring Topology Communication
 * Every MPI process sends a message to its next process and receives from
 * its previous process (ring: P0→P1→P2→P3→P0).
 *
 * Compile: mpicc -o problem3 problem3_ring_topology.c
 * Run:     mpirun -np 4 ./problem3
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send_val, recv_val;
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    MPI_Status status;
    double t_start, t_end;

    /* P0 initiates with value 100 */
    if (rank == 0) {
        send_val = 100;
        t_start = MPI_Wtime();
        MPI_Send(&send_val, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        printf("P%d sends %d to P%d\n", rank, send_val, next);

        MPI_Recv(&recv_val, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
        t_end = MPI_Wtime();
        printf("P%d receives %d from P%d\n", rank, recv_val, prev);
        printf("\nRing communication completed.\n");
        printf("Total ring time: %.6f seconds\n", t_end - t_start);
    } else {
        MPI_Recv(&recv_val, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
        printf("P%d receives %d from P%d\n", rank, recv_val, prev);

        send_val = recv_val; /* pass it along unchanged */
        MPI_Send(&send_val, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        printf("P%d sends %d to P%d\n", rank, send_val, next);
    }

    MPI_Finalize();
    return 0;
}
