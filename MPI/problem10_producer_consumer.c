/*
 * Problem 10: Producer-Consumer System
 * One or more producer processes generate data; consumer processes receive
 * and process data using point-to-point communication.
 *
 * Compile: mpicc -o problem10 problem10_producer_consumer.c
 * Run:     mpirun -np 3 ./problem10
 *
 * P0 = Producer, P1 and P2 = Consumers
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_ITEMS 8
#define TAG_DATA  10
#define TAG_DONE  99

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 3) {
        if (rank == 0)
            fprintf(stderr, "Requires at least 3 processes.\n");
        MPI_Finalize();
        return 1;
    }

    int items[] = {10, 20, 30, 40, 50, 60, 70, 80};
    int num_consumers = size - 1;  /* ranks 1..size-1 are consumers */
    MPI_Status status;

    if (rank == 0) {
        /* ---- PRODUCER ---- */
        int consumer_idx = 1;  /* round-robin among consumers */
        for (int i = 0; i < NUM_ITEMS; i++) {
            printf("P0 produced: %d\n", items[i]);
            MPI_Send(&items[i], 1, MPI_INT, consumer_idx, TAG_DATA, MPI_COMM_WORLD);
            consumer_idx = (consumer_idx % num_consumers) + 1;
        }
        /* Send termination signal to all consumers */
        int done = -1;
        for (int c = 1; c < size; c++)
            MPI_Send(&done, 1, MPI_INT, c, TAG_DONE, MPI_COMM_WORLD);

        printf("\nAll %d items processed successfully.\n", NUM_ITEMS);

    } else {
        /* ---- CONSUMER ---- */
        int item;
        int consumed = 0;
        while (1) {
            /* Probe for either data or done message */
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_DONE) {
                MPI_Recv(&item, 1, MPI_INT, 0, TAG_DONE, MPI_COMM_WORLD, &status);
                break;
            }

            MPI_Recv(&item, 1, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD, &status);
            printf("P%d consumed: %d\n", rank, item);
            consumed++;
        }
        /* Report back total consumed (optional) */
        MPI_Send(&consumed, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
