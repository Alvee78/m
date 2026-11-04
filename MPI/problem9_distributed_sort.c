/*
 * Problem 9: Distributed Sorting (Odd-Even Transposition Sort)
 * Each process sorts its local portion, then exchanges data with neighbors
 * to produce a globally sorted array.
 *
 * Compile: mpicc -o problem9 problem9_distributed_sort.c
 * Run:     mpirun -np 4 ./problem9
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int compare_int(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

/* Merge two sorted halves; keep lower half if keep_lower, else upper half */
void merge_and_keep(int *local, int *neighbor, int chunk, int keep_lower) {
    int *merged = (int *)malloc(2 * chunk * sizeof(int));
    int i = 0, j = 0, k = 0;
    while (i < chunk && j < chunk) {
        if (local[i] <= neighbor[j]) merged[k++] = local[i++];
        else                          merged[k++] = neighbor[j++];
    }
    while (i < chunk) merged[k++] = local[i++];
    while (j < chunk) merged[k++] = neighbor[j++];

    if (keep_lower)
        memcpy(local, merged, chunk * sizeof(int));
    else
        memcpy(local, merged + chunk, chunk * sizeof(int));

    free(merged);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int global_array[] = {45, 12, 78, 3, 56, 23, 89, 10};
    int N = 8;
    int chunk = N / size;

    int *local = (int *)malloc(chunk * sizeof(int));
    for (int i = 0; i < chunk; i++)
        local[i] = global_array[rank * chunk + i];

    /* Step 1: Local sort */
    qsort(local, chunk, sizeof(int), compare_int);

    printf("P%d (local sorted):", rank);
    for (int i = 0; i < chunk; i++) printf(" %d", local[i]);
    printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);

    /* Step 2: Odd-Even Transposition Sort across processes */
    int *neighbor_buf = (int *)malloc(chunk * sizeof(int));
    MPI_Status status;

    for (int phase = 0; phase < size; phase++) {
        int partner = -1;
        int is_lower = 0;

        if (phase % 2 == 0) {
            /* Even phase: (0,1), (2,3), ... */
            if (rank % 2 == 0 && rank + 1 < size) { partner = rank + 1; is_lower = 1; }
            else if (rank % 2 == 1)                 { partner = rank - 1; is_lower = 0; }
        } else {
            /* Odd phase: (1,2), (3,4), ... */
            if (rank % 2 == 1 && rank + 1 < size)  { partner = rank + 1; is_lower = 1; }
            else if (rank % 2 == 0 && rank > 0)     { partner = rank - 1; is_lower = 0; }
        }

        if (partner != -1) {
            MPI_Sendrecv(local,        chunk, MPI_INT, partner, 0,
                         neighbor_buf, chunk, MPI_INT, partner, 0,
                         MPI_COMM_WORLD, &status);
            merge_and_keep(local, neighbor_buf, chunk, is_lower);
        }
    }

    /* Step 3: Gather sorted array at root */
    int *sorted = NULL;
    if (rank == 0) sorted = (int *)malloc(N * sizeof(int));
    MPI_Gather(local, chunk, MPI_INT, sorted, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nSorted array:");
        for (int i = 0; i < N; i++) printf(" %d", sorted[i]);
        printf("\n");
        free(sorted);
    }

    free(local);
    free(neighbor_buf);
    MPI_Finalize();
    return 0;
}
