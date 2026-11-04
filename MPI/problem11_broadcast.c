/*
 * Problem 11: Broadcast Configuration Parameters
 * Root reads configuration parameters and broadcasts them to all processes
 * using MPI_Bcast. Each process then performs a computation.
 *
 * Compile: mpicc -o problem11 problem11_broadcast.c
 * Run:     mpirun -np 4 ./problem11
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int   iterations;
    int   threshold;
    int   block_size;
} Config;

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Create MPI datatype for Config struct */
    MPI_Datatype config_type;
    int          blocklengths[3] = {1, 1, 1};
    MPI_Aint     offsets[3];
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};

    Config dummy;
    MPI_Aint base;
    MPI_Get_address(&dummy,            &base);
    MPI_Get_address(&dummy.iterations, &offsets[0]);
    MPI_Get_address(&dummy.threshold,  &offsets[1]);
    MPI_Get_address(&dummy.block_size, &offsets[2]);
    offsets[0] -= base; offsets[1] -= base; offsets[2] -= base;

    MPI_Type_create_struct(3, blocklengths, offsets, types, &config_type);
    MPI_Type_commit(&config_type);

    Config cfg;

    if (rank == 0) {
        /* Root sets/reads config */
        cfg.iterations = 1000;
        cfg.threshold  = 50;
        cfg.block_size = 256;
        printf("Root broadcasting configuration...\n");
    }

    /* Broadcast config to all processes */
    MPI_Bcast(&cfg, 1, config_type, 0, MPI_COMM_WORLD);

    /* Each process reports received config */
    printf("P%d received: Iterations = %d, Threshold = %d, Block size = %d\n",
           rank, cfg.iterations, cfg.threshold, cfg.block_size);

    /* Each process performs a computation using config */
    long long local_result = 0;
    for (int i = 0; i < cfg.iterations; i++) {
        if ((i % cfg.block_size) < cfg.threshold)
            local_result += i;
    }
    printf("P%d: local computation result = %lld\n", rank, local_result);

    MPI_Type_free(&config_type);
    MPI_Finalize();
    return 0;
}
