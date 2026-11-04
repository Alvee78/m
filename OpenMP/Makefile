# OpenMP Solutions Makefile
# Requires: gcc with OpenMP support
#
# Build all:    make
# Build one:    make problem1
# Run one:      make run1
# Clean:        make clean

CC      = gcc
CFLAGS  = -O2 -Wall -fopenmp
MATH    = -lm

TARGETS = problem1  problem2  problem3  problem4  problem5  problem6  \
          problem7  problem8  problem9  problem10 problem11 problem12 \
          problem13 problem14 problem15 problem16 problem17 problem18

all: $(TARGETS)

problem1:  problem1_boundary_exchange.c
	$(CC) $(CFLAGS) -o $@ $<

problem2:  problem2_matrix_row_sum.c
	$(CC) $(CFLAGS) -o $@ $<

problem3:  problem3_ring_topology.c
	$(CC) $(CFLAGS) -o $@ $<

problem4:  problem4_ping_pong.c
	$(CC) $(CFLAGS) -o $@ $<

problem5:  problem5_matrix_multiply.c
	$(CC) $(CFLAGS) -o $@ $<

problem6:  problem6_halo_stencil.c
	$(CC) $(CFLAGS) -o $@ $<

problem7:  problem7_parallel_vs_sequential.c
	$(CC) $(CFLAGS) -o $@ $<

problem8:  problem8_race_condition.c
	$(CC) $(CFLAGS) -o $@ $<

problem9:  problem9_parallel_sort.c
	$(CC) $(CFLAGS) -o $@ $<

problem10: problem10_producer_consumer.c
	$(CC) $(CFLAGS) -o $@ $<

problem11: problem11_broadcast.c
	$(CC) $(CFLAGS) -o $@ $<

problem12: problem12_scatter_gather.c
	$(CC) $(CFLAGS) -o $@ $<

problem13: problem13_uneven_distribution.c
	$(CC) $(CFLAGS) -o $@ $<

problem14: problem14_reduce.c
	$(CC) $(CFLAGS) -o $@ $<

problem15: problem15_trapezoidal.c
	$(CC) $(CFLAGS) -o $@ $< $(MATH)

problem16: problem16_prefix_scan.c
	$(CC) $(CFLAGS) -o $@ $<

problem17: problem17_alltoall.c
	$(CC) $(CFLAGS) -o $@ $<

problem18: problem18_histogram.c
	$(CC) $(CFLAGS) -o $@ $<

# Run targets
run1:  problem1  ; OMP_NUM_THREADS=4 ./problem1
run2:  problem2  ; OMP_NUM_THREADS=4 ./problem2
run3:  problem3  ; OMP_NUM_THREADS=4 ./problem3
run4:  problem4  ; OMP_NUM_THREADS=2 ./problem4
run5:  problem5  ; OMP_NUM_THREADS=2 ./problem5
run6:  problem6  ; OMP_NUM_THREADS=4 ./problem6
run7:  problem7  ; OMP_NUM_THREADS=2 ./problem7
run8:  problem8  ; OMP_NUM_THREADS=2 ./problem8
run9:  problem9  ; OMP_NUM_THREADS=4 ./problem9
run10: problem10 ; OMP_NUM_THREADS=3 ./problem10
run11: problem11 ; OMP_NUM_THREADS=4 ./problem11
run12: problem12 ; OMP_NUM_THREADS=4 ./problem12
run13: problem13 ; OMP_NUM_THREADS=3 ./problem13
run14: problem14 ; OMP_NUM_THREADS=4 ./problem14
run15: problem15 ; OMP_NUM_THREADS=4 ./problem15
run16: problem16 ; OMP_NUM_THREADS=4 ./problem16
run17: problem17 ; OMP_NUM_THREADS=4 ./problem17
run18: problem18 ; OMP_NUM_THREADS=4 ./problem18

clean:
	rm -f $(TARGETS)
