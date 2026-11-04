/*
 * Problem 15: Trapezoidal Rule for Numerical Integration
 * Each process computes a portion of the integral. MPI_Reduce sums partial
 * results to get the full integral of f(x) = x^2 from 0 to 1.
 *
 * Compile: mpicc -o problem15 problem15_trapezoidal.c -lm
 * Run:     mpirun -np 4 ./problem15
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Function to integrate */
double f(double x) { return x * x; }

/* Trapezoidal rule for subinterval [a, b] with n intervals */
double trapezoid(double a, double b, int n) {
    double h      = (b - a) / n;
    double sum    = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; i++)
        sum += f(a + i * h);
    return sum * h;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a = 0.0, b = 1.0;
    int    n = 1000;           /* total number of intervals */
    int    local_n = n / size; /* intervals per process     */

    /* Each process computes its sub-interval */
    double h       = (b - a) / n;
    double local_a = a + rank * local_n * h;
    double local_b = local_a + local_n * h;

    double local_integral = trapezoid(local_a, local_b, local_n);

    printf("P%d: integrating [%.4f, %.4f] → partial integral = %.8f\n",
           rank, local_a, local_b, local_integral);

    /* Sum all partial integrals at root */
    double total_integral;
    MPI_Reduce(&local_integral, &total_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double exact = 1.0 / 3.0;
        double error = fabs(total_integral - exact);
        printf("\nNumerical integral ≈ %.6f\n", total_integral);
        printf("Exact integral      = %.6f\n", exact);
        printf("Absolute error      ≈ %.6e\n", error);

        /* Test with different interval counts */
        printf("\nConvergence study:\n");
        printf("%-12s %-18s %-12s\n", "Intervals", "Numerical", "Error");
        int ns[] = {10, 100, 1000, 10000, 100000};
        for (int k = 0; k < 5; k++) {
            double approx = trapezoid(a, b, ns[k]);
            printf("%-12d %-18.10f %-12.2e\n", ns[k], approx, fabs(approx - exact));
        }
    }

    MPI_Finalize();
    return 0;
}
