/*
 * Problem 15 (OpenMP): Parallel Trapezoidal Integration
 * Each thread computes a portion of the integral of f(x) = x^2.
 * OpenMP reduction combines partial results.
 *
 * Compile: gcc -fopenmp -o problem15 problem15_trapezoidal.c -lm
 * Run:     ./problem15
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double f(double x) { return x * x; }

int main(void) {
    double a = 0.0, b = 1.0;
    int    n = 1000;
    int    num_threads = 4;
    double h = (b - a) / n;
    double total_integral = 0.0;

    #pragma omp parallel num_threads(num_threads) reduction(+:total_integral)
    {
        int tid       = omp_get_thread_num();
        int local_n   = n / num_threads;
        double local_a = a + tid * local_n * h;
        double local_b = local_a + local_n * h;

        /* Trapezoidal rule for this thread's subinterval */
        double partial = 0.5 * (f(local_a) + f(local_b));
        for (int i = 1; i < local_n; i++)
            partial += f(local_a + i * h);
        partial *= h;

        total_integral += partial;

        #pragma omp critical
        printf("Thread %d: integrating [%.4f, %.4f] → partial = %.8f\n",
               tid, local_a, local_b, partial);
    }

    double exact = 1.0 / 3.0;
    printf("\nNumerical integral ≈ %.6f\n", total_integral);
    printf("Exact integral      = %.6f\n", exact);
    printf("Absolute error      ≈ %.6e\n", fabs(total_integral - exact));

    /* Convergence study */
    printf("\nConvergence study (single-threaded reference):\n");
    printf("%-12s %-18s %-12s\n", "Intervals", "Numerical", "Error");
    int ns[] = {10, 100, 1000, 10000, 100000};
    for (int k = 0; k < 5; k++) {
        int m = ns[k];
        double hh = (b - a) / m;
        double approx = 0.5 * (f(a) + f(b));
        for (int i = 1; i < m; i++) approx += f(a + i * hh);
        approx *= hh;
        printf("%-12d %-18.10f %-12.2e\n", m, approx, fabs(approx - exact));
    }

    return 0;
}
