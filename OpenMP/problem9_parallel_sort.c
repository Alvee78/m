/*
 * Problem 9 (OpenMP): Parallel Sorting
 * Each thread sorts its local portion of the array. After all threads finish,
 * a parallel merge step produces a globally sorted array.
 *
 * Compile: gcc -fopenmp -o problem9 problem9_parallel_sort.c
 * Run:     ./problem9
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_THREADS 4

int compare_int(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

/* Merge two sorted subarrays arr[l..m] and arr[m+1..r] */
void merge(int *arr, int *tmp, int l, int m, int r) {
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) tmp[k++] = arr[i++];
        else                   tmp[k++] = arr[j++];
    }
    while (i <= m) tmp[k++] = arr[i++];
    while (j <= r) tmp[k++] = arr[j++];
    memcpy(arr + l, tmp + l, (r - l + 1) * sizeof(int));
}

int main(void) {
    int global_array[] = {45, 12, 78, 3, 56, 23, 89, 10};
    int N     = 8;
    int chunk = N / NUM_THREADS;
    int tmp[8];

    printf("Original array:");
    for (int i = 0; i < N; i++) printf(" %d", global_array[i]);
    printf("\n\nLocal sorting:\n");

    /* Step 1: Each thread sorts its chunk in parallel */
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid   = omp_get_thread_num();
        int start = tid * chunk;
        qsort(global_array + start, chunk, sizeof(int), compare_int);

        #pragma omp critical
        {
            printf("Thread %d (P%d):", tid, tid);
            for (int i = start; i < start + chunk; i++)
                printf(" %d", global_array[i]);
            printf("\n");
        }
    }

    /* Step 2: Sequential merge passes (merge sorted chunks) */
    for (int width = chunk; width < N; width *= 2) {
        for (int l = 0; l < N; l += 2 * width) {
            int m = l + width - 1;
            int r = (l + 2 * width - 1 < N - 1) ? (l + 2 * width - 1) : (N - 1);
            if (m < r) merge(global_array, tmp, l, m, r);
        }
    }

    printf("\nSorted array:");
    for (int i = 0; i < N; i++) printf(" %d", global_array[i]);
    printf("\n");

    return 0;
}
