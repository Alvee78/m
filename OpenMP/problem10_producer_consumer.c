/*
 * Problem 10 (OpenMP): Producer-Consumer System
 * One producer thread generates data and puts it in a shared queue.
 * Multiple consumer threads dequeue and process items.
 *
 * Compile: gcc -fopenmp -o problem10 problem10_producer_consumer.c
 * Run:     ./problem10
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_ITEMS     8
#define QUEUE_CAPACITY 16

/* Simple thread-safe circular queue using a mutex */
typedef struct {
    int  data[QUEUE_CAPACITY];
    int  head, tail, count;
    int  done;
    omp_lock_t lock;
} Queue;

void queue_init(Queue *q) {
    q->head = q->tail = q->count = q->done = 0;
    omp_init_lock(&q->lock);
}

void queue_push(Queue *q, int val) {
    while (1) {
        omp_set_lock(&q->lock);
        if (q->count < QUEUE_CAPACITY) {
            q->data[q->tail] = val;
            q->tail = (q->tail + 1) % QUEUE_CAPACITY;
            q->count++;
            omp_unset_lock(&q->lock);
            return;
        }
        omp_unset_lock(&q->lock);
    }
}

int queue_pop(Queue *q, int *val) {
    omp_set_lock(&q->lock);
    if (q->count > 0) {
        *val   = q->data[q->head];
        q->head = (q->head + 1) % QUEUE_CAPACITY;
        q->count--;
        omp_unset_lock(&q->lock);
        return 1;
    }
    omp_unset_lock(&q->lock);
    return 0;
}

int main(void) {
    int items[]       = {10, 20, 30, 40, 50, 60, 70, 80};
    int num_consumers = 2;
    int total_threads = 1 + num_consumers;

    Queue q;
    queue_init(&q);

    int total_consumed = 0;

    #pragma omp parallel num_threads(total_threads)
    {
        int tid = omp_get_thread_num();

        if (tid == 0) {
            /* PRODUCER */
            for (int i = 0; i < NUM_ITEMS; i++) {
                queue_push(&q, items[i]);
                printf("P0 produced: %d\n", items[i]);
            }
            #pragma omp atomic write
            q.done = 1;
        } else {
            /* CONSUMER */
            int val;
            while (1) {
                int done_flag;
                #pragma omp atomic read
                done_flag = q.done;

                if (queue_pop(&q, &val)) {
                    #pragma omp critical
                    {
                        printf("P%d consumed: %d\n", tid, val);
                        total_consumed++;
                    }
                } else if (done_flag) {
                    break;
                }
            }
        }
    }

    printf("\nAll %d items processed successfully.\n", total_consumed);

    omp_destroy_lock(&q.lock);
    return 0;
}
