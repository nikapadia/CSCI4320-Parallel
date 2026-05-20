#include <stdio.h>
#include <stdlib.h> 
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

#define MAX_THREADS 128  

void *compute_pi (void *); 

unsigned long long hits[MAX_THREADS];
unsigned long long total_hits=0;
unsigned long long sample_points_per_thread=1024*1024*1024;
double pi_estimate = 0.0;

typedef struct {
    int thread_id;
    unsigned long long num_samples;
} thread_data_t;

int main(int argc, char *argv[]) 
{ 
    unsigned long long i;
    pthread_t p_threads[MAX_THREADS]; 
    thread_data_t thread_args[MAX_THREADS];
    int num_threads = MAX_THREADS;

    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads <= 0 || num_threads > MAX_THREADS) {
            fprintf(stderr, "Usage: %s [num_threads]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    int scaling_type = 0; // 0: weak, 1: strong
    unsigned long long total_samples = 0;

    if (argc > 2) {
        scaling_type = atoi(argv[2]);
        if (scaling_type < 0 || scaling_type > 1) {
            fprintf(stderr, "Usage: %s [num_threads] [scaling_type]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (scaling_type == 1) {
        total_samples = (32L << 30); // 32 billion
        sample_points_per_thread = total_samples / num_threads;
        // printf("Strong scaling: %lld samples per thread\n", sample_points_per_thread);
    } else {
        total_samples = (8L << 30); // 8 billion
        sample_points_per_thread = total_samples;
        // printf("Weak scaling: %lld samples per thread\n", sample_points_per_thread);
    }



    // Initialize hits array
    for (i = 0; i < num_threads; i++) {
        hits[i] = 0;
    }

    // Create threads
    for (i = 0; i < num_threads; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].num_samples = sample_points_per_thread;

        pthread_create(&p_threads[i], NULL, compute_pi, (void *)&thread_args[i]);
        // printf("Create thread %lx \n", p_threads[i]);
    }

    // Wait for threads to complete and accumulate results
    total_hits = 0;
    for (i = 0; i < num_threads; i++) {
        pthread_join(p_threads[i], NULL);
        total_hits += hits[i];
    }

    pi_estimate = (double)(((double)total_hits * 4.0) / ((double)(sample_points_per_thread * num_threads)));

    printf("Number of Threads = %d\n", num_threads);
    printf("Sample Points per Thread = %lld \n", sample_points_per_thread);
    printf("Total Hits = %lld \n", total_hits);
    printf("Pi estimate = %.20lf \n", pi_estimate);

    return (0);
}

void *compute_pi(void *arg) {
    unsigned long long i = 0;
    unsigned long long seed;
    thread_data_t *thread_args = (thread_data_t *)arg;
    int thread_id = thread_args->thread_id;
    unsigned long long num_samples = thread_args->num_samples;
    struct drand48_data seed_data;
    double rand_no_x = 0.0, rand_no_y = 0.0;
    unsigned long long local_hits = 0;

    printf("Thread %ld: Sample Points per Thread = %lld \n", pthread_self(), num_samples);

    seed = (unsigned long long)pthread_self(); // Use thread ID as seed
    local_hits = 0;

    srand48_r(seed, &seed_data);
    for (i = 0; i < num_samples; i++) {
        drand48_r(&seed_data, &rand_no_x);
        drand48_r(&seed_data, &rand_no_y);
        if (((rand_no_x - 0.5) * (rand_no_x - 0.5) +
             (rand_no_y - 0.5) * (rand_no_y - 0.5)) < 0.25) {
            local_hits++;
        }
    }

    printf("Thread %ld: local hits = %lld \n", pthread_self(), local_hits);

    hits[thread_id] = local_hits;

    pthread_exit(0);
}