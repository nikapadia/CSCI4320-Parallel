#include <mpi.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>
#include <iostream>

#define BASE_SAMPLES (1LL << 33)  // 8 billion samples per rank for weak scaling
#define TOTAL_STRONG_SAMPLES (96LL * (1LL << 30))  // 96 billion samples for strong scaling
#define THREADS_PER_BLOCK 256

// Function to get time in microseconds
unsigned long long getticks(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((unsigned long long)tv.tv_sec * 1000000ULL) + tv.tv_usec;
}

// CUDA Kernel for Monte Carlo π estimation
__global__ void monteCarloPiKernel(curandState *states, int *d_counts, long long samples_per_thread) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234 + idx, 0, 0, &states[idx]);

    long long local_count = 0;
    for (long long i = 0; i < samples_per_thread; i++) {
        float x = curand_uniform(&states[idx]) * 2.0f - 1.0f;
        float y = curand_uniform(&states[idx]) * 2.0f - 1.0f;
        if (x * x + y * y <= 1.0f) {
            local_count++;
        }
    }
    atomicAdd(d_counts, local_count);
}

extern "C" void runCudaLand(int rank) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cudaSetDevice(rank); // Assign each MPI rank to a GPU

    int *d_counts;
    curandState *d_states;
    cudaMallocManaged(&d_counts, sizeof(int));
    cudaMalloc(&d_states, THREADS_PER_BLOCK * sizeof(curandState));

    *d_counts = 0;

    long long samples_per_rank;
    if (rank % 2 == 0) { // Simulate "weak" or "strong" mode based on rank parity
        samples_per_rank = BASE_SAMPLES;  // Weak scaling
    } else {
        samples_per_rank = TOTAL_STRONG_SAMPLES / size;  // Strong scaling
    }

    long long samples_per_thread = samples_per_rank / (THREADS_PER_BLOCK * 256);
    int blocks = 256;
    int threads = THREADS_PER_BLOCK;

    // Create a cuRAND generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL + rank);

    MPI_Barrier(MPI_COMM_WORLD); // Sync before timing
    unsigned long long start_ticks = getticks();

    monteCarloPiKernel<<<blocks, threads>>>(d_states, d_counts, samples_per_thread);
    cudaDeviceSynchronize();

    unsigned long long elapsed_ticks = getticks() - start_ticks;

    int local_hits = *d_counts;
    int global_hits = 0;
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double pi_estimate = 4.0 * global_hits / (samples_per_rank * size);
        std::cout << "MPI Ranks: " << size << "\n";
        std::cout << "Estimated Pi: " << pi_estimate << "\n";
        std::cout << "Execution Time: " << elapsed_ticks / 1e6 << " seconds\n";
    }

    curandDestroyGenerator(gen);
    cudaFree(d_counts);
    cudaFree(d_states);
}
