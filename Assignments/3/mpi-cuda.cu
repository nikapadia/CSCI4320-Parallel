#include <mpi.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>
#include <iostream>
#include <cstdlib>
#include <iomanip>

#define THREADS_PER_BLOCK 256

typdef unsigned long long ticks;

// Tick counter from assignment 1
static __inline__ ticks getticks(void)
{
    unsigned int tbl, tbu0, tbu1;

    do
    {
        __asm__ __volatile__("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__("mftb %0" : "=r"(tbl));
        __asm__ __volatile__("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);

    return (((unsigned long long)tbu0) << 32) | tbl;
}

// Custom atomic function for long long since atomicAdd doesn't work apparently, even though it should. I'm not entirely sure why.
__device__ void atomicAddLongLong(long long *address, long long value) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed + value);
    } while (assumed != old);
}

// CUDA Kernel
__global__ void computePi(curandState *states, long long *d_counts, long long samples_per_thread) {
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
    atomicAddLongLong(d_counts, local_count); // Use custom atomic function for long long
    __syncthreads(); // Ensure all threads have completed before exiting
}

extern "C" void runCudaLand(int rank, long long samples_per_rank) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cudaSetDevice(rank); // Assign each MPI rank to a GPU

    long long *d_counts;
    curandState *d_states;
    cudaMallocManaged(&d_counts, sizeof(long long));
    cudaMalloc(&d_states, THREADS_PER_BLOCK * sizeof(curandState));

    cudaMemset(d_counts, 0, sizeof(long long)); // Properly initialize d_counts to 0

    long long samples_per_thread = samples_per_rank / (THREADS_PER_BLOCK * 256);
    int blocks = 256;
    int threads = THREADS_PER_BLOCK;

    // Create a cuRAND generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL + rank);

    MPI_Barrier(MPI_COMM_WORLD); // Sync before timing
    unsigned long long start_ticks = getticks();

    computePi<<<blocks, threads>>>(d_states, d_counts, samples_per_thread);
    cudaDeviceSynchronize();

    unsigned long long end_ticks = getticks();

    long long local_hits = *d_counts;
    long long global_hits = 0;
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    long long total_samples = 0;
    MPI_Reduce(&samples_per_rank, &total_samples, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double pi_estimate = 4.0 * global_hits / total_samples;
        std::cout << "Total Samples: " << total_samples << "\n";
        std::cout << "Total Hits: " << global_hits << "\n";
        std::cout << "MPI Ranks: " << size << "\n";
        std::cout << "Estimated Pi: " << std::setprecision(20) << pi_estimate << "\n";
        std::cout << "Ticks: " << (end_ticks - start_ticks) << "\n";
        std::cout << "Time (ms): " << (end_ticks - start_ticks) / 1000.0 << "\n";
        std::cout << "Time (s): " << (end_ticks - start_ticks) / 1000000.0 << "\n";
    }

    curandDestroyGenerator(gen);
    cudaFree(d_counts);
    cudaFree(d_states);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <samples_per_rank>\n";
        }
        MPI_Finalize();
        return 1;
    }

    long long samples_per_rank = std::atoll(argv[1]);
    runCudaLand(rank, samples_per_rank);

    MPI_Finalize();
    return 0;
}