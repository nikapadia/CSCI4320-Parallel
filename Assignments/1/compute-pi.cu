#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned long long ticks;

unsigned int g_iters = 1 << 30; // 1 billion samples

double *g_x = NULL; // modify to use cudaManageMalloc
double *g_y = NULL; // modify to use cudaManageMalloc

/*********************************************************************/
// POWER9 cycle counter - clock rate is 512,000,000 cycles per second.
/*********************************************************************/

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

void generate_random_vectors()
{
    for (unsigned int i = 0; i < g_iters; i++)
    {
        g_x[i] = (double)rand() / RAND_MAX;
        g_y[i] = (double)rand() / RAND_MAX;
    }
}

double pi_serial(unsigned int iters)
{
    unsigned int count = 0;
    ;
    for (unsigned int i = 0; i < iters; i++)
    {
        if (((g_x[i] - 0.5) * (g_x[i] - 0.5)) +
                ((g_y[i] - 0.5) * (g_y[i] - 0.5)) <=
            0.25)
            count++;
    }
    return (double)(((double)count * 4.0) / (double)iters);
}

// CUDA kernel
__global__ void pi_cuda(unsigned int iters, double *x, double *y, unsigned int *count)
{
    extern __shared__ unsigned int shared_hits[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    unsigned int hits = 0;

    if (idx < iters)
    {
        double dx = x[idx] - 0.5;
        double dy = y[idx] - 0.5;

        // quarter circle check
        if (dx * dx + dy * dy <= 0.25)
        {
            hits = 1;
        }
    }

    shared_hits[tid] = hits;
    __syncthreads();

    // reduce shared_hits to count[blockIdx.x]
    unsigned int s = blockDim.x / 2;
    for (; s > 0; s >>= 1) // >>= is bitwise right shift assignment
    {
        if (tid < s)
        {
            shared_hits[tid] += shared_hits[tid + s];
        }
        __syncthreads();
    }

    // count[blockIdx.x] is the sum of all shared_hits
    if (tid == 0)
    {
        count[blockIdx.x] = shared_hits[0];
    }
}

int main()
{
    unsigned int iters = g_iters; // 1 billion iterations
    double pi = 0.0;
    ticks start = 0;
    ticks finish = 0;

    srand(2147483647); // init with 8th Mersenne prime - 2^31-1

    cudaMallocManaged(&g_x, iters * sizeof(double));    
    cudaMallocManaged(&g_y, iters * sizeof(double));

    generate_random_vectors();

    printf("Testing serial version\n");

    start = getticks();
    pi = pi_serial(iters);
    finish = getticks();

    printf("PI is %18.16lf and computed (not including RNGs) in %llu ticks\n", pi, (finish - start));


    // test with 64, 128, 256, 512, 1024
    int blockSizes[] = {64, 128, 256, 512, 1024};
    for (int i = 0; i < 5; i++)
    {
        unsigned int blockSize = blockSizes[i];
        unsigned int numBlocks = (iters + blockSize - 1) / blockSize;
        unsigned int *counts;
        cudaMallocManaged(&counts, numBlocks * sizeof(unsigned int));
        
        printf("Testing with block size %d\n", blockSize);

        start = getticks();
        pi_cuda<<<numBlocks, blockSize, blockSize * sizeof(unsigned int)>>>(iters, g_x, g_y, counts);
        cudaDeviceSynchronize();
        finish = getticks();
        
        unsigned long long count = 0;
        for (int i = 0; i < numBlocks; i++)
        {
            count += counts[i];
        }
        
        pi = 4.0 * ((double)count / (double)iters);
        printf("PI with CUDA with %d block is %18.16lf in %llu ticks\n\n", blockSize, pi, (finish - start));

        cudaFree(counts);
    }

    return 0;
}
