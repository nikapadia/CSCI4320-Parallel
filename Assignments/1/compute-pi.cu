#include <stdlib.h>
#include <stdio.h>
#include<unistd.h>
#include<stdbool.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>

typedef unsigned long long ticks;

unsigned int g_iters=1<<30; // 1 billion samples

double *g_x=NULL; // modify to use cudaManageMalloc
double *g_y=NULL; // modify to use cudaManageMalloc

/*********************************************************************/
// POWER9 cycle counter - clock rate is 512,000,000 cycles per second.
/*********************************************************************/

static __inline__ ticks getticks(void)
{
  unsigned int tbl, tbu0, tbu1;

  do {
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);

  return (((unsigned long long)tbu0) << 32) | tbl;
}

void generate_random_vectors()
{
  for(unsigned int i=0; i<g_iters; i++)
    {
      g_x[i]=(double)rand()/RAND_MAX;
      g_y[i]=(double)rand()/RAND_MAX;
    }
}

double pi_serial(unsigned int iters)
{
  unsigned int count=0;;
  for(unsigned int i=0; i<iters; i++)
    {
      if (((g_x[i] - 0.5) * (g_x[i] - 0.5)) +
      	 ((g_y[i] - 0.5) * (g_y[i] - 0.5)) <= 0.25)	    
	count++;
    }
  return (double)(((double)count*4.0)/ (double)iters);
}

/* __global__ */
/* void pi_cuda(unsigned int iters, double *x, double *y, unsigned int *count) */
/* { */
/* 	printf("Your CUDA Code goes here \n"); */
//      allocate the right amount of shared memory for the blockSize - threads in block.
//      for each hit in the circle, set thread id shared memory index to 1
//      sync threads
//      block thread id 0, does the sum across all threads in block
//      writes sum to count[blockIdx.x]
//      CPU side sums up all counts.
/* } */

int main()
{
  unsigned int iters=g_iters; // 1 billion iterations
  double pi=0.0;
  ticks start=0;
  ticks finish=0;

  srand(2147483647); // init with 8th Mersenne prime - 2^31-1

  g_x = (double *)calloc( (size_t)iters, sizeof(double)); // rework using cudaMallocManaged
  g_y = (double *)calloc( (size_t)iters, sizeof(double)); // rework using cudaMallocManaged

  generate_random_vectors();

  start = getticks();
  pi = pi_serial(iters);
  finish = getticks();
  
  printf("PI is %18.16lf and computed (not including RNGs) in %llu ticks\n", pi, (finish - start));

  start = getticks();
  /**************************************************************************/
  /* You implement this *****************************************************/
  // pi = pi_cuda<<<numBlocks, blockSize>>>(iters, g_x, g_y, count_array );
  // don't forget to allocate count_array
  // need to sum count_array which is equal to the number of 
  /* You implement this *****************************************************/
  finish = getticks();
  
  printf("PI with CUDA is %18.16lf and computed (not including RNGs) in %llu ticks\n", pi, (finish - start));
  
  return 0;
}
