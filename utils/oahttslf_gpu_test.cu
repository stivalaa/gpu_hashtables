/*****************************************************************************
 * 
 * File:    oahttslf_gpu_test.cu
 * Author:  Alex Stivala
 * Created: April 2009
 *
 * Test harness for open addressing thread-safe lock-free hash table.
 * 
 * Usage:
 *    oahttslf_gpu_test
 *
 * Preprocessor symbols:
 * DEBUG          - include extra assertion checks etc.
 *
 * $Id: oahttslf_gpu_test.cu 4556 2013-01-13 02:20:29Z astivala $
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <sys/time.h>

#include "cutil_inline.h"
#include "curand_kernel.h"

#define NUM_BLOCKS  128
#define NUM_THREADS 512

#include "oahttslf_gpu_kernel.cu"



#define NUM_INSERTIONS  100000


/*
 *TODO FIXME 
 *  this actually only uses the low 64 bits, doesn't even store high 
 */


typedef struct twothings {
  uint64_t high, low;
} SET;



/*****************************************************************************
 *
 * global functions (callable from host only)
 *
 *****************************************************************************/


/*
 * init_rng()
 *
 * Initialize CURAND pseudrandom number generator
 * See CUDA Toolkit 4.1 CURAND Guide (p.21)
 *
 * Parameters:
 *    state - CURAND state for random number generation
 *
 */
__global__ void init_rng(curandState *state)
{
  int tid=blockIdx.x*blockDim.x+threadIdx.x;

  /* give each therad same seed, different sequence number, no offset */
  curand_init(1234, tid, 0, &state[tid]);
}
 
 
__global__ void insert_random(oahttslf_entry_t *hashtable,
                               curandState *state)
{
  SET s,snew;
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  int num_threads = blockDim.x*gridDim.x;
  int my_num_insertions = NUM_INSERTIONS / num_threads;
  curandState localState = state[tid];/* cache state in fast local memory */
  uint64_t  value;
  uint64_t ivalue;
  int q;
#ifdef DEBUG
  int insertcount = 0, foundcount = 0;
#endif

#ifdef DEBUG
  printf("tid %d doing %d insertions\n", tid, my_num_insertions);
#endif

#ifdef USE_INSTRUMENT
  reset_counters();
#endif

  for (q = 0; q < my_num_insertions; q++)
  {
    s.low = ((uint32_t)curand(&localState) << 31 | (uint32_t)curand(&localState)) + 1;  
    if (s.low == 0)
        s.low = 1;
    s.high = 0;
    if (!oahttslf_lookup(hashtable, s.low, &value))
    {
      snew.low = ((uint32_t)curand(&localState) << 31 | (uint32_t)curand(&localState)) + 1;  
      if (snew.low == 0)
          snew.low = 1;
      snew.high = 0;
      if (oahttslf_lookup(hashtable, snew.low, &ivalue))
      {
#ifdef DEBUG
      foundcount++;
#endif
        if (ivalue != snew.low) {
          printf("ASSERTION FAILURE: thread %d: ivalue=%llX snew.low=%llX\n", tid , ivalue, snew.low);
          return;
        }
      }

      value = (uint64_t)s.low;
      oahttslf_insert(hashtable, s.low, value);
#ifdef DEBUG
      insertcount++;
#endif
    }
    else
    {
#ifdef DEBUG
      foundcount++;
#endif
    /*  assert(value == (uint64_t)s.low); */
      if (value != s.low) {
          printf("ASSERTION FAILURE 2: thread %d: value=%llX s.low=%llX\n",  tid, value, s.low);
          return;
      }
    }
  }
  state[tid] = localState; /* copy back new state from local cache */
#ifdef DEBUG
  printf("tid %d insertcount = %d foundcount = %d\n", tid,insertcount,foundcount);
#endif
#ifdef USE_INSTRUMENT
  oahttslf_sum_stats();
#endif
}




/*****************************************************************************
 *
 * static functions
 *
 *****************************************************************************/


/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  
(from GNU libc manual) */
     
int
timeval_subtract (struct timeval *result, struct timeval *x, 
                  struct timeval *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }
     
  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;
     
  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


/***************************************************************************
 *
 * main
 *
 ***************************************************************************/

int main(int argc, char *argv[])
{
  int rc;
  struct timeval start_timeval,end_timeval,elapsed_timeval;  
  int etime;
  int blocks = NUM_BLOCKS;
  curandState *devStates;
  oahttslf_entry_t *devHashtable;


  gettimeofday(&start_timeval, NULL);
	// Pick the best GPU available, or if the developer selects one at the command line
	int devID = cutilChooseCudaDevice(argc, argv);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devID);
	printf("> GPU Device has Compute Capabilities SM %d.%d\n\n", deviceProp.major, deviceProp.minor);
	int version = (deviceProp.major * 0x10 + deviceProp.minor);
	if (version < 0x20) {
    fprintf(stderr, "device with compute capability 2.0 or better is required\n");
    exit(1);
  }
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "device init (first CUDA call) elapsed time %d ms\n", etime);

  dim3 dimGrid(blocks)      ; // blocks
  dim3 dimBlock(NUM_THREADS); // threads

  fprintf(stderr, "Execution configuration: Grid = (%d,%d,%d) Block = (%d,%d,%d)\n", dimGrid.x,dimGrid.y,dimGrid.z, dimBlock.x,dimBlock.y,dimBlock.z);

  fprintf(stderr, "Doing %d insertions total with %d threads\n",
          NUM_INSERTIONS, dimGrid.x*dimBlock.x);

  gettimeofday(&start_timeval, NULL);

  /* allocate space on device for random number generator state */
  if ((rc = cudaMalloc((void **)&devStates, 
                       blocks*NUM_THREADS*sizeof(curandState))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc devStates failed %d\n", rc);
    exit(1);
  }
  
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "alloc CURAND states (%.0f KB) elapsed time %d ms\n", 
          (float)blocks*NUM_THREADS*sizeof(curandState)/1024, etime);

  gettimeofday(&start_timeval, NULL);

  /* initialize device random number generator */
  init_rng<<<dimGrid, dimBlock>>>(devStates);
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "init_rng kernel error %d\n", rc);
  }
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "init_rng sync error %d\n", rc);
  }
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "init CURAND kernel elapsed time %d ms\n", etime);

  gettimeofday(&start_timeval, NULL);
  /* allocate hashtable on device */
  if ((rc = cudaMalloc((void **)&devHashtable, 
                       OAHTTSLF_SIZE*sizeof(oahttslf_entry_t))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc devHashtable failed %d\n", rc);
    exit(1);
  }
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "cudaMalloc %.1f MB hashtable elapsed time %d ms\n", 
          (double)OAHTTSLF_SIZE*sizeof(oahttslf_entry_t)/(1024*1024), etime);

  gettimeofday(&start_timeval, NULL);
  /* set hashtable to all empty keys/values */
  oahttslf_reset<<<dimGrid, dimBlock>>>(devHashtable);
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "oahttslf_reset kernel error %d\n", rc);
  }
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "oahttslf_reset sync error %d\n", rc);
  }
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "oahttslf_reset elapsed time %d ms\n", etime);


  gettimeofday(&start_timeval, NULL);

  /* Run the kernel */
  insert_random<<<dimGrid, dimBlock>>>(devHashtable, devStates);
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "insert_random sync error %d\n", rc);
  }

  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  printf("elapsed time %d ms\n", etime);

#ifdef USE_INSTRUMENT
  oahttslf_print_stats<<<dimGrid, dimBlock>>>();
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "oahttslf_print_stats sync error %d\n", rc);
  }
#endif

  cudaFree(devStates);
  cudaFree(devHashtable);
  cutilDeviceReset();
  exit(EXIT_SUCCESS);
}

