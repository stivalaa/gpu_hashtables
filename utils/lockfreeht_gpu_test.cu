/*****************************************************************************
 * 
 * File:    lockfreeht_gpu_test.cu
 * Author:  Alex Stivala
 * Created: November 2012
 *
 * Test harness for lock-free hash table on GPU. This versino is the
 * original oahttslf test used in my JPDC paper but modified to test
 * the lock free hashtable from:
 *
 *  Prabhakar Misra and Mainak Chaudhuri. Performance Evaluation of
 *  Concurrent Lock-free Data Structures on GPUs. In Proceedings of the
 *  18th IEEE International Conference on Parallel and Distributed
 *  Systems, December 2012.
 *
 * so we can do proper fair comparisons, with this test harness, and
 * also have put oahttslf into their test harness.
 * 
 * Usage:
 *    lockfreeht_gpu_test
 *
 * Preprocessor symbols:
 * DEBUG          - include extra assertion checks etc.
 *
 * $Id: lockfreeht_gpu_test.cu 4556 2013-01-13 02:20:29Z astivala $
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

//NB the standard value I was using of 10 000 000 insertions means 
// setting up the cell free pool takes over an hour (in fact more than 23
// hours - never had it complete as I had a 23 limit on PBS),
// so not practical for repeated tests (or in fact at all),
// need to use only 1 000 000 (at most)
#define NUM_INSERTIONS  100000 
#define NUM_ITEMS NUM_INSERTIONS

#include "LockFreeHashTableValues_kernel.cu"

#define NUM_BLOCKS  128
#define NUM_THREADS 512



typedef unsigned long long uint64_t;
typedef unsigned int       uint32_t;
typedef unsigned short     uint16_t;
typedef unsigned char      uint8_t;


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
 
 
__global__ void insert_random( curandState *state, Node **n)
{
  // n points to an array of pre-allocated free linked list nodes
  nodes = n; // set the device global memory nodes pointer

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

  for (q = 0; q < my_num_insertions; q++)
  {
    s.low = ((uint32_t)curand(&localState) << 31 | (uint32_t)curand(&localState)) + 1;  
    if (s.low == 0)
        s.low = 1;
    s.high = 0;
    LL bkt = Hash(s.low);
    if (!bucketList[bkt]->Search(s.low, value))
    {
      snew.low = ((uint32_t)curand(&localState) << 31 | (uint32_t)curand(&localState)) + 1;  
      if (snew.low == 0)
          snew.low = 1;
      snew.high = 0;
      bkt = Hash(snew.low);
      if (bucketList[bkt]->Search(snew.low, ivalue))
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
      bkt = Hash(s.low);
      bucketList[bkt]->Add(s.low, value);
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

  // Allocate hash table

  LinkedList* buckets[NUM_BUCKETS];
  LinkedList** Cbuckets;
  int i;
   unsigned int hTimer;
   cutCreateTimer(&hTimer);
   cutResetTimer(hTimer) ;
   cutStartTimer(hTimer) ;
  for(i=0;i<NUM_BUCKETS;i++){
#ifdef _CUTIL_H_
    CUDA_SAFE_CALL(cudaMalloc((void**)&(buckets[i]), sizeof(LinkedList)));
#else
    cudaMalloc((void**)&(buckets[i]), sizeof(LinkedList));
#endif
    LinkedList* l=new LinkedList(i);
#ifdef _CUTIL_H_
    CUDA_SAFE_CALL(cudaMemcpy(buckets[i], l, sizeof(LinkedList), cudaMemcpyHostToDevice));
#else
    cudaMemcpy(buckets[i], l, sizeof(LinkedList), cudaMemcpyHostToDevice);
#endif
  }

   cutStopTimer(hTimer) ;
  fprintf(stderr, "alloc and copy buckets in loop (%f MB) time: %f ms\n", ((double)sizeof(LinkedList)*NUM_BUCKETS)/(1024*1024), cutGetTimerValue(hTimer));

   cutResetTimer(hTimer) ;
   cutStartTimer(hTimer) ;

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&(Cbuckets), sizeof(LinkedList*)*NUM_BUCKETS));
#else
  cudaMalloc((void**)&(Cbuckets), sizeof(LinkedList*)*NUM_BUCKETS);
#endif

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMemcpy(Cbuckets, buckets, sizeof(LinkedList*)*NUM_BUCKETS, cudaMemcpyHostToDevice));
#else
  cudaMemcpy(Cbuckets, buckets, sizeof(LinkedList*)*NUM_BUCKETS, cudaMemcpyHostToDevice);
#endif

   cutStopTimer(hTimer) ;
  fprintf(stderr, "alloc and copy bucket pointers (%f MB) time: %f ms\n", ((double)sizeof(LinkedList*)*NUM_BUCKETS)/(1024*1024), cutGetTimerValue(hTimer));

   cutResetTimer(hTimer) ;
   cutStartTimer(hTimer) ;


  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "total hashtable allocation elapsed time %d ms\n", etime);



  gettimeofday(&start_timeval, NULL);

  // Initialize the device memory
  cutResetTimer(hTimer) ;
  cutStartTimer(hTimer) ;
  int b=(NUM_BUCKETS/32)+1;
  init<<<b, 32>>>(Cbuckets);
  cutStopTimer(hTimer) ;
  fprintf(stderr, "init kernel time: %f ms\n", cutGetTimerValue(hTimer));
  
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "total init kernel elapsed time %d ms\n", etime);

  fprintf(stderr, "allocating pool of free nodes...\n");
  gettimeofday(&start_timeval, NULL);
   cutResetTimer(hTimer) ;
   cutStartTimer(hTimer) ;

  // Allocate the pool of free nodes
  int adds = NUM_ITEMS;
  Node** pointers=(Node**)malloc(sizeof(Node*)*adds);
  Node** Cpointers;

  for(i=0;i<adds;i++){
#ifdef _CUTIL_H_
    CUDA_SAFE_CALL(cudaMalloc((void**)&pointers[i], sizeof(Node)));
#else
    cudaMalloc((void**)&pointers[i], sizeof(Node));
#endif
  }
  
   cutStopTimer(hTimer) ;
  fprintf(stderr, "alloc device free nodes in loop (%d calls to cudaMalloc) time: %f ms\n", adds, cutGetTimerValue(hTimer));

   cutResetTimer(hTimer) ;
   cutStartTimer(hTimer) ;

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Cpointers, sizeof(Node*)*adds));
  CUDA_SAFE_CALL(cudaMemcpy(Cpointers, pointers, sizeof(Node*)*adds, cudaMemcpyHostToDevice));
#else
  cudaMalloc((void**)&Cpointers, sizeof(Node*)*adds);
  cudaMemcpy(Cpointers, pointers, sizeof(Node*)*adds, cudaMemcpyHostToDevice);
#endif

   cutStopTimer(hTimer) ;
  fprintf(stderr, "alloc and copy device free node pointres (%f MB) time: %f ms\n", (double)sizeof(Node*)*adds/(1024*1024), cutGetTimerValue(hTimer));


  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  printf("total free node pool elapsed time %d ms\n", etime);

  gettimeofday(&start_timeval, NULL);

  /* Run the kernel */
  insert_random<<<dimGrid, dimBlock>>>(devStates, Cpointers);
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "insert_random sync error %d\n", rc);
  }

  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  printf("elapsed time %d ms\n", etime);

  cudaFree(devStates);
  cutilDeviceReset();
  exit(EXIT_SUCCESS);
}

