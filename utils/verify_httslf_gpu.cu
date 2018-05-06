/*****************************************************************************
 * 
 * File:    verify_httslf_gpu.cu
 * Author:  Alex Stivala
 * Created: December 2012
 *
 * Test harness for separate chaining thread-safe lock-free hash table.
 * 
 * Requires CUDA 4.x and device with compute capability 2.x and higher
 * as it uses atomic CAS on 64 bits and __device__ function recursion.
 * Also printf() from device function for debug.
 *
 * Usage:
 *    verify_httslf_gpu
 *
 * Preprocessor symbols:
 *
 * DEBUG          - include extra assertion checks etc.
 * USE_GOOD_HASH  - use mixing hash function rather than trivial one
 * ALLOW_DELETE  - allow keys to be removed
 *
 * $Id: verify_httslf_gpu.cu 4501 2013-01-01 09:18:30Z astivala $
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <sys/time.h>

#include "cutil_inline.h"

#define NUM_BLOCKS  128
#define NUM_THREADS 512

#include "httslf_gpu_kernel.cu"

#define NUM_INSERTIONS  100000


/*****************************************************************************
 *
 * global functions (callable from host only)
 *
 *****************************************************************************/


__global__ void insert_data(httslf_entry_t **hashtable, uint64_t *items)
{
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  int num_threads = blockDim.x*gridDim.x;
#ifdef USE_INSTRUMENT
  reset_counters();
#endif

  for (int i = tid; i < NUM_INSERTIONS; i+= num_threads)
  {
    httslf_insert(hashtable, items[i], items[i]);
  }
}

__global__ void lookup_data(httslf_entry_t **hashtable, uint64_t *items)
{
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  int num_threads = blockDim.x*gridDim.x;

  for (int i = tid; i < NUM_INSERTIONS; i+= num_threads)
  {
    uint64_t value;
    bool found = httslf_lookup(hashtable, items[i], &value);
    assert(found);
    assert(value == items[i]);
  }
#ifdef USE_INSTRUMENT
  httslf_sumcounters();
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
  cudaError rc;
  struct timeval start_timeval,end_timeval,elapsed_timeval;  
  int etime;
  int blocks = NUM_BLOCKS;
  httslf_entry_t **devHashtable;
  httslf_entry_t **devCellpool;
  uint64_t *items;
  uint64_t *devItems;

  srand(0);

  // generate data to insert
  if (!(items = (uint64_t *)malloc(sizeof(uint64_t) * NUM_INSERTIONS))) {
    fprintf(stderr, "malloc items failed\n");
    exit(1);
  }

  for (int i = 0; i < NUM_INSERTIONS; i++) {
    items[i] = rand();
  }


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
  /* allocate cell pool on device */
  size_t cell_size = sizeof(httslf_entry_t);
  unsigned int devCellpool_num_items = NUM_INSERTIONS + 1;
  fprintf(stderr, "devCellpool_num_items = %u\n", devCellpool_num_items);
  size_t devCellpool_size = cell_size * devCellpool_num_items;

  // instead of cell pool, set heap size large enough to use device malloc
  // on compute cabability 2.x and higher
  // double the size to allow plenty fo space for device malloc overhead
//  if ((rc = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4*devCellpool_size)) != cudaSuccess)
//  {
//    fprintf(stderr, "cudaDeviceSetLimit heap size failed %d (%s)\n",
//            rc, cudaGetErrorString(rc));
//    exit(1);
//  }
//  size_t heapsize;
//  if ((rc = cudaDeviceGetLimit(&heapsize, cudaLimitMallocHeapSize))!=cudaSuccess)
//  {
//    fprintf(stderr, "cudaDeviceGetLimit heap size failed %d (%s)\n",
//            rc, cudaGetErrorString(rc));
//  }
//  fprintf(stderr, "set cuda malloc heap size to %.1f MB\n", 
//          (double)heapsize/(1024*1024) );

  if ((rc = cudaMalloc((void **)&devCellpool, devCellpool_size)) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc devCellpool failed %d\n", rc);
    exit(1);
  }

  gettimeofday(&end_timeval, NULL);
  if ((rc = cudaMalloc((void **)&devCellpool, devCellpool_size)) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc devCellpool failed %d\n", rc);
    exit(1);
  }
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "cudaMalloc %.1f MB cellpool elapsed time %d ms\n", 
          (double)devCellpool_size/(1024*1024), etime);

  /* set globals on device for clel pool alloc */
  if ((rc = cudaMemcpyToSymbol("cellpool", &devCellpool, sizeof(httslf_entry_t *))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpyToSymbol cellpool failed %d\n", rc);
    exit(1);
  }

  /* set constanst on device for cell pool alloc */
  if ((rc = cudaMemcpyToSymbol((const char *)"total_num_cells", &devCellpool_num_items, sizeof(devCellpool_num_items))) != cudaSuccess) 
  {
    fprintf(stderr, "cudaMemcpyToSymbol poolsize failed%d\n",rc);
    exit(1);
  }

  /* allocate itmes on device and copy data */
  if ((rc = cudaMalloc((void **)&devItems, sizeof(uint64_t)*NUM_INSERTIONS)) != cudaSuccess) {
    fprintf(stderr, "cudaMalloc devItems failed %d\n", rc);
    exit(1);
  }
  gettimeofday(&start_timeval, NULL);
  if ((rc = cudaMemcpy(devItems, items, sizeof(uint64_t)*NUM_INSERTIONS, cudaMemcpyHostToDevice)) != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy items to device failed %d\n", rc);
    exit(1);
  }
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "cudaMemcpy %.1f MB items elapsed time %d ms\n",
          (double)(NUM_INSERTIONS*sizeof(uint64_t)/(1024*1024)),
          etime);



  gettimeofday(&start_timeval, NULL);
  /* allocate hashtable on device */
  if ((rc = cudaMalloc((void **)&devHashtable, 
                       HTTSLF_SIZE*sizeof(httslf_entry_t *))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc devHashtable failed %d\n", rc);
    exit(1);
  }
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "cudaMalloc %.1f MB hashtable elapsed time %d ms\n", 
          (double)HTTSLF_SIZE*sizeof(httslf_entry_t *)/(1024*1024), etime);

  gettimeofday(&start_timeval, NULL);
  /* set hashtable to all empty keys/values */
  httslf_reset<<<dimGrid, dimBlock>>>(devHashtable);
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "httslf_reset kernel error %d\n", rc);
  }
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "httslf_reset sync error %d (%s)\n", rc,cudaGetErrorString(rc));
  }
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "httslf_reset elapsed time %d ms\n", etime);

  gettimeofday(&start_timeval, NULL);

  /* Run the insert kernel */
  insert_data<<<dimGrid, dimBlock>>>(devHashtable, devItems);
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "insert_data sync error %d (%s)\n", rc,cudaGetErrorString(rc));
  }

  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  printf("insert elapsed time %d ms\n", etime);

  gettimeofday(&start_timeval, NULL);

  /* Run the lookup kernel */
  lookup_data<<<dimGrid, dimBlock>>>(devHashtable, devItems);
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "lookup_data sync error %d (%s)\n", rc,cudaGetErrorString(rc));
  }

  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  printf("lookup elapsed time %d ms\n", etime);

#ifdef USE_INSTRUMENT
  httslf_computestats<<<dimGrid, dimBlock>>>(devHashtable);
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "httslf_computestats sync error %d (%s)\n", rc,cudaGetErrorString(rc));
  }
  httslf_printstats<<<dimGrid, dimBlock>>>();
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "httslf_printstats sync error %d (%s)\n", rc,cudaGetErrorString(rc));
  }
#endif

  cudaFree(devHashtable);
  cutilDeviceReset();
  exit(EXIT_SUCCESS);
}

