/**********************************************************************************

 Test program for Lock-free hash table for CUDA; tested for CUDA 4.2 on 32-bit Ubuntu 10.10 and 64-bit Ubuntu 12.04.
 Developed at IIT Kanpur.

 This version changed by ADS to use the httslf hash table on GPU
 for comparison under identical conditions with the IIT Kanpur LockFreeHashTable

 Inputs: Percentage of add and delete operations (e.g., 30 50 for 30% add and 50% delete)
 Output: Prints the total time (in milliseconds) to execute the the sequence of operations

 Compilation flags: -O3 -arch sm_20 -I ~/NVIDIA_GPU_Computing_SDK/C/common/inc/ -DNUM_ITEMS=num_ops -DFACTOR=num_ops_per_thread 

 NUM_ITEMS is the total number of operations (mix of add, delete, search) to execute.

 FACTOR is the number of operations per thread.

 KEYS (cmomnad line parametr now)
 is the number of integer keys assumed in the range [10, 9+KEYS].
 The paper cited below states that the key range is [0, KEYS-1]. However, we have shifted the range by +10 so that
 the head sentinel key (the minimum key) can be chosen as zero. Any positive shift other than +10 would also work.

 The include path ~/NVIDIA_GPU_Computing_SDK/C/common/inc/ is needed for cutil.h.


 Related work:

 Prabhakar Misra and Mainak Chaudhuri. Performance Evaluation of Concurrent Lock-free Data Structures
 on GPUs. In Proceedings of the 18th IEEE International Conference on Parallel and Distributed Systems,
 December 2012.

 Stivala et al 2010 Lock-free parallel dynamic programming
 J Parallel Distrib Comput 70:389-848


 modified by Alex Stivala to include values so have key/value not just key
 $Id: iit_httslf_gpu_test.cu 4501 2013-01-01 09:18:30Z astivala $
***************************************************************************************/


#include <cassert> 
//#include"cutil.h"			// Comment this if cutil.h is not available
#include <cutil_inline.h>
#include"cuda_runtime.h"
#include"stdio.h"
#include <sys/time.h>

// Number of threads per block
#define NUM_THREADS 512

#include "httslf_gpu_kernel.cu"

#if __WORDSIZE == 64
typedef unsigned long long LL;
#else
typedef unsigned int LL;
#endif



// Supported operations
#define ADD (0)
#define DELETE (1)
#define SEARCH (2)



// The main kernel

__global__ void kernel(httslf_entry_t **hashtable, LL* items, LL* op, LL* result)
{
  // The array items holds the sequence of keys
  // The array op holds the sequence of operations
  // The array result, at the end, will hold the outcome of the operations
  // n points to an array of pre-allocated free linked list nodes

#ifdef USE_INSTRUMENT
  reset_counters();
#endif
  int tid,i;
  for(i=0;i<FACTOR;i++){		// FACTOR is the number of operations per thread
    tid=i*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=NUM_ITEMS) return;

    // Grab the operation and the associated key and execute
    LL itm=items[tid];
    if(op[tid]==ADD){
      httslf_insert(hashtable, itm, itm+1);
      result[tid] = itm;
    }
    if(op[tid]==DELETE){
#ifdef ALLOW_DELETE
      httslf_delete(hashtable, itm);
#endif
      result[tid] = itm;
    }
    if(op[tid]==SEARCH){
      LL val;
      bool found=httslf_lookup(hashtable, itm, &val);
      if (found) {
        assert(val == itm+1); // compute capability 2.x has assert in device
      }
    }
  }
#ifdef USE_INSTRUMENT
  httslf_sumcounters();
#endif
}

/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  
(from GNU libc manual) */
     
static int
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

int main(int argc, char** argv)
{
  unsigned int hTimer;
  double runtime;
  cudaError_t rc;
  httslf_entry_t **devHashtable;
  httslf_entry_t **devCellpool;
  int i;
  struct timeval start_timeval,end_timeval,elapsed_timeval;  
  int etime;

  if (argc != 4) {
     printf("Need three arguments: keys, percent add ops and percent delete ops (e.g., 100000 30 50 for 100000 keys and 30%% add and 50%% delete).\nAborting...\n");
     exit(1);
  }

  int KEYS = atoi(argv[1]);
  int adds=atoi(argv[2]);
  int deletes=atoi(argv[3]);

  if (adds+deletes > 100) {
     printf("Sum of add and delete precentages exceeds 100.\nAborting...\n");
     exit(1);
  }

#ifndef ALLOW_DELETE
  if (deletes > 0) {
    fprintf(stderr, "compiled without ALLOW_DELETE: delete not supported\n");
    exit(1);
  }
#endif

  fprintf(stderr, "NUM_ITEMS = %d, KEYS = %d\n", NUM_ITEMS, KEYS);
  fprintf(stderr, "adds = %d, deletes = %d\n", adds, deletes);
  // Calculate the number of thread blocks
  // NUM_ITEMS = total number of operations to execute
  // NUM_THREADS = number of threads per block
  // FACTOR = number of operations per thread

  int blocks=(NUM_ITEMS%(NUM_THREADS*FACTOR)==0)?NUM_ITEMS/(NUM_THREADS*FACTOR):(NUM_ITEMS/(NUM_THREADS*FACTOR))+1;


  fprintf(stderr, "blocks = %d NUM_THREADS = %d\n", blocks, NUM_THREADS);
  fprintf(stderr, "NUM_ITEMS = %d FACTOR = %d\n", NUM_ITEMS, FACTOR);

   cutCreateTimer(&hTimer) ;
   cutResetTimer(hTimer) ;
   cutStartTimer(hTimer) ;


  gettimeofday(&start_timeval, NULL);
  /* allocate cell pool on device */
  size_t cell_size = sizeof(httslf_entry_t);
  int num_adds=(NUM_ITEMS*adds)/100;
  unsigned int devCellpool_num_items = num_adds+1;
  fprintf(stderr, "devCellpool_num_items = %u\n", devCellpool_num_items);
  size_t devCellpool_size = cell_size * devCellpool_num_items;


  // instead of cell pool, set heap size large enough to use device malloc
  // on compute cabability 2.x and higher
  // doulbe the size of the heap to allow plenty of space for malloc overhead
//  if ((rc = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2*devCellpool_size)) != cudaSuccess)
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
  if ((rc = cudaMemcpyToSymbol("total_num_cells", &devCellpool_num_items, sizeof(devCellpool_num_items))) != cudaSuccess) 
  {
    fprintf(stderr, "cudaMemcpyToSymbol poolsize failed%d\n",rc);
    exit(1);
  }
  

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
  httslf_reset<<<blocks, NUM_THREADS>>>(devHashtable);
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "httslf_reset kernel error %d\n", rc);
  }
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "httslf_reset sync error %d\n", rc);
  }
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  fprintf(stderr, "httslf_reset elapsed time %d ms\n", etime);




  LL* op=(LL*)malloc(sizeof(LL)*NUM_ITEMS);		// Array of operations
  LL* items=(LL*)malloc(sizeof(LL)*NUM_ITEMS);		// Array of keys
  LL* result=(LL*)malloc(sizeof(LL)*NUM_ITEMS);		// Arrays of outcome
  srand(0);

  // NUM_ITEMS is the total number of operations to execute
  for(int i=0;i<NUM_ITEMS;i++){
    items[i]=10+rand()%KEYS;	// Keys
  }

  // Populate the op sequence
  for(i=0;i<(NUM_ITEMS*adds)/100;i++){
    op[i]=ADD;
  }
  for(;i<(NUM_ITEMS*(adds+deletes))/100;i++){
    op[i]=DELETE;
  }
  for(;i<NUM_ITEMS;i++){
    op[i]=SEARCH;
  }


   cutStopTimer(hTimer) ;
  fprintf(stderr, "host data generation time: %f ms\n", cutGetTimerValue(hTimer));

   cutResetTimer(hTimer) ;
   cutStartTimer(hTimer) ;

  // Allocate device memory

  LL* Citems;
  LL* Cop;
  LL* Cresult;
  
#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Cresult, sizeof(LL)*NUM_ITEMS));
  CUDA_SAFE_CALL(cudaMalloc((void**)&Citems, sizeof(LL)*NUM_ITEMS));
  CUDA_SAFE_CALL(cudaMalloc((void**)&Cop, sizeof(LL)*NUM_ITEMS));
  CUDA_SAFE_CALL(cudaMemcpy(Citems, items, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(Cop, op, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
#else
  cudaMalloc((void**)&Cresult, sizeof(LL)*NUM_ITEMS);
  cudaMalloc((void**)&Citems, sizeof(LL)*NUM_ITEMS);
  cudaMalloc((void**)&Cop, sizeof(LL)*NUM_ITEMS);
  cudaMemcpy(Citems, items, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
  cudaMemcpy(Cop, op, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
#endif

   cutStopTimer(hTimer) ;
  fprintf(stderr, "alloc and copy data (%f MB) to device time: %f ms\n", (double)sizeof(LL)*NUM_ITEMS*2/(1024*1024), cutGetTimerValue(hTimer));

   cutResetTimer(hTimer) ;
   cutStartTimer(hTimer) ;

  // Launch main kernel

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  kernel<<<blocks, NUM_THREADS>>>(devHashtable, Citems, Cop, Cresult);
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Print kernel execution time in milliseconds

  printf("elapsed time %lf\n",time);

  // Check for errors

  cudaError_t error = cudaGetLastError();
  if(cudaSuccess!=error){
    printf("error:CUDA ERROR (%d) {%s}\n",error, cudaGetErrorString(error));
    exit(-1);
  }

   cutStopTimer(hTimer) ;
  runtime =  cutGetTimerValue(hTimer);
  fprintf(stderr, "kernel exeuction time: %f ms\n", runtime);

   cutResetTimer(hTimer) ;
   cutStartTimer(hTimer) ;

  // Move results back to host memory

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMemcpy(result, Cresult, sizeof(LL)*NUM_ITEMS, cudaMemcpyDeviceToHost));
#else
  cudaMemcpy(result, Cresult, sizeof(LL)*NUM_ITEMS, cudaMemcpyDeviceToHost);
#endif

  cutStopTimer(hTimer) ;
  fprintf(stderr, "time to copy results back to host: %f ms\n", cutGetTimerValue(hTimer));


#ifdef USE_INSTRUMENT
  httslf_computestats<<<blocks, NUM_THREADS>>>(devHashtable);
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "httslf_computestats sync error %d (%s)\n", rc,cudaGetErrorString(rc));
  }
  httslf_printstats<<<blocks, NUM_THREADS>>>();
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "httslf_printstats sync error %d (%s)\n", rc,cudaGetErrorString(rc));
  }
#endif

  cudaFree(devHashtable);
  cutilDeviceReset();

  return 0;
}
