/*****************************************************************************
 *
 * File:    knapsack_gpu_oahttslf.cu
 * Author:  Alex Stivala
 * Created: April 2009
 *
 * $Id: knapsack_gpu_oahttslf.cu 4634 2013-01-31 03:17:42Z astivala $
 *
 * This is the GPU implementation using the oahttslf GPU lock free hash table.
 *
 *  Usage: knapsack_gpu_oahttslf [-nvy] [-r threads]  < problemspec
 *          -v: Verbose output 
 *          -n: assume no name in the first line of the file
 *          -y: show instrumentatino summary line (like -t but one line summary)
 *
 * The problemspec is in the format generated by gen2.c from David Pisinger
 * (http://www.diku.dk/hjemmesider/ansatte/pisinger/codes.html):
 *
 * numitems
 *      1 profit_1 weight_1
 *      2 profit_2 weight_2  
 *       ...
 *      numitems profit_numitems weight_numitems
 * capacity 
 *
 * all profits and weights are positive integers.
 *
 * 
 * Requires CUDA 4.x and device with compute capability 2.x and higher
 * as it uses atomic CAS on 64 bits and __device__ function recursion.
 * Also printf() from device function for debug.
 * It uses the CURAND library for pseudrandom number generation.
 *
 * Preprocessor symbols:
 *
 * DEBUG          - compile in lots of debugging code.
 * USE_INSTRUMENT - compile in (per-thread) instrumentation counts.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "cutil_inline.h"
#include "curand_kernel.h"

// Number of threads per block
#define NUM_THREADS 512 // FIXME some sensible value for this? (1024 max?)
// (these are defined prior to inclde of httslf_gpu_oahttslf_kernel.cu etc
// as they are used to define arrays for instrumentation if USE_INSTRUMENT)

#undef ALLOW_UPDATE
#undef ALLOW_DELETE

#include "oahttslf_gpu_kernel.cu"

#ifdef DEBUG
#define DEBUG_PRINT(x)  printf x
#else
#define DEBUG_PRINT(x) /* nothing */
#endif


/*****************************************************************************
 *
 * global constants
 *
 *****************************************************************************/

// size of constant memory on device to store profit,weight for each item
#define MAX_KNAPSACK_ITEMS 512



/* dodgy: 0 is the OAHTTSLF_EMPTY_KEY and OAHTTSLF_EMPTY_VALUE value,
   so if key or value is 0 set it to MAGIC_ZERO instead */
#define MAGIC_ZERO 0xffffffffffffffff

/*****************************************************************************
 *
 * type definitions
 *
 *****************************************************************************/

#ifndef _STDINT_H
typedef unsigned long long  uint64_t;
typedef unsigned int       uint32_t;
typedef unsigned short     uint16_t;
typedef unsigned char      uint8_t;
#endif



/* definition of type for an item */
typedef struct item_s
{
    unsigned int profit;
    unsigned int weight;
} item_t;


typedef unsigned int counter_t;

typedef struct stats_s 
{
    counter_t reuse;  /* number of times value already found in hashtable */
    counter_t hashcount; /* number of times value computed & stored in ht */
} stats_t;



/*****************************************************************************
 *
 * device constant data
 *
 *****************************************************************************/

// array of weight and profit for each item. indexed 1..NUM_KNAPSACK_ITEMS,
// element 0 is unused
__constant__ item_t c_KNAPSACK_ITEMS[MAX_KNAPSACK_ITEMS+1];
const char *c_KNAPSACK_ITEMS_symbol = "c_KNAPSACK_ITEMS";
 

/*****************************************************************************
 *
 * device global data
 *
 *****************************************************************************/

__device__ volatile bool finished = false; // set when a thread has finished the computation


#ifdef USE_INSTRUMENT
/* instrumentatino totals (summed over all threads) */
__device__ counter_t  total_reuse = 0, total_hashcount = 0;
#endif


/*****************************************************************************
 *
 * device shared data
 *
 *****************************************************************************/

#ifdef USE_INSTRUMENT
/* per-thread instrumentation */
/* initialize with knapsack_reset_stats() */
__shared__ stats_t stats[NUM_THREADS];
#endif


/*****************************************************************************
 *
 * host static data
 *
 *****************************************************************************/

static bool printstats; /* whether to print call stats */
static bool verbose;    /* verbose output  */
static bool show_stats_summary = 0; /* -y summary instrumentation stats */


static unsigned int CAPACITY; /* total capacity for the problem */
static unsigned int NUM_KNAPSACK_ITEMS; /* number of items */
static item_t *KNAPSACK_ITEMS;         /* array of item profits and weights (0 unused)*/



/*****************************************************************************
 *
 * host static functions
 *
 *****************************************************************************/

/* 
 * Read the input from stdin in the gen2.c format:
 *
 * numitems
 *      1 profit_1 weight_1
 *      2 profit_2 weight_2  
 *       ...
 *      numitems profit_numitems weight_numitems
 * capacity 
 *
 * all profits and weights are positive integers.
 *
 * Parameters:
 *     None.
 * Return value:
 *     None.
 * Uses global data (write): 
 *      KNAPSACK_ITEMS        - allocates array, sets profit and weight for each item
 *      CAPACITY     - sets capacity for problem
 *      NUM_KNAPSACK_ITEMS   - number of items
 */
static void readdata(void)
{
  unsigned int i,inum;

  if (scanf("%d", &NUM_KNAPSACK_ITEMS) != 1)
  {
    fprintf(stderr, "ERROR reading number of items\n");
    exit(EXIT_FAILURE);
  }
  if (!(KNAPSACK_ITEMS = (item_t *)malloc((NUM_KNAPSACK_ITEMS+1) * sizeof(item_t))))
  {
    fprintf(stderr,"malloc KNAPSACK_ITEMS failed\n");
    exit(EXIT_FAILURE);
  }
  for (i = 1; i <= NUM_KNAPSACK_ITEMS; i++)
  {
    if(scanf("%d %d %d", &inum, &KNAPSACK_ITEMS[i].profit, &KNAPSACK_ITEMS[i].weight) != 3)
    {
      fprintf(stderr, "ERROR reading item %d\n", i);
      exit(EXIT_FAILURE);
    }
    if (inum != i)
    {
      fprintf(stderr, "ERROR expecting item %d got %d\n", i, inum);
      exit(EXIT_FAILURE);
    }
  }  
  if (scanf("%d", &CAPACITY) != 1)
  {
    fprintf(stderr, "ERROR reading capacity\n");
    exit(EXIT_FAILURE);
  }
}



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


/*****************************************************************************
 *
 * device functions
 *
 *****************************************************************************/

#ifdef USE_INSTRUMENT
/*
 * knapsack_reset_stats()
 *
 * Initialize the per-thread intrumentatino counters in shared memory
 *
 * Parameters: None
 * Return value: None
 */
__device__ void knapsack_reset_stats(void)
{
  for (int i = threadIdx.x; i < NUM_THREADS; i += blockDim.x)
  {
    stats[i].reuse = 0;
    stats[i].hashcount = 0;
  }
  __syncthreads();
}


/*
 * knapsack_sum_stats()
 *
 * Sum up the per-thread counters 
 *
 * Also instead of doing a proper efficient reduction just uses atomicAdd
 * (on globals), also reqwuired compute 2.x (only debug/instrumentatinos,
 * no need for efficiency here, just keep it simple).
 *
 * Parameters: None
 * Retrun value: None
 */
__device__ void knapsack_sum_stats(void)
{
  unsigned int block_reuse = 0;
  unsigned int block_hashcount = 0;
  
  __syncthreads();
  
  if (threadIdx.x == 0)
  {
    for (int i = 0; i < NUM_THREADS; i++) 
    {
      block_reuse += stats[i].reuse;
      block_hashcount += stats[i].hashcount;
    }
    atomicAdd(&total_reuse, block_reuse);
    atomicAdd(&total_hashcount, block_hashcount);
  }
}

#endif /* USE_INSTRUMENT */


/*
 * oahttslf_insert_indices()
 *
 * Insert value for (i,j) into the hashtable
 *
 * Parameters:
 *    i,j - indices to build insertion key
 *    value - value to insert for the key
 *
 * Uses global data:
 *    devHashTable
 *
 * Return value:
 *    None.
 */
__device__ void insert_indices(unsigned int i, unsigned int j,
                               unsigned int value, oahttslf_entry_t *devHashtable)
{
  uint64_t key, val64;

  key = (i == 0 && j == 0 ? MAGIC_ZERO : 
         ((uint64_t)i << 32) | (j & 0xffffffff));
  val64 = (value == 0 ? MAGIC_ZERO : (uint64_t)value);
  oahttslf_insert(devHashtable, key, val64);
}



/*
 * lookup_indices()
 *
 * Get the value for (i,j) from the hashtable
 *
 * Parameters:
 *     i,j - indices to build key for lookup
 *     pvalue - (OUTPUT) value for key, only set if TRUE returned
 * 
 * Uses global data:
 *    devHashTable (readonly)
 *
 * Return value:
 *     TRUE if found, FALSE otherwise
 */
__device__ bool lookup_indices(unsigned int i, unsigned int j, 
                               unsigned int *pvalue, oahttslf_entry_t *devHashtable)
{
  uint64_t key,val64;
  bool found;

  key = (i == 0 && j == 0 ? MAGIC_ZERO :
         ((uint64_t)i << 32) | (j & 0xffffffff));
  found =  oahttslf_lookup(devHashtable, key, &val64);
  // avoid branch by setting invalid value if not found
  *pvalue = ((val64 == MAGIC_ZERO) ? 0 : (unsigned int)val64); 
  return found;
}

/*
 * dp_knapsack()
 *
 *      This version is multi-threaded, sharing hashtable used to
 *      store computed values between the threads.
 *      This function is called by dp_knapsack_kernel()
 *      with identical instances running
 *      in several threads. This functino itself is recursive
 *      (never creates threads itself) and diverges as there is 
 *      a random choice as to which of the paths we take first; we use
 *      parallelism to explore the search space concurrently with
 *      diverged paths due to this choice, but still reusing computed
 *      values by the shared lock-free hashtable.
 *
 *
 *      This version uses no bounding.
 *
 *      Parameters:   i - item index
 *                    w - total weight
 *                state - CURAND state for random number generation
 *         devHashtable - the hash table
 *
 *     global constant memory:
 *
 *            c_KNAPSACK_ITEMS - array of profit and weight for each item
 * 
 *     global memory:
 *             finished (readonly) -checked to see if compuattion done
 *
 *      Return value: 
 *                    value of d.p. at (i,w)
 *
 */
__device__ unsigned int dp_knapsack(unsigned int i, unsigned int w, 
                                    curandState *state, oahttslf_entry_t *devHashtable)
{
  unsigned int p,pwithout,pwith;
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  curandState localState = state[tid];

  //DEBUG_PRINT(("(1) blockIdx.x = %d blockDim.x = %d threadIx = %d tid = %d\n",
  //             blockIdx.x, blockDim.x, threadIdx.x, tid));
  //DEBUG_PRINT(("dp_knapsack_kernel tid = %d i = %u w = %u\n", tid,i,w));


//  if (finished)
//  {
//    //DEBUG_PRINT(("dp_knapsack_kernel tid = %d exiting (finished was set)\n",tid));
 //   return 0;//FIXME
//  }

  /* memoization: if value here already computed then do nothing */
  if (lookup_indices(i, w, &p, devHashtable))
  {
    //DEBUG_PRINT(("dp_knapsack_kernel tid = %d found %d,%d p = %d",
    //             tid, i, w, p));
#ifdef USE_INSTRUMENT
    stats[threadIdx.x].reuse++;
#endif
    return p;
  }

  if (i == 0 || w == 0)
  {
    p = 0;
  }
  else if (w < c_KNAPSACK_ITEMS[i].weight)
  {
    //DEBUG_PRINT(("dp_knapsack_kernel LT tid = %d w (%d) < k[%d] (%d)\n",tid,w,i,c_KNAPSACK_ITEMS[i].weight));

    p = dp_knapsack(i - 1, w, state, devHashtable);

    //DEBUG_PRINT(("dp_knapsack_kernel LT tid = %d p = %d\n", tid, p));
  }
  else
  {
    //DEBUG_PRINT(("dp_knapsack_kernel GE_start tid = %d\n", tid));

    if (curand(&localState) & 1)
    {
      pwithout = dp_knapsack(i - 1, w, state, devHashtable);
      pwith = dp_knapsack(i - 1, w - c_KNAPSACK_ITEMS[i].weight, state, devHashtable) + 
        c_KNAPSACK_ITEMS[i].profit;
    }
    else
    {
      pwith = dp_knapsack(i - 1, w - c_KNAPSACK_ITEMS[i].weight, state, devHashtable) +
        c_KNAPSACK_ITEMS[i].profit;
      pwithout = dp_knapsack(i - 1, w, state, devHashtable);
    }
    p = MAX(pwithout, pwith);

    //DEBUG_PRINT(("dp_knapsack_kernel GE tid = %d p = %d\n", tid, p));
  }

  //DEBUG_PRINT(("dp_knapsack_kernel end tid = %d p = %u\n", tid, p));

  state[tid] = localState;
#ifdef USE_INSTRUMENT
  stats[threadIdx.x].hashcount++;
#endif
  insert_indices(i, w, p, devHashtable);
  return p;
}


/*****************************************************************************
 *
 * global functions
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
 *    seed - seed for CURAND init
 *
 */
__global__ void init_rng(curandState *state, unsigned long long seed)
{
  int tid=blockIdx.x*blockDim.x+threadIdx.x;

  /* give each therad same seed, different sequence number, no offset */
  curand_init(seed, tid, 0, &state[tid]);
}
 
 

/*
 * dp_knapsack_kernel()
 *
 *    Caller interafce to the multithreaded version: just calls the actual
 *    device function
 *
 *
 *    Paramters:
 *            i - item index to start at
 *            w - total capacity to start at
 *            p - (output) score for this product set
 *        state - CURAND state for random number generation
 * devHashtable - the hash table
 *
 *     global memory:
 *             finished - set when compuattion done to tell other threads to end
 */
__global__ void dp_knapsack_kernel(unsigned int i, unsigned int w,
                                   unsigned int *p, curandState *state,
                                   oahttslf_entry_t *devHashtable)
{
 int tid=blockIdx.x*blockDim.x+threadIdx.x;
 unsigned int profit = 0;

  //DEBUG_PRINT(("(0) blockIdx.x = %d blockDim.x = %d threadIx = %d tid = %d\n",
  //             blockIdx.x, blockDim.x, threadIdx.x, tid));
  // testing if recursion works at all
//  int f = tid % 10;
//  int factorial = fact(f);
//  DEBUG_PRINT(("tid = %d f = %d fact = %d\n", tid, f, factorial));
//  return;

#ifdef DEBUG
  for (int j = 0;  j < i; j++) 
  {
    printf("item %d: weight = %d profit = %d\n", j, c_KNAPSACK_ITEMS[j].weight, c_KNAPSACK_ITEMS[j].profit);
  }
#endif


#ifdef USE_INSTRUMENT
  knapsack_reset_stats();
  reset_counters();
#endif
  

  profit = dp_knapsack(i, w, state, devHashtable);

#ifdef DEBUG
  if (!finished)
    DEBUG_PRINT(("dp_knapsack_kernel tid = %d FINISHED i = %u w = %u profit = %u\n", tid,i,w,profit));
#endif
//  finished = true;
//  __threadfence();

  if (profit != 0)
  {
     //DEBUG_PRINT(("SET profit = %d (tid = %d)\n", profit, tid));
     *p = profit;
  }

#ifdef USE_INSTRUMENT
  knapsack_sum_stats();
  oahttslf_sum_stats();
#endif
}

/*****************************************************************************
 *
 * host main
 *
 *****************************************************************************/

/*
 * print usage message and exit
 *
 */
static void usage(const char *program)
{
  fprintf(stderr, 
          "Usage: %s [-ntvy]  < problemspec\n"
          "  -n: assume no name in the first line of the file\n"
          "  -t: show statistics of operations\n"
          "  -v: Verbose output\n"
          "  -y: show instrumentatino summary line (like -t but one line summary)\n",
          program);
  
  exit(EXIT_FAILURE);
}




/*
 * main
 */
int main(int argc, char *argv[])
{
  int i = 0;
  char flags[100];
  int c;
  int otime, ttime, etime;
  unsigned int profit =0;
  unsigned int *d_profit ;
  struct rusage starttime,totaltime,runtime,endtime,opttime;
  struct timeval start_timeval,end_timeval,elapsed_timeval;
  unsigned int t;
  char name[100];
  int noname = 0;
  cudaError_t rc;
  curandState *devStates;
  unsigned int hTimer;
  size_t stacksize;
  oahttslf_entry_t *devHashtable;


  strcpy(flags, "[NONE]");


  while ((c = getopt(argc, argv, "nvyt?")) != -1)
  {
    switch(c) {
      case 'v':
	/* verbose output */
	verbose = 1;
	break;
      case 't':
	/* show stats */
	printstats = 1;
	break;
      case 'n':
        /* no name on first line of input */
        noname = 1;
        break;
      case 'y':
        /* show statistics summaary line of insturmentation */
        show_stats_summary = 1;
        break;
      default:
        usage(argv[0]);
   	    break;
    }
    if (i < (int)sizeof(flags)-1)
      flags[i++] = c;
  }

  if (i > 0)
    flags[i] = '\0';

  /* we should have no command line parameters */
  if (optind != argc)
    usage(argv[0]);
 
	// Pick the best GPU available, or if the developer selects one at the command line
	int devID = cutilChooseCudaDevice(argc, argv);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devID);
	fprintf(stderr, "> GPU Device has Compute Capabilities SM %d.%d\n\n", deviceProp.major, deviceProp.minor);
	int version = (deviceProp.major * 0x10 + deviceProp.minor);
	if (version < 0x20) {
    fprintf(stderr, "device with compute capability 2.0 or better is required\n");
    exit(1);
  }

  // start time AFTER first CUDA call so as not to count the annoying
  // and apparently unavoidable approx. 4 second init overhead
  gettimeofday(&start_timeval, NULL);

  // We need L1 cache to store the stack (only applicable to sm_20 and higher)
  if ((rc = cudaFuncSetCacheConfig(dp_knapsack_kernel,
                                   cudaFuncCachePreferL1)) != cudaSuccess)
  {
    fprintf(stderr, "cudaFuncSetCacheConfig failed %d\n", rc);
    exit(1);
  }
  const int STACKSIZE = 65536; /* in bytes */
  if ((rc = cudaDeviceSetLimit(cudaLimitStackSize, STACKSIZE)) != cudaSuccess)
  {
    fprintf(stderr, "cudaDeviceSetLimit failed %d\n",rc);
    exit(1);
  }
  if ((rc = cudaDeviceGetLimit(&stacksize, cudaLimitStackSize)) != cudaSuccess)
  {
    fprintf(stderr, "cudaDeviceGetLimit failed %d\n",rc);
    exit(1);
  }
  fprintf(stderr, "cuda stack size = %.1f KB\n", (double)stacksize/1024);
  assert(stacksize == STACKSIZE);

  if (noname) 
    strcpy(name,"[NONE]\n");
  else
    fgets(name,sizeof(name)-1,stdin);


  getrusage(RUSAGE_SELF, &starttime);

  readdata(); /* read into the KNAPSACK_ITEMS array and set CAPACITY, NUM_KNAPSACK_ITEMS */

  if (NUM_KNAPSACK_ITEMS > MAX_KNAPSACK_ITEMS)
  {
    fprintf(stderr, 
            "num knapsack items %d exceeds %d, increase MAX_KNAPSACK_ITEMS\n",
            NUM_KNAPSACK_ITEMS, MAX_KNAPSACK_ITEMS);
    exit(1);
  }

  cutCreateTimer(&hTimer) ;
  cutResetTimer(hTimer) ;
  cutStartTimer(hTimer) ;

  /* copy the knapsack items to device constant memory */
  if ((rc = cudaMemcpyToSymbol(c_KNAPSACK_ITEMS_symbol, KNAPSACK_ITEMS, 
                               (1+NUM_KNAPSACK_ITEMS)*sizeof(item_t)))!= cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpyTosymbol failed %d\n", rc);
  }
  
  cutStopTimer(hTimer);
  fprintf(stderr, "copy %.1f KB of knapsack data to constant memory: %f ms\n",
          (double)sizeof(NUM_KNAPSACK_ITEMS*sizeof(item_t)/1024.0,
                         cutGetTimerValue(hTimer)));

  int blocks = 512;//FIXME some sensible value for this? (65535 max)



  cutResetTimer(hTimer) ;
  cutStartTimer(hTimer) ;

  /* allocate space on device for random number generator state */
  if ((rc = cudaMalloc((void **)&devStates, 
                       blocks*NUM_THREADS*sizeof(curandState))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc devStates failed %d\n", rc);
    exit(1);
  }
  
  /* initialize device random number generator */
  init_rng<<<blocks, NUM_THREADS>>>(devStates, time(NULL));
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "init_rng kernel error %d\n", rc);
  }
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "init_rng sync error %d\n", rc);
  }

  fprintf(stderr, "allocate and initialize CURAND device RNG for %d threads: %f ms\n",
          blocks*NUM_THREADS, cutGetTimerValue(hTimer));

  cutResetTimer(hTimer) ;
  cutStartTimer(hTimer) ;

  /* allocate hashtable on device */
  if ((rc = cudaMalloc((void **)&devHashtable, 
                       OAHTTSLF_SIZE*sizeof(oahttslf_entry_t))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc devHashtable failed %d\n", rc);
    exit(1);
  }
  cutStopTimer(hTimer) ;
  fprintf(stderr, "cudaMalloc %.1f MB hashtable elapsed time %d ms\n", 
          (double)OAHTTSLF_SIZE*sizeof(oahttslf_entry_t)/(1024*1024), cutGetTimerValue(hTimer));

  
  cutResetTimer(hTimer) ;
  cutStartTimer(hTimer) ;
  // Initialize the device memory
  
  oahttslf_reset<<<blocks, NUM_THREADS>>>(devHashtable);
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "oahttslf_reset kernel error %d\n", rc);
  }
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "oahttslf_reset sync error %d\n", rc);
  }

  cutStopTimer(hTimer) ;
  fprintf(stderr, "oahttslf_reset kernel elapsed time %d ms\n", 
          cutGetTimerValue(hTimer));



  if (cudaMalloc((void **)&d_profit, sizeof(unsigned int)) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc d_profit failed\n");
    exit(1);
  }

  cutResetTimer(hTimer) ;
  cutStartTimer(hTimer) ;
  
  /* Run the kernel */
  dim3 dimGrid(blocks)      ; // blocks
  dim3 dimBlock(NUM_THREADS); // threads

  dp_knapsack_kernel<<<dimGrid, dimBlock>>>(NUM_KNAPSACK_ITEMS, CAPACITY, d_profit, devStates, devHashtable);
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "knapsack kernel error %d\n", rc);
  }
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "knapsack sync error %d\n", rc);
  }
  cutStopTimer(hTimer) ;
  fprintf(stderr, "knapsack kernel time: %f ms\n", cutGetTimerValue(hTimer));
 
  cudaMemcpy(&profit, d_profit, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  
  getrusage(RUSAGE_SELF, &endtime);
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  /* timeval_subtract(&endtime,&starttime,&runtime); */
  runtime = endtime;
  ttime = 1000 * runtime.ru_utime.tv_sec + runtime.ru_utime.tv_usec/1000 
          + 1000 * runtime.ru_stime.tv_sec + runtime.ru_stime.tv_usec/1000;
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;


#ifdef USE_INSTRUMENT
  counter_t host_total_reuse, host_total_hashcount;
  if ((rc = cudaMemcpyFromSymbol(&host_total_reuse, "total_reuse",
                                 sizeof(counter_t))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpyFromSymbol total_reuse failed %d (%s)\n",
            rc, cudaGetErrorString(rc));
    exit(1);
  }
  if ((rc = cudaMemcpyFromSymbol(&host_total_hashcount, "total_hashcount",
                                 sizeof(counter_t))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpyFromSymbol total_hashcount failed %d (%s)\n",
            rc, cudaGetErrorString(rc));
    exit(1);
  }
#endif /* USE_INSTRUMENT */

  printf("%u %u %u %d %d %s %s", 
	 profit, 
#ifdef USE_INSTRUMENT
         host_total_reuse, host_total_hashcount,
#else
         0, 0,
#endif
         ttime, etime, flags, name);

  if (show_stats_summary)
  {
#ifdef USE_INSTRUMENT
    unsigned int num_keys, total_retry_count;
    if ((rc = cudaMemcpyFromSymbol(&total_retry_count, "global_retry_count",
                                   sizeof(counter_t))) != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpyFromSymbol total_retry_count failed %d (%s)\n",
              rc, cudaGetErrorString(rc));
      exit(1);
    }
    if ((rc = cudaMemcpyFromSymbol(&num_keys, "global_new_insert_count",
                                   sizeof(counter_t))) != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpyFromSymbol global_new_insert_count failed %d (%s)\n",
              rc, cudaGetErrorString(rc));
      exit(1);
    }
    printf("INSTRUMENT hc=%lu,re=%lu,re/hc=%f,hn=%u,or=%ld\n", 
           host_total_hashcount, host_total_reuse,
           (float)host_total_reuse / host_total_hashcount, 
           num_keys,
           total_retry_count
      );

    oahttslf_print_stats<<<dimGrid, dimBlock>>>();
    cutilDeviceSynchronize();
    if ((rc = cudaGetLastError()) != cudaSuccess)
    {
      fprintf(stderr, "oahttslf_print_stats sync error %d\n", rc);
    }
#else
    printf("COMPILED WITHOUT -DUSE_INSTRUMENT : NO STATS AVAIL\n");
#endif /* USE_INSTRUMENT */
  }



  /* clean up */
  cudaFree(devStates);
  cutilDeviceReset();
  free(KNAPSACK_ITEMS);
  exit(0);
  
}
