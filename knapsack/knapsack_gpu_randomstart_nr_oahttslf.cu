/*****************************************************************************
 *
 * File:    knapsack_gpu_randomstart_oahttslf.cu
 * Author:  Alex Stivala & Peter J Stuckey
 * Created: April 2009
 *
 * $Id: knapsack_gpu_randomstart_nr_oahttslf.cu 4616 2013-01-25 01:07:43Z astivala $
 *
 * This is the GPU implementation using the oahttslf GPU lock free hash table.
 * This version does not use recursion.
 * This version has an additional feature in which we try
 * stating some threads at random (i,w) positions in the problem rather
 * than starting all at the top (solution) values of (i, w)
 *
 *
 *  Usage: knapsack_gpu_randomstart_oahttslf [-nvy] [-r threads]  < problemspec
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
 * as it uses atomic CAS on 64 bits.
 * Also printf() from device function for debug.
 * It uses the CURAND library for pseudrandom number generation.
 *
 * Preprocessor symbols:
 *
 * DEBUG          - compile in lots of debugging code.
 * USE_SHARED     - use shared memory not global for stacks
 * USE_HTTSLF     - use httslf gpu kernel instead of oahttslf gpu kerne
 * USE_INSTRUMENT - compile in (per-thread) instrumentation counts.
 *
 *****************************************************************************/

#define NOTDEF_XXX_NORAND /* yes we DO want randomizatino */

#undef USE_SHARED

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "cutil_inline.h"
#include "curand_kernel.h"

// number of thread blocks
#define NUM_BLOCKS  512 //some sensible value for this? (65535 max)

// Number of threads per block
#define NUM_THREADS 32 //some sensible value for this? (1024 max)
// (these are defined prior to inclde of httslf_gpu_oahttslf_kernel.cu etc
// as they are used to define arrays for instrumentation if USE_INSTRUMENT)

#ifdef USE_HTTSLF
#undef ALLOW_DELETE   /* never need to delete keys */
#include "httslf_gpu_kernel.cu"
#define HASHTABLE_TYPE httslf_entry_t**
#define INSERT_FUNCTION httslf_insert
#define LOOKUP_FUNCTION httslf_lookup
#else
#undef ALLOW_UPDATE   /* never need to update a value once insterted */
#undef ALLOW_DELETE   /* never need to delete keys */
#include "oahttslf_gpu_kernel.cu"
#define HASHTABLE_TYPE oahttslf_entry_t*
#define INSERT_FUNCTION oahttslf_insert
#define LOOKUP_FUNCTION oahttslf_lookup
#endif

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

#define MAXITEMS 1024  /*  used for stack size */

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

typedef unsigned long long uint64_t;
typedef unsigned int       uint32_t;
typedef unsigned short     uint16_t;
typedef unsigned char      uint8_t;




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

#ifdef USE_SHARED
//
// per-thread stacks, using shared memory for speed
//
// index as [ threadIdx.x * (MAXITEMS+1) + stackindex ]
#define STACKINDEX(i) (threadIdx.x * (MAXITEMS+1) + (i))
__shared__ bool stackc[NUM_THREADS * (MAXITEMS+1)];  /* call or answer */
__shared__ unsigned int stacki[NUM_THREADS * (MAXITEMS+1)];
__shared__ unsigned int stackw[NUM_THREADS * (MAXITEMS+1)];
#else
// instead of shared, use global memory allocated with cudaMalloc
// (to avoid linker errors if size gets too large)
// index as [ thread_id * (MAXITEMS+1) + stackindex ] 
// NB the macro below dpends on local variable 
//  tid = blockIdx.x*blockDim.x+threadIdx.x
#define STACKINDEX(i) (tid * (MAXITEMS+1) + (i))
#define TOTAL_NUM_THREADS (NUM_THREADS * NUM_BLOCKS)
__device__ bool *stackc;  /* call or answer */
__device__ unsigned int *stacki;
__device__ unsigned int *stackw;
#endif /* USE_SHARED */

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
 * insert_indices()
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
                               unsigned int value,  HASHTABLE_TYPE devHashtable)
{
  uint64_t key, val64;
#ifdef USE_HTTSLF
  key = ((uint64_t)i << 32) | (j & 0xffffffff);
  val64 = (uint64_t)value;
#else
  key = (i == 0 && j == 0 ? MAGIC_ZERO : 
         ((uint64_t)i << 32) | (j & 0xffffffff));
  val64 = (value == 0 ? MAGIC_ZERO : (uint64_t)value);
#endif
#ifdef USE_INSTRUMENT
  stats[threadIdx.x].hashcount++;
#endif
  INSERT_FUNCTION(devHashtable, key, val64);
}



/*
 * lookup_indices()
 *
 * Get the value for (i,j) from the hashtable
 *
 * Parameters:
 *     i,j - indices to build key for lookup
 *     pvalue - (OUTPUT) value for key, only set if true returned
 * 
 * Uses global data:
 *    devHashTable (readonly)
 *
 * Return value:
 *     true if found, false otherwise
 */
__device__ bool lookup_indices(unsigned int i, unsigned int j, 
                               unsigned int *pvalue, HASHTABLE_TYPE devHashtable)
{
  uint64_t key;
  uint64_t val64 = 0;
  bool found;

#ifdef USE_HTTSLF
  key = ((uint64_t)i << 32) | (j & 0xffffffff);
#else
  key = (i == 0 && j == 0 ? MAGIC_ZERO :
         ((uint64_t)i << 32) | (j & 0xffffffff));
#endif
  found = LOOKUP_FUNCTION(devHashtable, key, &val64);
  // avoid branch by using ternay operator to set to 0 if not found
  // since val64 is initalized to 0 (note this depends on LOOKUP_FUNCTION
  // not setting return parmater &val64 if not found)
  *pvalue = ((val64 == MAGIC_ZERO) ? 0 : (unsigned int)val64); 
  return found;
}


/*
 * lookk - lookup a computed value for (i,w) from the hash table
 *
 * Parameters:
 *     i,w - indices to build key for lookup
 *     pvalue - (OUTPUT) value for key, only set if true returned
 *     devHashtable - the hash table on the device
 * Return value:
 *     The value p (profit) in the table for (i,w) or 0 if not there
 *
 */
__device__ unsigned int lookk(unsigned int i, unsigned int w,
                              HASHTABLE_TYPE devHashtable) 
{
  unsigned int p = 0;
  lookup_indices(i, w, &p, devHashtable);
  return p;
}



/*
 * dp_knapsack_nr()
 *
 *      This version is multi-threaded, sharing hashtable used to
 *      store computed values between the threads.
 *      This version is not recursive, it maintains its own explicit stack.
 *      This function is called by dp_knapsack_thread()
 *      with identical instances running
 *      in several threads. This functino itself is not recursive
 *      (and never creates threads itself) and diverges as there is 
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
 *
 */
__device__ unsigned int dp_knapsack_nr(unsigned int i, unsigned int w,
                                       curandState *state, HASHTABLE_TYPE devHashtable)
{
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  curandState localState = state[tid];
  unsigned int p;
  bool c;
  int top = 1;
  unsigned int oi = i, ow = w;
 

#ifdef DEBUG
  printf("knapsack_nr(%d,%d)\n",i,w);
#endif 

  stacki[STACKINDEX(top)] = i;
  stackw[STACKINDEX(top)] = w;
  stackc[STACKINDEX(top)] = true;
  while(top) {
    assert(threadIdx.x < NUM_THREADS);
    assert(top <= MAXITEMS);
    if (finished)
      return 0;
    i = stacki[STACKINDEX(top)];
    w = stackw[STACKINDEX(top)];
    c = stackc[STACKINDEX(top)];
    top--;
#ifdef DEBUG
    printf("knapsack_nr(%d,%d,%d)\n",i,w,c);
#endif    
    if (c) {
      if (i == 1) { 
	p = (w >= c_KNAPSACK_ITEMS[i].weight ? c_KNAPSACK_ITEMS[i].profit : 0);
#ifdef DEBUG
        printf("knapsack_nr(%d,%d,%d) = %d\n",i,w,c,p);
#endif    	
        insert_indices(i, w, p, devHashtable);
      }
      else if (lookk(i-1,w,devHashtable) > 0 && lookk(i-1,w - c_KNAPSACK_ITEMS[i].weight,devHashtable) > 0) {
	p = MAX(lookk(i-1,w,devHashtable), 
                (w >= c_KNAPSACK_ITEMS[i].weight ? lookk(i-1,w - c_KNAPSACK_ITEMS[i].weight,devHashtable) + c_KNAPSACK_ITEMS[i].profit : 0));
#ifdef DEBUG
        printf("knapsack_nr(%d,%d,%d) = %d\n",i,w,c,p);
#endif    	
        insert_indices(i, w, p, devHashtable);
      }
      else {
#ifdef NOTDEF_XXX_NORAND
      if (curand(&localState) & 1)  {
    	top++;
    	stacki[STACKINDEX(top)] = i;
    	stackw[STACKINDEX(top)] = w;
    	stackc[STACKINDEX(top)] = false;
    	if (i >= 1 && lookk(i-1,w,devHashtable) == 0) {
    	  top++;
    	  stacki[STACKINDEX(top)] = i-1;
    	  stackw[STACKINDEX(top)] = w;
    	  stackc[STACKINDEX(top)] = true;
    	}
    	if (i >= 1 && w >= c_KNAPSACK_ITEMS[i].weight && lookk(i-1,w-c_KNAPSACK_ITEMS[i].weight,devHashtable) == 0) {
    	  top++;
    	  stacki[STACKINDEX(top)] = i-1;
    	  stackw[STACKINDEX(top)] = w - c_KNAPSACK_ITEMS[i].weight;
    	  stackc[STACKINDEX(top)] = true;
    	}
     }
    else {
#endif /*NOTDEF_XXX_NORAND*/
    	top++;
    	stacki[STACKINDEX(top)] = i;
    	stackw[STACKINDEX(top)] = w;
    	stackc[STACKINDEX(top)] = false;
    	if (i >= 1 && w >= c_KNAPSACK_ITEMS[i].weight && lookk(i-1,w-c_KNAPSACK_ITEMS[i].weight,devHashtable) == 0) {
    	  top++;
    	  stacki[STACKINDEX(top)] = i-1;
    	  stackw[STACKINDEX(top)] = w - c_KNAPSACK_ITEMS[i].weight;
    	  stackc[STACKINDEX(top)] = true;
    	}
    	if (i >= 1 && lookk(i-1,w,devHashtable) == 0) {
    	  top++;
    	  stacki[STACKINDEX(top)] = i-1;
    	  stackw[STACKINDEX(top)] = w;
    	  stackc[STACKINDEX(top)] = true;
    	}
#ifdef NOTDEF_XXX_NORAND
    }
#endif /*NOTDEF_XXX_NORAND*/
    }
    } else {
      p = MAX(lookk(i-1,w,devHashtable), 
	      (w >= c_KNAPSACK_ITEMS[i].weight ? lookk(i-1,w - c_KNAPSACK_ITEMS[i].weight,devHashtable) +c_KNAPSACK_ITEMS[i].profit : 0));
#ifdef DEBUG
      printf("knapsack_nr(%d,%d,%d) = %d\n",i,w,c,p);
#endif    	
      insert_indices(i, w, p, devHashtable);
    }
  }
  insert_indices(oi, ow, p, devHashtable);
#ifdef DEBUG
  printf("knapsack_nr(%d,%d) = %d\n",i,w,p);
#endif 
  state[tid] = localState;
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
                                   HASHTABLE_TYPE devHashtable)
{
 int tid=blockIdx.x*blockDim.x+threadIdx.x;
 unsigned int profit = 0;

#ifdef USE_INSTRUMENT
  reset_counters();
#endif

  //DEBUG_PRINT(("(0) blockIdx.x = %d blockDim.x = %d threadIx = %d tid = %d\n",
  //             blockIdx.x, blockDim.x, threadIdx.x, tid));
  

#ifdef DEBUG
  for (int j = 0;  j < i; j++) 
  {
    printf("item %d: weight = %d profit = %d\n", j, c_KNAPSACK_ITEMS[j].weight, c_KNAPSACK_ITEMS[j].profit);
  }
#endif

  /* start half the threads at a random point. These threads will be
   * in a "forever" loop, we only care about threads actually computing
   * the required solution terminating - when one of them terminates the
   * solution is found. The random threaed just go and start at another
   * random point if they finish first.
   */
  bool randomstart = (tid > 31);
  if (randomstart)
  {
    /* experimental "random" case: just keep starting at random points
     * hoping we compute something helpful to the solution. The idea is
     * to avoid large growth in unncessary recomputation (h/h_0) for large
     * number of threads
     * TODO more sensible choices of points, not totally random
     */  
    while (!finished)
    {
      curandState localState = state[tid];
      unsigned int random_i = curand(&localState) % i;
      unsigned int random_w = curand(&localState) % w;
      state[tid] = localState;
      (void)dp_knapsack_nr(random_i, random_w, state, devHashtable);
    }
  }
  else
  {
    /* this thread starts at the (i,w) value to solve the actual problem */
     profit = dp_knapsack_nr(i, w, state, devHashtable);
     finished = true;
     __threadfence();// FIXME do we need some sort of sync or threadfence here?
     //DEBUG_PRINT(("SET profit = %d (tid = %d)\n", profit, tid));
     if (profit != 0)
       *p = profit;
  }
#ifdef USE_INSTRUMENT
#ifdef USE_HTTSLF
  httslf_sumcounters();
#else
  oahttslf_sum_stats();
#endif /* USE_HTTSLF */
  knapsack_sum_stats();
#endif /* USE_INSTRUMENT */
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
  HASHTABLE_TYPE devHashtable;

#ifndef NOTDEF_XXX_NORAND
  fprintf(stderr, "NO RANDOMIZATION\n");
#endif

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
  if (verbose)
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
  // per-thread stacks, hard limit of total of 512KB
  // local memory per thread (where stack is stored), so cannot use
  // all of that for stack either
  // see http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
  const int STACKSIZE = 1024; /* in bytes */
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
  if (verbose)
    fprintf(stderr, "cuda stack size = %.1f KB\n", (double)stacksize/1024);

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
  if (NUM_KNAPSACK_ITEMS > MAXITEMS)
  {
    fprintf(stderr, "number of items %d is too large, increase MAXITEMS\n",
        NUM_KNAPSACK_ITEMS);
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
  if (verbose)
    fprintf(stderr, "copy %.1f KB of knapsack data to constant memory: %f ms\n",
            (double)sizeof(NUM_KNAPSACK_ITEMS*sizeof(item_t)/1024.0),
                           cutGetTimerValue(hTimer));



  cutResetTimer(hTimer) ;
  cutStartTimer(hTimer) ;

  /* allocate space on device for random number generator state */
  if ((rc = cudaMalloc((void **)&devStates, 
                       NUM_BLOCKS*NUM_THREADS*sizeof(curandState))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc devStates failed %d\n", rc);
    exit(1);
  }
  
  /* initialize device random number generator */
  init_rng<<<NUM_BLOCKS, NUM_THREADS>>>(devStates, time(NULL));
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "init_rng kernel error %d\n", rc);
  }
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "init_rng sync error %d\n", rc);
  }

  if (verbose)
    fprintf(stderr, "allocate and initialize CURAND device RNG for %d threads: %f ms\n",
          NUM_BLOCKS*NUM_THREADS, cutGetTimerValue(hTimer));

  cutResetTimer(hTimer) ;
  cutStartTimer(hTimer) ;

#ifdef USE_HTTSLF
  httslf_entry_t **devCellpool;
  gettimeofday(&start_timeval, NULL);
  /* allocate cell pool on device */
  size_t cell_size = sizeof(httslf_entry_t);
  unsigned int devCellpool_num_items = 67108864 ;  /* 2^26 */
  if (verbose)
    fprintf(stderr, "devCellpool_num_items = %u\n", devCellpool_num_items);
  size_t devCellpool_size = cell_size * devCellpool_num_items;
  if ((rc = cudaMalloc((void **)&devCellpool, devCellpool_size)) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc devCellpool failed %d\n", rc);
    exit(1);
  }

  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
  if (verbose)
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
  if (verbose)
    fprintf(stderr, "cudaMalloc %.1f MB hashtable elapsed time %d ms\n", 
          (double)HTTSLF_SIZE*sizeof(httslf_entry_t *)/(1024*1024), etime);
#else
  /* allocate hashtable on device */
  if ((rc = cudaMalloc((void **)&devHashtable, 
                       OAHTTSLF_SIZE*sizeof(oahttslf_entry_t))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc devHashtable failed %d\n", rc);
    exit(1);
  }
  cutStopTimer(hTimer) ;
  if (verbose)
    fprintf(stderr, "cudaMalloc %.1f MB hashtable elapsed time %d ms\n", 
            (double)OAHTTSLF_SIZE*sizeof(oahttslf_entry_t)/(1024*1024), cutGetTimerValue(hTimer));
#endif /* USE_HTTSLF*/

#ifndef USE_SHARED
  /* allocate the per-thread stacks in device global memory */
  /* cudaMemcpyToSymbol() the pointers to allocated device memory to the
     global device pionteers rather than passing as parameters for 
     convenience so code is same as using shared memory, just using macros */
  bool *cuda_stackc;  /* call or answer */
  unsigned int *cuda_stacki;
  unsigned int *cuda_stackw;
  if ((rc = cudaMalloc((void **)&cuda_stackc, 
           sizeof(bool) * TOTAL_NUM_THREADS * (MAXITEMS+1))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc stackc failed %d (%s)\n", rc, cudaGetErrorString(rc));
    exit(1);
  }
  if ((rc = cudaMemcpyToSymbol("stackc", &cuda_stackc, sizeof(bool*))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpyTosymbol stackc failed %d (%s)\n", rc,
              cudaGetErrorString(rc));
    exit(1);
  }
  
  if ((rc = cudaMalloc((void **)&cuda_stacki, 
           sizeof(unsigned int) * TOTAL_NUM_THREADS * (MAXITEMS+1))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc stacki failed %d (%s)\n", rc, cudaGetErrorString(rc));
    exit(1);
  }
  if ((rc = cudaMemcpyToSymbol("stacki", &cuda_stacki, sizeof(unsigned int*))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpyTosymbol stacki failed %d (%s)\n", rc,
              cudaGetErrorString(rc));
    exit(1);
  }
  
  if ((rc = cudaMalloc((void **)&cuda_stackw, 
           sizeof(unsigned int) * TOTAL_NUM_THREADS * (MAXITEMS+1))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc stackw failed %d (%s)\n", rc, cudaGetErrorString(rc));
    exit(1);
  }
  if ((rc = cudaMemcpyToSymbol("stackw", &cuda_stackw, sizeof(unsigned int*))) != cudaSuccess)
  {
    fprintf(stderr, "cudaMemcpyTosymbol stackw failed %d (%s)\n", rc,
              cudaGetErrorString(rc));
    exit(1);
  }
#endif /* USE_SHARED */
  
#ifdef USE_HTTSLF
  gettimeofday(&start_timeval, NULL);
  /* set hashtable to all empty keys/values */
  httslf_reset<<<NUM_BLOCKS, NUM_THREADS>>>(devHashtable);
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
  if (verbose)
    fprintf(stderr, "httslf_reset elapsed time %d ms\n", etime);
#else
  cutResetTimer(hTimer) ;
  cutStartTimer(hTimer) ;
  // Initialize the device memory
  
  oahttslf_reset<<<NUM_BLOCKS, NUM_THREADS>>>(devHashtable);
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
  if (verbose)
    fprintf(stderr, "oahttslf_reset kernel elapsed time %d ms\n", 
            cutGetTimerValue(hTimer));

#endif /*USE_HTTSLF*/

  if (cudaMalloc((void **)&d_profit, sizeof(unsigned int)) != cudaSuccess)
  {
    fprintf(stderr, "cudaMalloc d_profit failed\n");
    exit(1);
  }

  cutResetTimer(hTimer) ;
  cutStartTimer(hTimer) ;
 
  if (verbose)
    fprintf(stderr, "NUM_BLOCKS = %d, NUM_THREADS = %d\n", NUM_BLOCKS,NUM_THREADS);

  /* Run the kernel */
  dim3 dimGrid(NUM_BLOCKS)      ; // blocks
  dim3 dimBlock(NUM_THREADS); // threads

  dp_knapsack_kernel<<<dimGrid, dimBlock>>>(NUM_KNAPSACK_ITEMS, CAPACITY, d_profit, devStates, devHashtable);
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "knapsack kernel error %d (%s)\n", rc, cudaGetErrorString(rc));
  }
  cutilDeviceSynchronize();
  if ((rc = cudaGetLastError()) != cudaSuccess)
  {
    fprintf(stderr, "knapsack sync error %d (%s)\n", rc, cudaGetErrorString(rc));
  }
  cutStopTimer(hTimer) ;
  if (verbose)
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

  printf("%u %d %d %d %d %s %s", 
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



#ifdef USE_HTTSLF
    httslf_computestats<<<dimGrid, dimBlock>>>(devHashtable);
    cutilDeviceSynchronize();
    if ((rc = cudaGetLastError()) != cudaSuccess)
    {
      fprintf(stderr, "httslf_computestats sync error %d (%s)\n", rc,cudaGetErrorString(rc));
      exit(1);
    }
    httslf_printstats<<<dimGrid, dimBlock>>>();
    cutilDeviceSynchronize();
    if ((rc = cudaGetLastError()) != cudaSuccess)
    {

      fprintf(stderr, "httslf_printstats sync error %d (%s)\n", rc,cudaGetErrorString(rc));

      exit(1);
    }
#else
    oahttslf_print_stats<<<dimGrid, dimBlock>>>();
    cutilDeviceSynchronize();
    if ((rc = cudaGetLastError()) != cudaSuccess)
    {
      fprintf(stderr, "oahttslf_print_stats sync error %d\n", rc);
    }
#endif /* USE_HTTSLF */
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

      fprintf(stderr, "httslf_printstats sync error %d (%s)\n", rc,cudaGetErrorString(rc));

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
