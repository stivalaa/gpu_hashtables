/*****************************************************************************
 *
 * File:    knapsack_threadcall.c
 * Author:  Alex Stivala
 * Created: April 2009
 *
 * $Id: knapsack_threadcall.c 4516 2013-01-03 23:46:45Z astivala $
 *
 * pthrads implementation using the httslf lockfree hashtable of 0/1
 * knapsack problem, paralleizing by starting recursive call in 
 * new thread (when thread limit not reached, else in same thread).
 *
 *  Usage: knapsack_threadcall [-tv] [-r maxthreads] < problemspec
 *          -t: show statistics of operations
 *          -v: Verbose output 
 *          -r: number of threads to use
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
 * gcc version 4.1.0 or greater is required, in order to use the
 * __sync_fetch_and_add() and __sync_fetch_and_sub() builtins
 * (this module was developed on Linux 2.6.22 (x86) with gcc 4.1.3)
 * except on Solaris, where atomic.h functions are used.
 *
 *
 * Preprocessor symbols:
 *
 * DEBUG          - compile in lots of debugging code.
 * USE_MUTEX      - use mutex not __sync_fetch_and_add() to serialize 
 *                  num_active_threads
 * USE_INSTRUMENT - compile in (per-thread) instrumentation counts.
 *                  Note that if this is not defined, then the -t and -i
 *                  options do not work.
 * SOLARIS        - compiling for SOLARIS, use atomic_ operations instead
 *                  of gcc __sync builtins
 *                  (TODO abstract this into atomicdefs.h)
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <assert.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef SOLARIS
#include <atomic.h>
#endif

#include "bpautils.h"
#include "httslf.h"


#undef USE_MUTEX /* mutex slows down dramatically, slower with more threads */

/* the master thead has id 0 (don't change this; we assume first in array) */
#define MASTER_THREAD_ID 0

#define DEFAULT_MAX_THREADS 1

void *dp_knapsack_thread(void *threadarg);
unsigned int dp_knapsack_thread_master(unsigned int i, unsigned int w);

/*****************************************************************************
 *
 * type definitions
 *
 *****************************************************************************/


/* definition of a type for a 2tuple */
/* 
 * The key for the hashtable is a 2-tuple, i.e. we look up by an (i,j)
 * tuple, where i,j would have been the indices of our 2-dimensional matrix
 */
typedef struct tuple2_s 
{
  unsigned int i;
  unsigned int j;
} tuple2_t;


typedef unsigned long counter_t;

typedef struct stats_s 
{
    counter_t count_dp_entry;  /* calls of dp */
    counter_t count_dp_entry_notmemoed; /* where not memo value return */
} stats_t;



typedef struct thread_data_s
{
    int thread_id; /* id of this thread to index into per-thread arrays */
    /* (i,w) for this thread to start at 
     *   i     - item index
     *   w     - total weight
     */
    unsigned int i;
    unsigned int w;
} thread_data_t;

/* definition of type for an item */
typedef struct item_s
{
    unsigned int profit;
    unsigned int weight;
} item_t;

/*****************************************************************************
 *
 * static data
 *
 *****************************************************************************/

static unsigned int max_threads = DEFAULT_MAX_THREADS; /* max number of threads allowed */
static bool printstats; /* whether to print call stats */
static bool verbose;    /* verbose output  */


static unsigned int CAPACITY; /* total capacity for the problem */
static unsigned int NUM_ITEMS; /* number of items */
static item_t *ITEMS;         /* array of item profits and weights (0 unused)*/

#ifdef USE_INSTRUMENT
/* instrumentation is per-thread, each thread only writes to its own element */
static stats_t stats[MAX_NUM_THREADS];
#endif


#ifdef USE_MUTEX
static pthread_mutex_t num_active_threads_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

/* the number of active threads. Serialize with __sync_fetch_and_add() etc */
static unsigned int num_active_threads = 1; /* we count the master thread */ 

/* instrumentatino totals (summed over all threads) */
counter_t  total_count_dp_entry = 0, total_count_dp_entry_notmemoed = 0;


/*****************************************************************************
 *
 * static functions
 *
 *****************************************************************************/


/*
 * httslf_hash()
 *
 * Compute hash value for the table.
 * We will simply shove the low 16 bits of each index in the tuple
 * into a word and return the value modulo the tablesize.
 *
 * Paramters:
 *     key - ptr to 2-tuple to compute hash value for
 *
 * Return value:
 *     hash value
 */
static unsigned int httslf_hash(const void *vkey) 
{
  unsigned int hashval;
  const tuple2_t *key = (const tuple2_t *)vkey;

  hashval = (key->i << 16) | (key->j & 0xffff);
  return hashval % HTTSLF_SIZE; 

}


/*
 * httslf_keymatch()
 *
 * Compare two key structures
 *
 * Parameters:
 *    s1 - key struct ptr for first
 *    s2 - key struct ptr for second
 *
 * Return value:
 *    nonzero if s1 and s2 are equal, else 0.
 *
 */
static int httslf_keymatch(const void *vs1, const void *vs2)
{
  const tuple2_t *s1 = (const tuple2_t *)vs1;
  const tuple2_t *s2 = (const tuple2_t *)vs2;

  return (s1->i == s2->i && s1->j == s2->j);
}



/* insert by (i,j) into table */
static void httslf_insert_indices(unsigned int i, unsigned int j, 
                                  unsigned int value, int thread_id);

/* lookup by (i,j) */
static void *httslf_lookup_indices(unsigned int i, unsigned int j);


/*
 * httslf_insert_indices()
 *
 * Insert value for (i,j) into the hashtable
 *
 * Parameters:
 *    i,j - indices to build insertion key
 *    value - value to insert for the key
 *    thread_id = therad id (0,1,..n) not pthrad_d
 *
 * Return value:
 *    None.
 */
static void httslf_insert_indices(unsigned int i, unsigned int j,
                                  unsigned int value, int thread_id)
{
  tuple2_t key;
  unsigned int uvalue;

  key.i = i;
  key.j = j;
  uvalue = value;
  httslf_insert(&key, &uvalue, thread_id);
}



/*
 * httslf_lookup_indices()
 *
 * Get the value for (i,j) from the hashtable
 *
 * Parameters:
 *     i,j - indices to build key for lookup
 * 
 * Return value:
 *     pointer to value for key, NULL if not found.
 */
static void *httslf_lookup_indices(unsigned int i, unsigned int j)
{
  tuple2_t key;

  key.i = i;
  key.j = j;
  return httslf_lookup(&key);
}




/*
 * dp_knapsack_thread_call()
 *
 * Utility function to start a new thread running
 * dp_knapsack_thread() if we  have not
 * reached the thread limit, otherwise call the function in this
 * thread
 *
 *      Parameters:   thread_id - our thread id for calling thread
 *                    new_thread_id (r/w) - thread id for new thread (or  own)
 *                    thread_data - structures passed to each thread
 *                    threads     - thread handles
 *                    actindex (r/w) - current index in above 2 arrays
 *                    i     - item index
 *                    w     - total weight
 *
 *      Uses global data:
 *                  read/write:
 *                     num_active_threads - number of active threads currently
 *                  readonly:
 *                    max_threads - number of threads we can use
 *
 *
 * Return value: None.
 * 
 *    The caller has to get the results from the hashtable at (i,w)
 *    when the thread (if there is one) has finished.
 */
static void dp_knapsack_thread_call(int thread_id, 
                                    int *new_thread_id,
                                    thread_data_t thread_data[],
                                    pthread_t threads[],
                                    int *actindex,
                                    unsigned int i, unsigned int w)
{
  static const char *funcname = "dp_knapsack_thread_call";

  thread_data_t dummy_thread_data; /* for passing parameter in same thread */
  int rc;
  unsigned int old_num_active_threads;

#ifdef USE_MUTEX
  pthread_mutex_lock(&num_active_threads_mutex);
  old_num_active_threads = num_active_threads;
#else
#ifdef SOLARIS
  old_num_active_threads = atomic_inc_uint_nv(&num_active_threads) - 1;
#else
  old_num_active_threads = num_active_threads;    
  while (old_num_active_threads < max_threads &&
         __sync_fetch_and_add(&num_active_threads,1) != old_num_active_threads)
    old_num_active_threads = num_active_threads ;
#endif
#endif /* USE_MUTEX */

  if (old_num_active_threads < max_threads)
  {
    fprintf(stderr,"starting thread id %d\n", old_num_active_threads);
    *new_thread_id = old_num_active_threads;
    thread_data[*actindex].thread_id = old_num_active_threads;
    thread_data[*actindex].i = i;
    thread_data[*actindex].w = w;
#ifdef USE_MUTEX
    num_active_threads++;
    pthread_mutex_unlock(&num_active_threads_mutex);
#endif
    if ((rc = pthread_create(&threads[*actindex], NULL,
                             dp_knapsack_thread,
                             (void *)&thread_data[*actindex])))
      bpa_fatal_error(funcname, "pthread_create() failed (%d)\n", rc);

    (*actindex)++;
  }
  else
  {
#ifdef USE_MUTEX
    pthread_mutex_unlock(&num_active_threads_mutex);
#endif
    *new_thread_id = thread_id;
    dummy_thread_data.thread_id = thread_id;
    dummy_thread_data.i = i;
    dummy_thread_data.w = w;
    dp_knapsack_thread((void *)&dummy_thread_data);
  }
}


/*****************************************************************************
 *
 * external functions
 *
 *****************************************************************************/


/*
 * dp_knapsack_thread()
 *
 *
 *      This version is multi-threaded, sharing hashtable used to
 *      store computed S values between the threads.
 *
 *      This version is the memory function version; instead of computing
 *      the whole  array bottom-up, it is computed recursively top-down
 *      and values stored in it, and reused (memoization) if already
 *      computed. 
 *
 *      This version uses no bounding.
 *
 *
 *
 *      Parameters:   threadarg - thread data for this thread
 *
 *
 *      Uses global data:
 *                  read/write:
 *                    num_active_threads -number of running threads
 *                    count_dp_entry[thread_id] - count of calls here
 *                    count_dp_notmemoed[thread_id] -
 *                                              where not return memoed value
 *
 *                  readonly:
 *                    max_threads - number of threads we can use
 *
 *      Return value: NULL
 *                     (Declared void * for pthreads)
 *
 */
void *dp_knapsack_thread(void *threadarg)
{
  static const char *funcname = "dp_knapsack_thread";
  unsigned int i,w,p,pwithout,pwith;
  bool comp_notp, comp_pwith;
  int rc;
  int t;

  /* structures passed to each thread  as parameter */
  thread_data_t thread_data[MAX_NUM_THREADS];

  /* pthreads thread handles */
  pthread_t threads[MAX_NUM_THREADS];

  int actindex = 0;
  int new_thread_id;
  unsigned int old_num_active_threads;
  thread_data_t *mydata = (thread_data_t *)threadarg;
  
  i = mydata->i;
  w = mydata->w;

#ifdef DEBUG
  bpa_log_msg(funcname, "%d\t\t%d\t%d\n",mydata->thread_id,i,w);
#endif

#ifdef USE_INSTRUMENT
  stats[mydata->thread_id].count_dp_entry++;
#endif

  /* memoization: if value here already computed then do nothing */
  if (httslf_lookup_indices(i, w) != NULL)
    return NULL;

#ifdef USE_INSTRUMENT
  stats[mydata->thread_id].count_dp_entry_notmemoed++;
#endif

  comp_notp = FALSE; comp_pwith = FALSE;
  if (i == 0)
  {
    p = 0;
  }
  else if (w < ITEMS[i].weight)
  {
    dp_knapsack_thread_call(mydata->thread_id,
                            &new_thread_id,
                            thread_data, threads, &actindex,
                            i - 1, w);
    comp_notp = TRUE;
  }
  else
  {
    dp_knapsack_thread_call(mydata->thread_id,
                            &new_thread_id,
                            thread_data, threads, &actindex,
                            i - 1, w);

    dp_knapsack_thread_call(mydata->thread_id,
                            &new_thread_id,
                            thread_data, threads, &actindex,
                            i - 1, w - ITEMS[i].weight);

    comp_pwith = TRUE;
  }

  /* FIXME TODO for now we will wait for all threads here.
     We should change this so that instead of crude starting of threads
     we maintain a pool of threads that signal when done, then we can
     use those values and start them with new work. PRobably not even
     start/end threads, just start them all, and use mutex/condition variables
     to co-ordinate */
  for (t = 0; t < actindex; t++)
  {
    fprintf(stderr, "%d / %d\n", thread_data[t].thread_id,num_active_threads);
    if ((rc = pthread_join(threads[t], NULL)))
      bpa_fatal_error(funcname, "pthread_join failed (%d)\n", rc);

#ifdef USE_MUTEX
    pthread_mutex_lock(&num_active_threads_mutex);
    num_active_threads--;
    pthread_mutex_unlock(&num_active_threads_mutex);
#else
#ifdef SOLARIS
    atomic_dec_uint(&num_active_threads);
#else
    do
      old_num_active_threads = num_active_threads;    
    while (__sync_fetch_and_sub(&num_active_threads,1)!=old_num_active_threads);
#endif
#endif /* USE_MUTEX */


  }

  /* get values from hashtable. They must be there as either calls
     were synchronous or the thread has been joined. */
  if (comp_pwith)
  {
    pwithout = *(unsigned int *)httslf_lookup_indices(i - 1, w);
    pwith = *(unsigned int *)httslf_lookup_indices(i - 1, w - ITEMS[i].weight) 
      + ITEMS[i].profit;

    p = MAX(pwithout, pwith);
  }
  else if (comp_notp)
  {
    p = *(unsigned int *)httslf_lookup_indices(i - 1, w);
  }

#ifdef DEBUG
  bpa_log_msg(funcname, "%d\tS\t%d\t%d\t%d\n",mydata->thread_id,i,w,p);
#endif
  httslf_insert_indices(i, w, p, mydata->thread_id);
  return NULL;
}

#ifdef USE_INSTRUMENT
/*
 * compute_total_counts() 
 *
 *    sum the per-thread stats into total stats
 *
 *
 *      Parameters: None.
 *      Return value: None.
 *      Uses global data:
 *                  read/write:
 *                    total_count_dp_entry 
 *                    total_count_dp_entry_notmemoed
 *                  readonly:
 *                    stats
 *                    max_threads -max number of threads allowed
 */
static void compute_total_counts()
{
  unsigned int t;
  total_count_dp_entry = 0;
  total_count_dp_entry_notmemoed = 0;
  for (t = 0; t < max_threads; t++)
  {
    total_count_dp_entry += stats[t].count_dp_entry;
    total_count_dp_entry_notmemoed += stats[t].count_dp_entry_notmemoed;
  }
}

#endif


/*
 * dp_knapsack_thread_master()
 *
 *   Caller interface to the multithreaded version:
 *   just calls the actual implementation after setting up thread
 *   parameter block so that the instance called in this thread
 *   is the "master" thread (thread_id 0).
 *
 *      Parameters:   i     - item index
 *                    w     - total weight (capacity )
 *
 *      Uses global data:
 *                  read/write:
 *                    hashtable in httslf.c
 *                    total_count_dp_entry 
 *                    total_count_dp_entry_notmemoed
 *                  readonly:
 *                    printstats -flag to print instrumentation data
 *                    max_threads -max number of threads allowed
 *
 *      Return value: value of d.p. at (i,w).
 *
 */
unsigned int dp_knapsack_thread_master(unsigned int i, unsigned int w)
{
  thread_data_t master_thread_data;
  httslf_entry_t *ent;
  unsigned int t;


  master_thread_data.thread_id = MASTER_THREAD_ID;
  master_thread_data.i = i;
  master_thread_data.w = w;

  /* run master in this thread */
  ent = dp_knapsack_thread(&master_thread_data);

  if (printstats)
  {
#ifdef USE_INSTRUMENT
    for (t = 0; t < max_threads; t++)
    {
      printf("stats for thread %d:\n", t);
      printf("  calls to dp = %lu\n", stats[t].count_dp_entry);
      printf("  calls to dp where not memoed = %lu\n",
             stats[t].count_dp_entry_notmemoed);
    }
    compute_total_counts();
    printf("totals:\n");
    printf("  calls to dp = %lu\n", total_count_dp_entry);
    printf("  calls to dp where not memoed = %lu\n",
           total_count_dp_entry_notmemoed);
#else
    printf("COMPILED WITHOUT -DUSE_INSTRUMENT : NO STATS AVAIL\n");
#endif
  }
  return *(unsigned int *)httslf_lookup_indices(i, w);
}


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
 *      ITEMS        - allocates array, sets profit and weight for each item
 *      CAPACITY     - sets capacity for problem
 *      NUM_ITEMS   - number of items
 */
static void readdata(void)
{
  int i,inum;

  if (scanf("%d", &NUM_ITEMS) != 1)
  {
    fprintf(stderr, "ERROR reading number of items\n");
    exit(EXIT_FAILURE);
  }
  ITEMS = (item_t *)bpa_malloc((NUM_ITEMS+1) * sizeof(item_t));
  for (i = 1; i <= NUM_ITEMS; i++)
  {
    if(scanf("%d %d %d", &inum, &ITEMS[i].profit, &ITEMS[i].weight) != 3)
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

/*
 * print usage message and exit
 *
 */
static void usage(const char *program)
{
  fprintf(stderr, 
          "Usage: %s [-tv] [-r maxthreads] < problemspec\n"
          "  -t: show statistics of operations\n"
          "  -v: Verbose output\n"
          "  -r: number of worker threads to use (default %d)\n",
          program, DEFAULT_MAX_THREADS);
  
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
  int profit;
  struct rusage starttime,totaltime,runtime,endtime,opttime;
  struct timeval start_timeval,end_timeval,elapsed_timeval;
  char *name = "FIXME";

  gettimeofday(&start_timeval, NULL);

  while ((c = getopt(argc, argv, "vtr:?")) != -1)
  {
    switch(c) {
      case 'r':
        /* number of worker threads */
        if (atoi(optarg) < 1)
        {
          fprintf(stderr, "number of worker threads must be >= 1\n");
          usage(argv[0]);
        }
        else if (atoi(optarg) > MAX_NUM_THREADS)
        {
          fprintf(stderr, "maximum number of threads is %d\n", MAX_NUM_THREADS);
          usage(argv[0]);
        }
        max_threads = atoi(optarg);
        break;
      case 'v':
	/* verbose output */
	verbose = 1;
        bpa_set_verbose(verbose);
	break;
      case 't':
	/* show stats */
	printstats = 1;
	break;
      default:
        usage(argv[0]);
	break;
    }
    if (i < (int)sizeof(flags)-1)
      flags[i++] = c;
  }
  flags[i] = '\0';

  /* we should have no command line parameters */
  if (optind != argc)
    usage(argv[0]);
  
  httslf_initialize(sizeof(tuple2_t), sizeof(unsigned int),
                    httslf_hash, NULL, 
                    httslf_keymatch, NULL, MAX_NUM_THREADS);

  getrusage(RUSAGE_SELF, &starttime);

  readdata(); /* read into the ITEMS array and set CAPACITY, NUM_ITEMS */
  profit = dp_knapsack_thread_master(NUM_ITEMS, CAPACITY);

  getrusage(RUSAGE_SELF, &endtime);
  gettimeofday(&end_timeval, NULL);
  timeval_subtract(&elapsed_timeval, &end_timeval, &start_timeval);
  /* timeval_subtract(&endtime,&starttime,&runtime); */
  runtime = endtime;
  ttime = 1000 * runtime.ru_utime.tv_sec + runtime.ru_utime.tv_usec/1000 
          + 1000 * runtime.ru_stime.tv_sec + runtime.ru_stime.tv_usec/1000;
  etime = 1000 * elapsed_timeval.tv_sec + elapsed_timeval.tv_usec/1000;
#ifdef USE_INSTRUMENT
  compute_total_counts();
#else
  total_count_dp_entry_notmemoed = 0;
  total_count_dp_entry = 0;
#endif
  printf("%d %d %d %d %d %s %s\n", 
	 profit, total_count_dp_entry, total_count_dp_entry_notmemoed,
         ttime, etime, flags, name);
  free(ITEMS);
  exit(0);
  
}
