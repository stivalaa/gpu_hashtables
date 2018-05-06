/**********************************************************************************

 Test harness fo Lock-free hash table for POSIX threads
 Developed at IIT Kanpur.
 This version changed by ADS to use the oahttslf hash table for comparison
 under identical conditions with the IIT Kanpur LockFreeHashTable

 Inputs: Percentage of add and delete operations (e.g., 30 50 for 30% add and 50% delete)
 Output: Prints the total time (in milliseconds) to execute the the sequence of operations


 Compilation flags: -O3 -pthread -DNUM_ITEMS=num_ops -DMAX_NUM_THREADS=num_threads 


 NUM_ITEMS is the total number of operations (mix of add, delete, search) to execute.

 MAX_NUM_THREADS is the maximum possible number of worker threads.

 KEYS (now a command line parameter)
  is the number of integer keys assumed in the range [10, 9+KEYS].
 The paper cited below states that the key range is [0, KEYS-1]. However, we have shifted the range by +10 so that
 the head sentinel key (the minimum key) can be chosen as zero. Any positive shift other than +10 would also work.

 Related work:

 Prabhakar Misra and Mainak Chaudhuri. Performance Evaluation of Concurrent Lock-free Data Structures
 on GPUs. In Proceedings of the 18th IEEE International Conference on Parallel and Distributed Systems,
 December 2012.

 Stivala et al 2010 Lock-free parallel dynamic programming
 J Parallel Distrib Comput 70:389-848


 modified by Alex Stivala to include values so have key/value not just key
 $Id: iit_oahttslf_test.cpp 4497 2012-12-31 05:58:43Z astivala $
***************************************************************************************/

#include"stdio.h"
#include"stdlib.h"
#include"time.h"
#include"pthread.h"
#include"assert.h"
#include"sys/time.h"

#include "oahttslf.h"

// Supported operations
#define ADD (0)
#define DELETE (1)
#define SEARCH (2)

uint64_t items[NUM_ITEMS];		// Array of keys associated with operations
uint64_t op[NUM_ITEMS];               // Array of operations
uint64_t result[NUM_ITEMS];		// Array of outcomes

int num_threads;


// The worker thread function
// The thread id is passed as an argument

void* Thread (void* t)
{  
  unsigned int tid=(unsigned long) t;
  int i;
  int foundcount = 0;
  int searchedcount = 0;

  // //START
  // // simple test in single thread: add and then search
  // uint64_t val = 0;
  // h.Add(123, 124, NUuint64_t);
  // bool foundit = h.Search(123, val);
  // assert(foundit);
  // assert(val == 124);
  // //END

  for (i=tid;i<NUM_ITEMS;i+=num_threads) {
     // Grab the operations and execute
     unsigned int item = items[i];
     uint64_t value = item + 1;
     uint64_t oldvalue;

     switch(op[i]){
       case ADD:
//         fprintf(stderr, "XXX ADD %lld, %lld\n", item, value);
         result[i]=10+oahttslf_insert(item, value, tid );
         break;
       case DELETE:
#ifdef ALLOW_DELETE
         oldvalue = oahttslf_delete(item, tid);
         if (!(oldvalue == OAHTTSLF_EMPTY_VALUE || oldvalue == item+1)) {
           fprintf(stderr, "tid = %ld item =%ld oldvalue = %ld\n", tid, item, oldvalue);
           assert(oldvalue == OAHTTSLF_EMPTY_VALUE || oldvalue == item+1);
         }
         result[i]=20+oldvalue; // don't know why the original does this int+bool
#endif
         break;
       case SEARCH:
//         fprintf(stderr, "XXX SERACH %lld\n", item);
         ++searchedcount;
         bool found = oahttslf_lookup(item, &value);
         result[i]=30+found; // don't know why the original does this int+bool
         if (found) {
           ++foundcount;
           assert(value == item+1);
         }
         break;
     }
  }
  fprintf(stderr, "thread %d found %d of %d searched-for items\n",
          tid, foundcount, searchedcount);
}

int main(int argc, char** argv)
{

  if (argc != 5) {
     printf("Need four arguments: nubmer of keys, percent add ops and percent delete ops and number of threads (e.g., 100000 30 10 8 for 100000 keys with 30%% add and 10%% delete with 8 threads).\nAborting...\n");
     exit(1);
  }

  // Extract operations ratio and number of threads
  int KEYS = atoi(argv[1]);
  int adds=atoi(argv[2]);
  int deletes=atoi(argv[3]);
  num_threads=atoi(argv[4]);

#ifndef ALLOW_DELETE
  if (deletes > 0) {
    fprintf(stderr, "compiled without ALLOW_DELETE: deletes not supported\n");
    exit(1);
  }
#endif

  if (adds+deletes > 100) {
     printf("Sum of add and delete precentages exceeds 100.\nAborting...\n");
     exit(1);
  }
	if (num_threads < 1 || num_threads > MAX_NUM_THREADS) {
		fprintf(stderr, "num threads must be between 1 and %d\n", MAX_NUM_THREADS);
		exit(1);
	}

  fprintf(stderr, "NUM_ITEMS = %d, KEYS = %d\n", NUM_ITEMS, KEYS);
  fprintf(stderr, "adds = %d, deletes = %d, num_threads = %d\n", adds, deletes,num_threads);

#ifdef USE_INSTRUMENT
  oahttslf_reset();
#endif

  // Initialize thread stack

  void* status;
  pthread_t threads[MAX_NUM_THREADS];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 102400);

  int rc;
  long t;
  int i, j;

  srand(0);

  // Populate key array
  // NUM_ITEMS is the total number of operations
  for(i=0;i<NUM_ITEMS;i++){
    items[i]=10+rand()%KEYS;		// KEYS is the number of integer keys
  }

  // Populate op array
  for(i=0;i<(NUM_ITEMS*adds)/100;i++){
    op[i]=ADD;
  }
  for(;i<(NUM_ITEMS*(adds+deletes))/100;i++){
    op[i]=DELETE;
  }
  for(;i<NUM_ITEMS;i++){
    op[i]=SEARCH;
  }
  
  struct timeval tv0,tv1;
  struct timezone tz0,tz1;

  // Spawn threads

  gettimeofday(&tv0,&tz0);
  for(t=0;t<num_threads;t++){
    rc = pthread_create(&threads[t], &attr, Thread, (void *)(t));
    if (rc){
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  // Join threads

  for(t=0; t<num_threads; t++) {
    rc = pthread_join(threads[t], &status);
  }
  gettimeofday(&tv1,&tz1);

//  PrintList();


  // Print time in ms

  printf("elapsed time %.0lf ms\n",((float)((tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec)))/1000.0);

#ifdef USE_INSTRUMENT
  printf("total collision count = %llu\n", oahttslf_total_collision_count());
  oahttslf_printstats();
  
#endif
#ifdef USE_CONTENTION_INSTRUMENT
  printf("total retry count = %lu\n", oahttslf_total_retry_count());
#endif

  return 0;
}
