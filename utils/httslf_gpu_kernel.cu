/*****************************************************************************
 * 
 * File:    httslf_gpu_kernel.cu
 * Author:  Alex Stivala
 * Created: December 2012
 *
 * Separate chaining thread-safe lock-free hash table.
 * CUDA version , using cell pool on GPU, no memory allocation,
 * no functino pointers
 * for hashfunction etc., hardcoded to just use 64 bit int keys like oahttslf
 * 
 * 
 * Requires CUDA 4.x and device with compute capability 2.x and higher
 * as it uses atomic CAS on 64 bits.
 * Also printf() from device function for debug.
 * It uses the CURAND library for pseudrandom number generation.
 *
 *
 * $Id: httslf_gpu_kernel.cu 4634 2013-01-31 03:17:42Z astivala $
 *
 * Preprocessor symbols:
 *
 * USE_GOOD_HASH  - use mixing hash function rather than trivial one
 * ALLOW_DELETE  - allow keys to be removed
 * USE_INSTRUMENT - compile in instrumentation counts
 * 
 *****************************************************************************/

#include <stdio.h>
#include <string.h>



#define USE_GOOD_HASH



/*****************************************************************************
 *
 * constants
 *
 *****************************************************************************/

/* size of the hash table (must be a power of 2)*/
/*#define HTTSLF_SIZE  134217728 */   /* 2^27 */ /* too large */
#define HTTSLF_SIZE  67108864   /* 2^26 */
/*#define HTTSLF_SIZE  131072 */   /* 2^17 */

/*****************************************************************************
 *
 * types
 *
 *****************************************************************************/



#ifndef _STDINT_H
typedef unsigned long long uint64_t;
typedef unsigned int       uint32_t;
typedef unsigned short     uint16_t;
typedef unsigned char      uint8_t;
#else
#define uint64_t unsigned long long
#define uint32_t unsigned int
#define uint16_t unsigned short
#define uint8_t  unsigned char
#endif




class httslf_entry_t
{
  public:
    uint64_t        key;
    uint64_t        value;
#ifdef ALLOW_DELETE
    bool            deleted;
#endif /* ALLOW_DELETE */
    volatile httslf_entry_t *next;

    httslf_entry_t(uint64_t k, uint64_t v) : key(k), value(v)
#ifdef ALLOW_DLEETE
    , deleted(false) 
#endif/* ALLOW_DELETE */
    {}
};


/*****************************************************************************
 *
 * device constant data (set on host from main code)
 *
 *****************************************************************************/

__device__ __constant__ unsigned int total_num_cells; /* total number of cells in pool */


/*****************************************************************************
 *
 * shared memory
 *
 *****************************************************************************/
// note shared memory cannot have initializeers - need to call the device
// fucntion reset_counters() to initialize
#ifdef USE_INSTRUMENT
__shared__ unsigned int collision_count[NUM_THREADS]; // number of collisions (per thread)
__shared__ unsigned int retry_count[NUM_THREADS]; // number of CAS failures (per thread)
__shared__ unsigned int new_insert_count[NUM_THREADS]; // number of new items successfuly inserted (per thread)
__shared__ unsigned int calls_to_insert_count[NUM_THREADS]; // number of calls to insert functon (per thread counter)
#endif

/*****************************************************************************
 *
 * device global data (allocatd on host from main code)
 *
 *****************************************************************************/

__device__ httslf_entry_t *cellpool; /* allocated cell pool memory */
__device__ unsigned int nextcell_index = 1; /* index of next cell in pool */

/*****************************************************************************
 *
 * device functions (callabel from gpu only)
 *
 *****************************************************************************/

/*
 * gpu_cellpool_alloc()
 *
 *   allocate a new cell from pool, application specific version for
 *   httslf_entry_t cells, uses array indexing so index is just 0,1,2,...n
 *   so incremetned with atomicInc not atomicAdd of runtime cell size
 *
 *   Parameters:
 *        None.
 *
 *   Return value:
 *      Pointer to new cell, or NULL if none available.
 *
 *   Uses static data:
 *      nextcellindex -index of next cell in cellpool (read/write, atomicAdd serialized)
 */
__device__ httslf_entry_t *gpu_cellpool_alloc()
{
  unsigned int cellindex;
  

  cellindex = atomicInc(&nextcell_index, total_num_cells);
  assert(cellindex < total_num_cells);
  assert(cellindex > 0);
  return &cellpool[cellindex];
}


#ifdef USE_GOOD_HASH
/*
  hash a 64 bit value into 32 bits. From:
  (Thomas Wang, Jan 1997, Last update Mar 2007, Version 3.1)
  http://www.concentric.net/~Ttwang/tech/inthash.htm
  (found by reference in NIST Dictionary of Algorithms and Data Structures)
*/
__device__ unsigned long hash6432shift(unsigned long long key)
{
  key = (~key) + (key << 18); /* key = (key << 18) - key - 1; */
  key = key ^ (key >> 31);
  key = key * 21; /* key = (key + (key << 2)) + (key << 4); */
  key = key ^ (key >> 11);
  key = key + (key << 6);
  key = key ^ (key >> 22);
  return (unsigned long) key;
}

#endif


__device__ unsigned int hash_function(uint64_t key) {
  unsigned int i;
  unsigned long q;

#ifdef USE_GOOD_HASH
  q = hash6432shift(key);
#else
  q = key;
#endif

  i = q & (HTTSLF_SIZE - 1); /* depends on OAHTTSLF_SIZE being 2^n */

  return i;
}


#ifdef USE_INSTRUMENT
/*
 * reset__counters()
 *
 * Initialize the per-thread intrumentatino counters in shared memory
 *
 * Parameters: None
 * Return value: None
 */
__device__ void reset_counters(void)
{
  for (int i = threadIdx.x; i < NUM_THREADS; i += blockDim.x)
  {
    collision_count[i] = 0;
    retry_count[i] = 0;
    new_insert_count[i] = 0;
    calls_to_insert_count[i] = 0;
  }
  __syncthreads();
}
#endif


/*
 * httslf_insert()
 *
 * Insert a key/value pair into the hashtable
 * NB This only allows insertion of a NEW key - if the key already
 * exists, we do nothing (and do NOT update the value if it is different).
 * This is for use in dynamic programming with multiple threads
 * simple case (no bounding) where a particular key once its value is set
 * is the optimal - any other thread can only ever compute the same value
 * anyway. The case where we have bounds and values can change is more
 * difficult, and not handled by this function.
 *
 * Parameters:
 *    hashtable - the hash table
 *    key   - ptr to key to insert
 *    value - value to insert for the key
 *
 * Return value:
 *    Pointer to entry inserted.
 */
__device__ volatile httslf_entry_t *httslf_insert(httslf_entry_t * *hashtable,
                                                  uint64_t key, uint64_t value)
{
  unsigned int h;
  volatile httslf_entry_t *ent, *prev, *oldent, *newent = NULL;
  volatile httslf_entry_t *inserted_entry = NULL;
  bool retry;
  int loopcount = 0;

#ifdef USE_INSTRUMENT
  calls_to_insert_count[threadIdx.x]++;
#endif

  h = hash_function(key);
  do
  {
    ent = hashtable[h];

    oldent = ent; /* old value for CAS logic */
    /* look for existing entry with this key, unlink any nodes marked deleted
       on the way */
    prev = NULL;
    retry = false;
//    printf("XXX h = %08X ent = %p\n", h, ent); 
    assert(h < HTTSLF_SIZE);
#ifdef ALLOW_DELETE
    assert(((uint64_t)ent & 0xf) == 0);
#endif
    while (ent && key != ent->key)
    {
#ifdef ALLOW_DELETE
      if (ent->deleted) 
      {
        if (prev && atomicCAS((unsigned long long *)&prev->next, (unsigned long long)ent, (unsigned long long)ent->next) != (unsigned long long)ent)
        {
          retry = true; /* restart from head of chain if CAS fails */
          break;
        }
        else if (!prev && atomicCAS((unsigned long long *)&hashtable[h], (unsigned long long)ent, (unsigned long long)ent->next) != (unsigned long long)ent)
        {
          retry =true;
          break;
        }
      }
#endif/*ALLOW_DELETE*/
      prev = ent;
      ent = ent->next;
//      if (((uint64_t)ent & 0xf) != 0) {
//        printf("assertion failued, ent & 0xf != 0: ent = %p, prev = %p, h = %x\n",  ent, prev, h); //XXX
//        printf("  hashtable[h] = %p\n", hashtable[h]);//XXX
//        printf("  prev->key = %lx prev->value = %lx prev->next = %lx\n", prev->key, prev->value, prev->next);//XXX
//        printf("  oldent = %p, newent = %p, loopcount = %d\n", oldent, newent, loopcount);//XXX
//      }
#ifdef ALLOW_DELETE
      assert(((uint64_t)ent & 0xf) == 0);
#endif
    }
    if (retry) 
    {
#ifdef USE_INSTRUMENT
      retry_count[threadIdx.x]++;
#endif
      continue;
    }
    if (!ent)
    {
      if (!newent)
      {
#ifdef USE_INSTRUMENT
        if (oldent)
          collision_count[threadIdx.x]++;
        new_insert_count[threadIdx.x]++;
#endif
       newent = gpu_cellpool_alloc();
//device malloc is very slow      newent = (httslf_entry_t *)malloc(sizeof(httslf_entry_t));
        assert(newent!=NULL);
        newent->key = key;
        newent->value = value;
#ifdef ALLOW_DELETE
        newent->deleted = false;
#endif
        /* insert at head of list using CAS instruction - if we lose
           the race (another thread is inserting this key also) then
           we re-try (in do while loop).
        */
      }
      newent->next = oldent;
      inserted_entry = newent;
    }
    else
    {
      /* key already exists, just ignore the new one
         NB we do NOT update the value here, see header comment */
      inserted_entry = ent;
      break;
    }
    loopcount++;
   __threadfence();
  }
  while (atomicCAS((unsigned long long *)&hashtable[h], (unsigned long long)oldent, (unsigned long long)newent) != (unsigned long long)oldent);
   
  
//  if (loopcount > 1) printf("XXX loopcount = %d\n",loopcount); //XXX
#ifdef USE_INSTRUMENT
  if (loopcount > 1)
    retry_count[threadIdx.x] += loopcount-1;
#endif

  return inserted_entry;
}



/*
 * httslf_lookup()
 *
 * Get the value for a key from the hashtable
 *
 * Parameters:
 *     hashtable - the hash table
 *     key - ptr to key to look up
 *     value - (out) value (if found)
 * 
 * Return value:
 *      true if value found 
 */
__device__ bool httslf_lookup(httslf_entry_t **hashtable,
                              uint64_t key, uint64_t *value)
{
  unsigned int h;
  volatile httslf_entry_t *ent;
  h = hash_function(key);
  ent = hashtable[h];
  while (ent && key != ent->key)
    ent = ent->next;
//      if (((uint64_t)ent & 0xf) != 0) {
//        printf("lookup assertion failed, ent & 0xf != 0: ent = %p\n",  ent); //XXX
//      }
#ifdef ALLOW_DELETE
      assert(((uint64_t)ent & 0xf) == 0);
  if (ent && !ent->deleted) 
#else
  if (ent )
#endif /*ALLOW_DELETE*/
  {
    *value = ent->value;
    return true;
  }
  else
  {
    return false;
  }
}






#ifdef ALLOW_DELETE
/*
 * httslf_delete()
 *
 * Delete a key/value pair fomr the hashtable
 *
 * In fact it just marks the entry as deleted, it is unlinked from the chain
 * by subsequent insert or delete operations.
 *
 * Note it also does not actual free the node to reclaim space to avoid ABA
 * problem entirely.
 * TODO allow delete to reclaim space. Will require a double-word (ie 128 bit)
 * CAS to include a refcount to do this properly (not availabe on CUDA yet).
 *
 * Parameters:
 *    hashtable - the hash table
 *    key       -  key to delete
 *
 * Return value:
 *    None.
 */
__device__ void httslf_delete(httslf_entry_t **hashtable, uint64_t key)
{
  unsigned int h;
  volatile httslf_entry_t *ent, *prev;

  h = hash_function(key);

retry:
  ent = hashtable[h];
  /* look for existing entry with this key, unlink any nodes marked deleted
      on the way */
  prev = NULL;
  while (ent && key != ent->key)
  {
#ifdef ALLOW_DELETE
    if (ent->deleted) 
    {
      if (prev && atomicCAS((unsigned long long *)&prev->next, (unsigned long long)ent, (unsigned long long)ent->next) != (unsigned long long)ent)
      {
        goto retry;    /* restart from head of chain if CAS fails */
      }
      else if (!prev && atomicCAS((unsigned long long *)&hashtable[h], (unsigned long long)ent, (unsigned long long)ent->next) != (unsigned long long)ent)
      {
        goto retry;
      }
    }
#endif /*ALLOW_DELETE*/
    prev = ent;
    ent = ent->next;
  }
  if (ent)
  {
    ent->deleted = true;
  }

}

#endif /* ALLOW_DELETE */

#ifdef USE_INSTRUMENT
/*
 * httslf_sumcounters()
 *
 * Sum up the per-thread counters
 *   Also uses atomicInc and atmoicAdd and atmoicCAS
 *  on global memory so need compute 2.x anyway
 *
 *   Parameters:  None
 *   Return value: None
 */
__device__ unsigned int global_retry_count = 0;
__device__ unsigned int global_collision_count = 0;
__device__ unsigned int global_new_insert_count = 0;
__device__ unsigned int global_calls_to_insert_count = 0;
__device__ void httslf_sumcounters(void)
{
  const int tid = blockDim.x * blockIdx.x + threadIdx.x; // thread id

  unsigned int block_retry_count = 0;
  unsigned int block_collision_count = 0;
  unsigned int block_new_insert_count = 0;
  unsigned int block_calls_to_insert_count = 0;
  
  __syncthreads();

  if (threadIdx.x == 0)
  {
    for (int i = 0; i < NUM_THREADS; i++) 
    {
      block_retry_count += retry_count[i];
      block_collision_count += collision_count[i];
      block_new_insert_count += new_insert_count[i];
      block_calls_to_insert_count += calls_to_insert_count[i];
    }
    atomicAdd(&global_retry_count, block_retry_count);
    atomicAdd(&global_collision_count, block_collision_count);
    atomicAdd(&global_new_insert_count, block_new_insert_count);
    atomicAdd(&global_calls_to_insert_count, block_calls_to_insert_count);
  }
}
#endif /* USE_INSTRUMENT */

/*****************************************************************************
 *
 * global functions (callable from host only)
 *
 *****************************************************************************/


/*
 * oahttslf_reset()
 * 
 * reset all the table entries to empty
 *
 * Parameters: 
 *       hashtable - the hash table
 *
 * Return value: None
 *
 */
__global__ void httslf_reset(httslf_entry_t **hashtable)
{
  // each thread does as many iterations as necessary to cover all slots
  // in the hash table
  const int tid = blockDim.x * blockIdx.x + threadIdx.x; // thead id
  for (int i = tid; i < HTTSLF_SIZE; i += gridDim.x * blockDim.x)
  {
    hashtable[i] = NULL;
  }
}


/*
 * httslf_computestats()
 *
 *   Compute statistics about the hash table 
 *   Also uses atomicInc and atmoicAdd and atmoicCAS
 *  on global memory so need compute 2.x anyway
 *
 *   Parameters: 
 *        hashtable - the hash table
 *   Return value: None
 */
__device__ unsigned int num_items=0, num_entries=0,sum_chain_length=0,max_chain_length=0;
__global__ void httslf_computestats(httslf_entry_t **hashtable)
{

  unsigned int chain_length;
  volatile httslf_entry_t *ent;
  const int tid = blockDim.x * blockIdx.x + threadIdx.x; // thread id

  /* each thread does as many entires as necessary to count whole
     table, using atomic incremetn of global counters. would no doubt
     be more efficent to have thread block counters in shared memory
     then do reducction operation to get total but this is just
     debugging/instrumentatinon not used in acutal data structre usage,
     no need to make it more cmoplicated for efficiency */

  for (int i = tid; i < HTTSLF_SIZE; i += gridDim.x * blockDim.x)
  {
    chain_length = 0;
    if ((ent = hashtable[i]) != NULL)
      atomicInc(&num_entries, UINT_MAX);
    while (ent)
    {
      atomicInc(&num_items, UINT_MAX);
      chain_length++;
      ent = ent->next;
    }
    atomicAdd(&sum_chain_length, chain_length);
    unsigned int old_max_chain_length;
    do 
    {
      old_max_chain_length = max_chain_length;
      if (chain_length <= old_max_chain_length)
        break;
    }
    while (atomicCAS(&max_chain_length, old_max_chain_length, chain_length)
           != old_max_chain_length);
  }
}

#ifdef USE_INSTRUMENT
/*
 * httslf_printstats()
 *
 *  print using device prinf the global sstats computers computed from
 *  httslf_computestats() and httslf_sumcounters()
 *  NEeds capability 2.x to do device printf, but really just doing it this
 * way because easier than cudaMemcpy the stats fields back to host.
 *   Parameters:  None
 *   Return value: None
 */
__global__ void httslf_printstats(void)
{

  float avg_chain_length;
  const int tid = blockDim.x * blockIdx.x + threadIdx.x; // thread id

  if (tid == 0) {
    avg_chain_length = (float)sum_chain_length / num_entries;

    /* printf("total collision count = %u\n", global_collision_count); */
    /* printf("total retry count = %u\n", global_retry_count); */
    /* printf("total insertino count = %u\n", global_new_insert_count); */
    /* printf("total calls to insert function= %u\n", global_calls_to_insert_count); */
    /* printf("num slots used  : %u\n", num_entries); */
    /* printf("num items       : %u (%f%% full)\n", num_items, */
    /*       100.0*(float)num_items/HTTSLF_SIZE); */
    /* printf("max chain length: %u\n", max_chain_length); */
    /* printf("avg chain length: %f\n", avg_chain_length); */
    
    printf("HTINSTRUMENT collision=%u,retry=%u,insertion=%u,insertcalls=%u,slots=%u,items=%u,full=%f%%,maxchainlen=%u,avgchainlen=%f\n", 
           global_collision_count, global_retry_count, global_new_insert_count,
           global_calls_to_insert_count, num_entries, num_items, 
           100.0*(float)num_items/HTTSLF_SIZE,
           max_chain_length, avg_chain_length);
  }
}
#endif /* USE_INSTRUMENT */

