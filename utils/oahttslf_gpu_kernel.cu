/*****************************************************************************
 * 
 * File:    oahttslf_gpu_kernel.cu
 * Author:  Alex Stivala
 * Created: November 2012
 *
 * Open addressing (closed hashing) thread-safe lock-free hash table.
 * Uses linear probing.
 * 
 * $Id: oahttslf_gpu_kernel.cu 4634 2013-01-31 03:17:42Z astivala $
 *
 *
 * Requires CUDA 4.x and device with compute capability 2.x and higher
 * as it uses atomic CAS on 64 bits and __device__ function recursion.
 * Also printf() from device function for debug.
 * It uses the CURAND library for pseudrandom number generation.
 *
 *
 * Each entry is a key-value pair. Writing is synchornized with CAS logic.
 * Note we depend on the empty key/value being 0.
 *
 *
 * Preprocessor symbols:
 *
 * NUM_THREADS    -  number of threads per block
 *
 * USE_INSTRUMENT - compile in instrumentation counts
 * USE_GOOD_HASH  - use mixing hash function rather than trivial one
 * DEBUG          - include extra assertion checks etc.
 * ALLOW_UPDATE  - allow insert to update value of existing key
 * ALLOW_DELETE  - allow keys to be removed
 *
 *****************************************************************************/

#include <stdio.h>
#include <string.h>
#include <assert.h>

#define ALLOW_UPDATE


#define CAS64 atomicCAS    /* CUDA compare-and-swap */

/*****************************************************************************
 *
 * constants
 *
 *****************************************************************************/


/* size of the hash table (must be a power of 2)*/
/*#define OAHTTSLF_SIZE  134217728 */   /* 2^27 */ /* too large */
#define OAHTTSLF_SIZE  67108864   /* 2^26 */
/*#define OAHTTSLF_SIZE  131072 */  /* 2^17 */

/* marks unused slot (a key cannot have this value) */
#define OAHTTSLF_EMPTY_KEY 0

#ifdef ALLOW_DELETE
/* marks deleted slot (a key cannot hvae this value). Note that this is
   different from OAHTTSLF_EMPTY_KEY to avoid the ABA problem. */
#define OAHTTSLF_DELETED_KEY 0xffffffffffffffff
#endif

/* marks unset value (a value cannot have this value) */
#define OAHTTSLF_EMPTY_VALUE 0

/* note we depend on the above two empty key/value being 0 since table
   is decalred static so initially zero */


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


/* linear probing step size */
#define OAHTTSLF_PROBE_STEP 1


/*****************************************************************************
 *
 * types
 *
 *****************************************************************************/


typedef struct oahttslf_entry_s
{
    uint64_t key;   /* TODO need to be able to have a SET (128 bit) key */
    uint64_t value;
} oahttslf_entry_t;


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
#endif

/*****************************************************************************
 *
 * device functions (callable from gpu only)
 *
 *****************************************************************************/


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

  i = q & (OAHTTSLF_SIZE - 1); /* depends on OAHTTSLF_SIZE being 2^n */

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
  }
  __syncthreads();
}

/*
 * oahttslf_sum_stats()
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
__device__ unsigned int global_retry_count = 0;
__device__ unsigned int global_collision_count = 0;
__device__ unsigned int global_new_insert_count = 0;
__device__ void oahttslf_sum_stats(void)
{
  unsigned int block_retry_count = 0;
  unsigned int block_collision_count = 0;
  unsigned int block_new_insert_count = 0;
  
  __syncthreads();
  
  if (threadIdx.x == 0)
  {
    for (int i = 0; i < NUM_THREADS; i++) 
    {
      block_retry_count += retry_count[i];
      block_collision_count += collision_count[i];
      block_new_insert_count += new_insert_count[i];
    }
    atomicAdd(&global_retry_count, block_retry_count);
    atomicAdd(&global_collision_count, block_collision_count);
    atomicAdd(&global_new_insert_count, block_new_insert_count);

  }
}
#endif /* USE_INSTRUMENT*/



/*
 * oahttslf_getent()
 *
 * Get the entry for a key from the hashtable
 *
 * Parameters:
 *     hasthable - the hashtable
 *     key -  key to look up
 *     vacant - (OUT) TRUE if reutrn pointer to entry for key
 *                    is not currently occupied by key
 *  Return value:
 *     pointer to entry with key, or for key (but currently empty) in hashtable
 *     or NULL if hashtable is full
 */
__device__ oahttslf_entry_t *oahttslf_getent(oahttslf_entry_t *hashtable,
                                             uint64_t key, bool *vacant)
{
  unsigned int h;
  oahttslf_entry_t *ent;
  int probes = 0;
  uint64_t entkey;

  h = hash_function(key);
  ent = &hashtable[h];
  entkey = ent->key;
  while (probes < OAHTTSLF_SIZE - 1 && entkey != key && entkey != OAHTTSLF_EMPTY_KEY)
  {
    ++probes;
    h = (h + OAHTTSLF_PROBE_STEP) & (OAHTTSLF_SIZE - 1); /*SIZE must be 2^n*/
    ent = &hashtable[h];
    entkey = ent->key;
  }
#ifdef USE_INSTRUMENT
  collision_count[threadIdx.x] += probes;
#endif
  if (probes >= OAHTTSLF_SIZE - 1)
    return NULL;
  else if (entkey == OAHTTSLF_EMPTY_KEY)
    *vacant = true;
  else
    *vacant = false;
  return ent;
}



/*
 * oahttslf_insert()
 *
 * Insert a key/value pair into the hashtable, or update the value
 * for existing key.
 *
 * Parameters:
 *    hashtable - the hash table
 *    key   - key to insert
 *    value - value to insert for the key
 *
 */
__device__ void oahttslf_insert(oahttslf_entry_t *hashtable,
                               uint64_t key, uint64_t value)
{
  oahttslf_entry_t *ent;
  uint64_t oldvalue;
#ifdef DEBUG
  uint64_t entkey;
#endif
  bool vacant;


  assert(key != OAHTTSLF_EMPTY_KEY);
#ifdef ALLOW_DELETE
  assert(key != OAHTTSLF_DELETED_KEY);
#endif
  assert(value != OAHTTSLF_EMPTY_VALUE);

  ent = oahttslf_getent(hashtable, key, &vacant);
  if (!ent)
  {
    printf("hash table full\n"); /* TODO expand table */
    return;
  }
  oldvalue = ent->value;
#ifdef DEBUG
  entkey = ent->key;
#endif
  if (vacant)
  {
    if (CAS64(&ent->key, (uint64_t)OAHTTSLF_EMPTY_KEY, key) != OAHTTSLF_EMPTY_KEY) {
#ifdef USE_INSTRUMENT
      retry_count[threadIdx.x]++;
#endif
      return oahttslf_insert(hashtable, key, value);  /* tail-recursive call to retry */
    }
#ifdef USE_INSTRUMENT
    else
      new_insert_count[threadIdx.x]++;
#endif
#ifdef DEBUG
    entkey =key;
#endif
  }
#ifdef DEBUG
  /*assert(key == ent->key);*/
  if (key != entkey)
  {
          printf( "OAHTTSLF ASSERTION FAILURE: key=%llX entkey=%llX\n",  key, entkey);
          return;
  }
#endif
#ifdef ALLOW_UPDATE
  if (oldvalue == value)  /* shortcut to avoid expense of CAS instruction */
    return;
  if (CAS64(&ent->value, oldvalue, value) != oldvalue) {
#ifdef USE_INSTRUMENT
    retry_count[threadIdx.x]++;
#endif
    oahttslf_insert(hashtable, key, value);  /* tail-recursive call to retry */
  }
#else
  ent->value = value;
#endif
  return;
}



/*
 * oahttslf_lookup()
 *
 * Get the value for a key from the hashtable
 *
 * Parameters:
 *     hashtable - the hash table
 *     key -  key to look up
 *     value - (output) value for key,ony set if TRUE returned.
 *
 * Return value:
 *     true if key is in hashtable else false
 */
__device__ bool oahttslf_lookup(oahttslf_entry_t *hashtable,
                                uint64_t key, uint64_t *value)
{
  oahttslf_entry_t *ent;
  bool vacant;
  uint64_t val;

  ent = oahttslf_getent(hashtable, key, &vacant);
  if (ent)
  {
    val = ent->value;
    if (!vacant && val != OAHTTSLF_EMPTY_VALUE)
    {
      *value = val;
      return true;
    }
  }
  return false;
}


#ifdef ALLOW_DELETE
/*
 * oahttslf_delete()
 *
 * Delete a key from the hashtable.
 *
 * Deletion in fact just change the key to OAHTTSLF_DELETED_KEY so the
 * key is no longer found. Note we do NOT change it to OAHTTSLF_EMPTY_KEY
 * (which would allow the slot to be reused) since that can introduce the
 * ABA problem with CAS. Ensuring a key can never go back to OAHTTSLF_EMPTY_KEY
 * once it has changed frmo that initial value avoids the ABA problem
 * entirely, but note this mean deleting a key does not actually allow
 * us to reclaim space.
 * TODO allow delete to reclaim space. Will require a double-word (ie 128 bit)
 * CAS to include a refcount to do this properly (not availabe on CUDA yet).
 *
 * Parameters:
 *    hashtable - the hash table
 *    key   - key to remove
 *
 * Return value:
 *    Value for the key prior to the delete (OAHTTSLF_EMPTY_VALUE
 *    if it wasn't there)
 */
__device__ uint64_t oahttslf_delete(oahttslf_entry_t *hashtable,
                         uint64_t key)
{
  oahttslf_entry_t *ent;
  uint64_t entkey, oldvalue;
  bool vacant;

  assert(key != OAHTTSLF_EMPTY_KEY);

  oldvalue = OAHTTSLF_EMPTY_VALUE;

  ent = oahttslf_getent(hashtable, key, &vacant);
  if (!ent)
  {
    /* hash table is full and we found no slot for this key, do nothing */
    return OAHTTSLF_EMPTY_VALUE;
  }
  entkey = ent->key;
  if (!vacant)
  {
    oldvalue = ent->value;
    if (CAS64(&ent->key, entkey, (uint64_t)OAHTTSLF_DELETED_KEY) != entkey) {
#ifdef USE_INSRUMENT
      retry_count[threadIdx.x]++;
#endif
      return oahttslf_delete(hashtable, key);  /* tail-recursive call to retry */
    }
  }
  ent->value = OAHTTSLF_EMPTY_VALUE;
  return oldvalue;
}
#endif /* ALLOW_DELETE */


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
__global__ void oahttslf_reset(oahttslf_entry_t *hashtable)
{
  // each thread does as many iterations as necessary to cover all slots
  // in the hash table
  const int tid = blockDim.x * blockIdx.x + threadIdx.x; // thead id
  for (int i = tid; i < OAHTTSLF_SIZE; i += gridDim.x * blockDim.x)
  {
    hashtable[i].key = OAHTTSLF_EMPTY_KEY;
    hashtable[i].value = OAHTTSLF_EMPTY_VALUE;
  }
}


#ifdef USE_INSTRUMENT
/*
 * oahttslf_print_stats()
 *
 * print stats from oahtllsf_sum_stats() to stdount using device printf
 * (just because it is easy than cudaMemcpy each variable back to host)
 *
 * Parameters: None
 * Retrun value: None
 */
__global__ void oahttslf_print_stats(void)
{
  const int tid = blockDim.x * blockIdx.x + threadIdx.x; // thead id
  if (tid  == 0) 
  {
    printf("HTINSTRUMENT collision=%u,retry=%u,insertion=%u\n",
           global_collision_count, global_retry_count, global_new_insert_count);
  }
}
#endif

