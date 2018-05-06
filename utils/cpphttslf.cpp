/*****************************************************************************
 * 
 * File:    cpphttslf.cpp
 * Author:  Alex Stivala
 * Created: April 2009
 *
 * Separate chaining thread-safe lock-free hash table.
 * C++ version (uses new instead of malloc), also no functino pointers
 * for hashfunction etc., hardcoded to just use 64 bit int keys like oahttslf
 * 
 * 
 * gcc version 4.1.0 or greater is required, in order to use the
 * __sync_val_compare_and_swap()
 * and __sync_val_fetch_and_add() builtins
 * (this module was developed on Linux 2.6.22 (x86) with gcc 4.1.3),
 * except on Solaris, where we use SUNWSpro compiler and
 * cas32()/caslong()/cas64()/casptr() in 
 * /usr/include/sys/atomic.h)
 *
 * $Id: cpphttslf.cpp 4489 2012-12-20 23:53:57Z astivala $
 *
 * Preprocessor symbols:
 *
 * USE_GOOD_HASH  - use mixing hash function rather than trivial one
 * USE_THREAD_CP_ALLOC  - use per-thread cellpool allocator
 * 
 *****************************************************************************/


#define USE_GOOD_HASH



#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cpphttslf.h"
#include "cellpool.h"
#include "atomicdefs.h"
#ifdef USE_THREAD_CP_ALLOC
#include "threadcellpool.h"
#endif

/*****************************************************************************
 *
 * static data
 *
 *****************************************************************************/


/* The hash table */
/* Each entry is head of chain pointer and
   is serialized with CAS logic in httslf_insert() */
/* TODO: change to allocate dynamically so can have multiple */
static httslf_entry_t *hashtable[HTTSLF_SIZE];

 /* counters for hash collisions: serialize with __sync_fetch_and_add() */
static unsigned int insert_collision_count = 0; /* TODO implement this */
static unsigned int lookup_collision_count = 0; /* TODO implement this */



/*****************************************************************************
 *
 * static functions
 *
 *****************************************************************************/


#ifdef USE_GOOD_HASH
/*
  hash a 64 bit value into 32 bits. From:
  (Thomas Wang, Jan 1997, Last update Mar 2007, Version 3.1)
  http://www.concentric.net/~Ttwang/tech/inthash.htm
  (found by reference in NIST Dictionary of Algorithms and Data Structures)
*/
static unsigned long hash6432shift(unsigned long long key)
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


static unsigned int hash_function(uint64_t key) {
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


/*****************************************************************************
 *
 * external functions
 *
 *****************************************************************************/

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
 *    key   - ptr to key to insert
 *    value - value to insert for the key
 *    thread_id - id (0..n not pthread_t) for this thread
 *
 * Return value:
 *    Pointer to entry inserted.
 */
httslf_entry_t *httslf_insert(uint64_t key, uint64_t value, int thread_id)
{
  static const char *funcname = "httslf_insert";
  unsigned int h;
  httslf_entry_t *ent, *prev, *oldent, *newent = NULL;
  httslf_entry_t *inserted_entry = NULL;
  bool retry;

  h = hash_function(key);
  do
  {
    ent = hashtable[h];
    oldent = ent; /* old value for CAS logic */
    /* look for existing entry with this key, unlink any nodes marked deleted
       on the way */
    prev = NULL;
    retry = false;
    while (ent && key != ent->key)
    {
#ifdef ALLOW_DELETE
      if (ent->deleted) 
      {
        if (prev && CASPTR(&prev->next, ent, ent->next) != ent)
        {
          retry = true; /* restart from head of chain if CAS fails */
          break;
        }
        else if (!prev && CASPTR(&hashtable[h], ent, ent->next) != ent)
        {
          retry =true;
          break;
        }
      }
#endif/*ALLOW_DELETE*/
      prev = ent;
      ent = ent->next;
    }
    if (retry)
      continue;
    if (!ent)
    {
      if (!newent)
      {
#ifdef USE_THREAD_CP_ALLOC
        newent = (httslf_entry_t *)perthread_cellpool_alloc(thread_id);
        newent->key = key;
        newent->value = value;
#ifdef ALLOW_DELETE
        newent->deleted = false;
#endif
#else
        newent = new httslf_entry_t(key, value);
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
  }
  while (CASPTR(&hashtable[h], oldent, newent) != oldent);
  
  return inserted_entry;
}



/*
 * httslf_lookup()
 *
 * Get the value for a key from the hashtable
 *
 * Parameters:
 *     key - ptr to key to look up
 *     value - (out) value (if found)
 * 
 * Return value:
 *      true if value found 
 */
bool httslf_lookup(uint64_t key, uint64_t *value)
{
  unsigned int h;
  httslf_entry_t *ent;
  h = hash_function(key);
  ent = hashtable[h];
  while (ent && key != ent->key)
    ent = ent->next;
#ifdef ALLOW_DELETE
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







/*
 * httslf_validate()
 *
 * Test for duplicate keys in the lists -this should not happen
 *
 * Parameters:
 *    None
 *
 * Return value:
 *    0 if duplicate keys found else 1
 */
int httslf_validate(void)
{
  httslf_entry_t *ent1, *ent2;
  int i;

  for (i = 0; i < HTTSLF_SIZE; i++)
      for (ent1 = hashtable[i]; ent1 != NULL; ent1 = ent1->next)
          for (ent2 = ent1->next; ent2 != NULL; ent2 = ent2->next)
              if (ent1->key == ent2->key)
              {
                  return 0;
              }
  return 1;
}

/* TODO function to free hash table entries */


/*
 * httslf_printstats()
 *
 *   Compute and print statistics about the hash table to stdout
 *
 *   Parameters: None
 *   Return value: None
 */
void httslf_printstats(void)
{
  unsigned int num_items=0, num_entries=0;
  unsigned int chain_length,max_chain_length=0,sum_chain_length=0;
  float avg_chain_length;
  int i;
  httslf_entry_t *ent;

  for (i = 0; i < HTTSLF_SIZE; i++)
  {
    chain_length = 0;
    if ((ent = hashtable[i]) != NULL)
      num_entries++;
    while (ent)
    {
      num_items++;
      chain_length++;
      ent = ent->next;
    }
    sum_chain_length += chain_length;
    if (chain_length > max_chain_length)
      max_chain_length = chain_length;
  }
  avg_chain_length = (float)sum_chain_length / num_entries;
  printf("num slots used  : %u\n", num_entries);
  printf("num items       : %u (%f%% full)\n", num_items,
         100.0*(float)num_items/HTTSLF_SIZE);
  printf("max chain length: %u\n", max_chain_length);
  printf("avg chain length: %f\n", avg_chain_length);
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
 *    key   -  key to delete
 *
 * Return value:
 *    None.
 */
void httslf_delete(uint64_t key)
{
  static const char *funcname = "httslf_delete";
  unsigned int h;
  httslf_entry_t *ent, *prev;

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
      if (prev && CASPTR(&prev->next, ent, ent->next) != ent)
      {
        goto retry;    /* restart from head of chain if CAS fails */
      }
      else if (!prev && CASPTR(&hashtable[h], ent, ent->next) != ent)
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


void httslf_initialize(int num_threads)
{
#if defined(USE_THREAD_CP_ALLOC)
  size_t cell_size = sizeof(httslf_entry_t);
  /* need to pass number of cells as a parameter in this function */
  if (!(perthread_cellpool_initialize(cell_size, (size_t)10000000, num_threads)))
  {
    fprintf(stderr, "cellpool_initialize() failed\n");
    exit(1);
  }
#endif
}
