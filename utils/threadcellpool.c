/*****************************************************************************
 * 
 * File:    threadcellpool.c
 * Author:  Alex Stivala
 * Created: December 2012
 *
 * Simple per-therad cell pool allocator. By allocating a big chunk of memory at
 * initialization time, divided up into one block for each thread,
 * we can just get one cell at a time from it without any locking or
 * synchronization at all since each thread has its own cell pool.
 *
 *
 * TODO don't even have  facility to free/reuse cells, don't need it
 * since we just terminate the process.
 * TODO only one cell pool allowed (file static data). Should change
 * to have handles so user can allocate different pools.
 * 
 *
 * $Id: threadcellpool.c 4443 2012-12-10 06:11:18Z astivala $
 *
 *****************************************************************************/


#include <stdlib.h>
#include "threadcellpool.h"

static void  **cellpool; /* The cell pool, one per thread. */
static size_t cellsize; /* size of a cell */
static size_t poolsize; /* total size of cell pool (for each thread) */
static void  **nextcell; /* pointer to next cell in the pool, for each thread*/


/*
 * perthread_cellpool_alloc()
 *
 *   allocate a new cell of previously intialized size from pool for this thread
 *
 *   Parameters:
 *     thead_id (0..num_threads, NOT pthread_t) for this thread
 *
 *   Return value:
 *      Pointer to new cell, or NULL if none available.
 *
 *   Uses static data:
 *      nextcell[thread_id] -pointer to next cell in cellpool (read/write)
 *      cellpool[thread_id] -pointer to cellpool for this thread (read)
 *      poolsize - size of cellpool (for each thread)
 */
void *perthread_cellpool_alloc(int thread_id)
{
  void *cell, *newnextcell;

  if ((char *)(nextcell[thread_id]) >= (char *)(cellpool[thread_id]) + poolsize)
    return NULL;
  cell = nextcell[thread_id];
  newnextcell = (char *)cell + cellsize;
  nextcell[thread_id] = newnextcell;
  return cell;
}

/*
 * perthread_cellpool_initialize()
 *
 *   initialize the cell pool
 *
 *   Parameters:
 *     cell_size - Size of each cell
 *     num_cells - Number of cells to allocate for each thread
 *     num_threads - Number of threads
 *
 *   Return value:
 *     Pointer to start of cell pool memory (first cell) for thread id 0 or NULL
 *     on failure
 *
 *    Uses static data:
 *       cellpool - the cell pool (write)          
 *       cellsize - size of each cell (write)
 *       poolsize - total size of cell pool (write)
 *       nextcell - pointer to next cell in cellpool (write)
 */
void **perthread_cellpool_initialize(size_t cell_size, int num_cells, int num_threads)
{
  int i;

  cellsize = cell_size;
  poolsize = num_cells * cell_size;
  if (!(cellpool = malloc(sizeof(void *) * num_threads)))
    return NULL;
  if (!(nextcell = malloc(sizeof(void *) * num_threads)))
    return NULL;
  for (i = 0; i < num_threads; i++)
  {
    if (posix_memalign(&cellpool[i], 16, poolsize) != 0)
      return NULL;
    nextcell[i] = cellpool[i];
  }
  return cellpool;
}

