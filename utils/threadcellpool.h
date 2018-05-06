#ifndef THREADCELLPOOL_H
#define THREADCELLPOOL_H
/*****************************************************************************
 * 
 * File:    threadcellpool.h
 * Author:  Alex Stivala
 * Created: December 2012
 *
 * Declarations for simple per-thread cell pool allocator.
 * 
 *
 * $Id: threadcellpool.h 4469 2012-12-17 23:50:31Z astivala $
 *
 *****************************************************************************/


#ifdef __cplusplus
extern "C" {
#endif

/* allocate a new cell of previously intialized size from pool */
void *perthread_cellpool_alloc(int thread_id);

/* initialize the cell pool */
void **perthread_cellpool_initialize(size_t cell_size, int num_cells, int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* THREADCELLPOOL_H */
