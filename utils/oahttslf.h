#ifndef OAHTTSLF_H
#define OAHTTSLF_H
/*****************************************************************************
 * 
 * File:    oahttslf.h
 * Author:  Alex Stivala
 * Created: April 2009
 *
 * Declarations for open addressing thread-safe lock-free hash table.
 * 
 *
 * $Id: oahttslf.h 4620 2013-01-25 01:28:26Z astivala $
 *
 * Preprocessor symbols:
 *
 *
 * USE_GOOD_HASH  - use mixing hash function rather than trivial one
 * DEBUG          - include extra assertion checks etc.
 * ALLOW_UPDATE  - allow insert to update value of existing key
 * USE_INSTRUMENT - compile in (per-thread) instrumentation counts.
 * USE_CONTENTION_INSTRUMENT - per-thread contention counts (only).
 * ALLOW_DELETE  - allow keys to be removed
 *
 *****************************************************************************/

#include "bpautils.h"


#ifdef __cplusplus
extern "C" {
#endif


/* size of the hash table (must be a power of 2)*/
/*#define OAHTTSLF_SIZE  134217728 */   /* 2^27 */ /* too large */
#define OAHTTSLF_SIZE  67108864   /* 2^26 */
/*#define OAHTTSLF_SIZE  131072*/   /* 2^17 */

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


typedef unsigned long long uint64_t;
typedef unsigned int       uint32_t;
typedef unsigned short     uint16_t;
typedef unsigned char      uint8_t;


/* insert into hashtable. Returns old value. */
uint64_t oahttslf_insert(uint64_t key, uint64_t value, int thread_id);

/* lookup in hashtable */
bool oahttslf_lookup(uint64_t key, uint64_t *value);

/* test for invalid structure */
int oahttslf_validate(void);

/* compute and print stats about hash table */
void oahttslf_printstats(void);

/* reset all table entries to empty */
void oahttslf_reset(void);

/* return number of keys in table */
unsigned int oahttslf_num_entries();

#ifdef USE_INSTRUMENT
/* add up per-thread counters and return total */
unsigned int oahttslf_total_key_count();

/* add up per-thread collision counters and return total */
unsigned long long oahttslf_total_collision_count();
#endif

#ifdef USE_CONTENTION_INSTRUMENT
/* add up per-thread retry counters and return total */
unsigned int oahttslf_total_retry_count();
#endif

/* insert into hashtable. Returns old value. */
double oahttslf_insert_double(uint64_t key, double value, int thread_id);

/* lookup in hashtable */
bool oahttslf_lookup_double(uint64_t key, double *value);

#ifdef ALLOW_DELETE
/* delete key from hashtable */
uint64_t oahttslf_delete(uint64_t key, int thread_id);
#endif

#ifdef __cplusplus
}
#endif

#endif /* OAHTTSLF_H */
