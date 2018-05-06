#ifndef CPPHTTSLF_H
#define CPPHTTSLF_H
/*****************************************************************************
 * 
 * File:    cpphttslf.h
 * Author:  Alex Stivala
 * Created: April 2009
 *
 * Declarations for separate chaining thread-safe lock-free hash table
 * C++ version (uses new instead of malloc), also no functino pointers
 * for hashfunction etc., hardcoded to just use 64 bit int keys like oahttslf
 * 
 *
 * $Id: cpphttslf.h 4556 2013-01-13 02:20:29Z astivala $
 *
 *****************************************************************************/


typedef unsigned long long uint64_t;
typedef unsigned int       uint32_t;
typedef unsigned short     uint16_t;
typedef unsigned char      uint8_t;


/* size of the hash table (must be a power of 2)*/
/*#define HTTSLF_SIZE  134217728 */   /* 2^27 */ /* too large */
/*#define HTTSLF_SIZE  67108864  */  /* 2^26 */
#define HTTSLF_SIZE  131072   /* 2^17 */

class httslf_entry_t
{
  public:
    uint64_t        key;
    uint64_t        value;
#ifdef ALLOW_DELETE
    bool            deleted;
#endif /* ALLOW_DELETE */
    httslf_entry_t *next;

    httslf_entry_t(uint64_t k, uint64_t v) : key(k), value(v)
#ifdef ALLOW_DLEETE
    , deleted(false) 
#endif/* ALLOW_DELETE */
    {}
};



/* insert into hashtable */
httslf_entry_t *httslf_insert(uint64_t key, uint64_t value, int thread_id);

/* lookup in hashtable */
bool httslf_lookup(uint64_t key, uint64_t *value);

/* test for invalid structure */
int httslf_validate(void);

/* compute and print stats about hash table */
void httslf_printstats(void);

#ifdef ALLOW_DELETE
/* dete a key frmo the table */
void httslf_delete(uint64_t key) ;
#endif /* ALLOW_DELETE */

void httslf_initialize(int num_threads);

#endif /* CPPHTTSLF_H */
