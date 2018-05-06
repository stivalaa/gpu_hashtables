#!/bin/bash

# pbs launching script:
 	
#PBS -N parallel_gpu_hashtables_iit

#PBS -l walltime=500:00:00
#PBS -q gpu

# Reserve all 8 cores on a single gpu node (need SMP shared memory and gpu)
#PBS -l nodes=1:ppn=8

# run the test harness for the concurrent lock-free hash tables with 
# different operation mixes similar to results in
#   Prabhakar Misra and Mainak Chaudhuri. Performance Evaluation of Concurrent 
#   Lock-free Data Structures on GPUs. In Proceedings of the 18th IEEE 
#   International Conference on Parallel and Distributed Systems, December 2012.
# $Id: timetests_iit_gpu_edward_pbs_script.sh 4588 2013-01-23 00:05:41Z astivala $

cd $PBS_O_WORKDIR
set CONV_RSH = ssh

module load cuda

echo -n "Start: "
date

MAX_THREADS=8

# number of keys to run with
NUMKEYS="1000 10000 100000 1000000"

# percentages of ADD_DELETE operations
# (the remainder after ADD,DELETE are SEARCH).
OPMIXES="30_0 10_10 20_20 40_40 50_10"


for numkeys in $NUMKEYS
do
  OUTDIR="edward_keys${numkeys}_output"
  if [ ! -d ${OUTDIR} ]; then
    mkdir ${OUTDIR}
  fi
  for opmix in $OPMIXES
  do
      adds=`echo $opmix | cut -d_ -f1`
      deletes=`echo $opmix | cut -d_ -f2`
    ./timetests.sh "iit_oahttslf_test $numkeys $adds $deletes 1" "iit_oahttslf_test $numkeys $adds $deletes" $MAX_THREADS > ${OUTDIR}/iit_cpu_oahttslf_${opmix}.edward.rtab
    ./timegputests.sh "iit_oahttslf_gpu_test $numkeys $adds $deletes" > ${OUTDIR}/iit_gpu_oahttslf_${opmix}.edward.rtab
    ./timetests.sh "iit_httslf_test $numkeys $adds $deletes 1" "iit_httslf_test $numkeys $adds $deletes" $MAX_THREADS > ${OUTDIR}/iit_cpu_httslf_${opmix}.edward.rtab
    ./timegputests.sh "iit_httslf_gpu_test $numkeys $adds $deletes" > ${OUTDIR}/iit_gpu_httslf_${opmix}.edward.rtab
    ./timetests.sh "../LockFreeDS/cpu/hashtable/LockFreeHashTableValues $numkeys $adds $deletes 1" "../LockFreeDS/cpu/hashtable/LockFreeHashTableValues $numkeys $adds $deletes" $MAX_THREADS > ${OUTDIR}/iit_cpu_lockfreehashtable_${opmix}.edward.rtab
    ./timegputests.sh "../LockFreeDS/gpu/hashtable/LockFreeHashTableValues $numkeys $adds $deletes" > ${OUTDIR}/iit_gpu_lockfreehashtable_${opmix}.edward.rtab
  done
done

times

echo -n "End: "
date

