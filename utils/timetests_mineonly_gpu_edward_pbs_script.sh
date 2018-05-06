#!/bin/bash

# pbs launching script:
 	
#PBS -N parallel_gpu_hashtables

#PBS -l walltime=23:00:00
#PBS -q gpu

# Reserve all 8 cores on a single gpu node (need SMP shared memory and gpu)
#PBS -l nodes=1:ppn=8

# run the test harness for the concurrent lock-free hash tables with 
# on a gpu node with 1 to 8 threads and also on the GPU
#
# $Id: timetests_mineonly_gpu_edward_pbs_script.sh 4618 2013-01-25 01:18:09Z astivala $

cd $PBS_O_WORKDIR
set CONV_RSH = ssh

module load cuda

echo -n "Start: "
date

MAX_THREADS=8

./timetests.sh "oahttslftest 1" "oahttslftest" $MAX_THREADS > cpu_oahttslf.edward.rtab

./timegputests.sh "oahttslf_gpu_test" > gpu_oahttslf.edward.rtab

#./timetests.sh "../LockFreeDS/cpu/hashtable/iitlockfreehttest 1" "../LockFreeDS/cpu/hashtable/iitlockfreehttest" $MAX_THREADS > cpu_lockfreehashtable.edward.rtab

#./timegputests.sh "./lockfreeht_gpu_test" > gpu_lockfreehashtable.edward.rtab


./timetests.sh "httslftest 1" "httslftest" $MAX_THREADS > cpu_httslf.edward.rtab

./timegputests.sh "httslf_gpu_test" > gpu_httslf.edward.rtab

times

echo -n "End: "
date

