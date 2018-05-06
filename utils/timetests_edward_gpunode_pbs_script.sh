#!/bin/bash

# pbs launching script:
 	
#PBS -N parallel_hashtables

#PBS -l walltime=100:00:00
#PBS -q gpu

# Reserve all 8 cores on a single gpu node (need SMP shared memory and gpu)
#PBS -l nodes=1:ppn=8

cd $PBS_O_WORKDIR
set CONV_RSH = ssh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alexs/phd/paralleldp/streamflow
#export TBB21_INSTALL_DIR=/home/alexs/tbb21_20080605oss
#. $TBB21_INSTALL_DIR/em64t/cc4.1.0_libc2.4_kernel2.6.16.21/bin/tbbvars.sh

./timetests.sh httest ../LockFreeDS/cpu/hashtable/iitlockfreehttest 8 > iitlockfreehttest.edwardgpunode.rtab

./timetests.sh httest oahttslftest 8 > oahttslftest.edwardgpunode.rtab
./timetests.sh httest httslftest 8 > edwardgpunode.rtab

#./timetests.sh httest ../nbds.0.4.3/output/nbdstest 8 > nbdstest.edwardgpunode.rtab


