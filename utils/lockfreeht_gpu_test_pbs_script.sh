#!/bin/bash

# pbs launching script:
 	
#PBS -N gpu_lockfreeht

#PBS -l walltime=23:00:00
#PBS -q gpu

# Reserve all 8 cores on a single gpu node (need SMP shared memory and gpu)
#PBS -l nodes=1:ppn=8


cd $PBS_O_WORKDIR
set CONV_RSH = ssh

module load cuda

echo -n "Start: "
date

time ./lockfreeht_gpu_test

echo -n "End: "
date

