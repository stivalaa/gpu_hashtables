#!/bin/bash

# pbs launching script:
 	
#PBS -N nonrecursive_knapsack

# use 'serial' queue on edward as it is SMP not MPI
#PBS -q serial

#PBS -l walltime=200:00:0

# Reserve 16 cores on a single node (need SMP shared memory)
#PBS -l nodes=1:ppn=16


cd $PBS_O_WORKDIR
set CONV_RSH = ssh


time ./runall_nonrecursive.sh '-y' edward_nonrecursive_output 16

times

