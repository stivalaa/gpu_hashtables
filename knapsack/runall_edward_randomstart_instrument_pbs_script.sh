#!/bin/bash

# pbs launching script:
 	
#PBS -N knapsack_16core_randomstart_instrument

# use 'serial' queue on edward as it is SMP not MPI
#PBS -q serial

#PBS -l walltime=200:00:0

# Reserve 16 cores on a single node (need SMP shared memory)
#PBS -l nodes=1:ppn=16


cd $PBS_O_WORKDIR
set CONV_RSH = ssh


time ./runall_randomstart.sh '-y' edward_randomstart_instrument_output 16

times

