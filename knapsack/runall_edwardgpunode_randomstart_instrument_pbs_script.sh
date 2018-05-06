#!/bin/bash

# pbs launching script:
 	
#PBS -N knapsack_randomstart_instrument

# use gpu queue to put on a gpu node  (8 cores)
#PBS -q gpu

#PBS -l walltime=160:00:0

# Reserve 8 cores on a single node (need SMP shared memory)
#PBS -l nodes=1:ppn=8


cd $PBS_O_WORKDIR
set CONV_RSH = ssh


time ./runall_randomstart.sh '-y' edwardgpunode_randomstart_instrument_output 8

times

