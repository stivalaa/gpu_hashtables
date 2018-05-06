#!/bin/bash

# pbs launching script:
 	
#PBS -N knapsack_simple

#PBS -q gpu
#PBS -l walltime=8:00:0
#PBS -l nodes=1


cd $PBS_O_WORKDIR
set CONV_RSH = ssh


time ./runsimple.sh '-y' edwardgpunode_output 8

times

