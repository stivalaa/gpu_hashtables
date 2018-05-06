#!/bin/bash

# pbs launching script:
 	
#PBS -N gpu_norandomstart_knapsack

# use gpu queue to put on a gpu node  (8 cores)
#PBS -q gpu

#PBS -l walltime=200:00:0

# Reserve 8 cores on the single node 
#PBS -l nodes=1:ppn=8


cd $PBS_O_WORKDIR
set CONV_RSH = ssh


time ./runall_gpu_norandomstart.sh '' teslam2070_norandomstart_output

times

