#!/bin/bash

# pbs launching script:
 	
#PBS -N gpu_memcheck_iit_httslf

#PBS -l walltime=96:00:00
#PBS -q gpu

# Reserve 4 cores on a gpu node, which has 8 core, since there are two gpu
# card per node and we only use one, can run another gpu job onthis node
#PBS -l nodes=1:ppn=4


cd $PBS_O_WORKDIR
set CONV_RSH = ssh

module load cuda

echo -n "Start: "
date

while [ $? -eq 0 ]; do
  #cuda-memcheck ./iit_httslf_gpu_test 100000 30 0
                 ./iit_httslf_gpu_test 100000 30 0
done

echo -n "End: "
date
times

