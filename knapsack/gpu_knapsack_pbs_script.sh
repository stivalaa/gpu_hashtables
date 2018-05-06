#!/bin/bash

# pbs launching script:
 	
#PBS -N cuda_dp_knapsack

#PBS -q gpu
#PBS -l walltime=2:00:0
#PBS -l nodes=1:ppn=8

module load cuda

cd $PBS_O_WORKDIR
set CONV_RSH = ssh

echo -n "Start: "
date

INSTANCE=instances/gen.3.500.500.88
#INSTANCE=instances/gen.4.500.500.58
#INSTANCE=instances/gen.1.500.500.10
time output=`./knapsack_simple   < ${INSTANCE}`
echo -n "simple: "
echo $output
correct_profit=`echo "$output" | awk '{print $1}'`
echo -n "httslf-r1: "
time ./knapsack_httslf -r1  < ${INSTANCE}
echo -n "httslf-r8: "
time ./knapsack_httslf -r8  < ${INSTANCE}
echo -n "oahttslf-r1: "
time ./knapsack_oahttslf -r1  < ${INSTANCE}
echo -n "oahttslf-r8: "
time ./knapsack_oahttslf -r8  < ${INSTANCE}
echo -n "oahttslf_nr-r1: "
time ./knapsack_nr_oahttslf -r1  < ${INSTANCE}
echo -n "oahttslf_nr-r8: "
time ./knapsack_nr_oahttslf -r8  < ${INSTANCE}
echo -n "oahttslf_randomstart-r1: "
time ./knapsack_randomstart_oahttslf -r1 < ${INSTANCE}
echo -n "oahttslf_randomstart-r8: "
time ./knapsack_randomstart_oahttslf -r8 < ${INSTANCE}
echo -n "oahttslf_gpu: "
time output=`./knapsack_gpu_oahttslf  < ${INSTANCE} | grep -v '^>' | grep -v '^$'`
profit=`echo "$output" | awk '{print $1}'`
echo $output
if [ $profit -ne $correct_profit ]; then
  echo FAILED
fi
echo -n "oahttslf_gpu_nr: "
time output=`./knapsack_gpu_nr_oahttslf  < ${INSTANCE} | grep -v '^>' | grep -v '^$'`
profit=`echo "$output" | awk '{print $1}'`
echo $output
if [ $profit -ne $correct_profit ]; then
  echo FAILED
fi
echo -n "httslf_gpu_nr: "
time output=`./knapsack_gpu_nr_httslf  < ${INSTANCE} | grep -v '^>' | grep -v '^$'`
profit=`echo "$output" | awk '{print $1}'`
echo $output
if [ $profit -ne $correct_profit ]; then
  echo FAILED
fi
echo -n "oahttslf_randomstart_gpu_nr: "
time output=`./knapsack_gpu_randomstart_nr_oahttslf  < ${INSTANCE} | grep -v '^>' | grep -v '^$'`
profit=`echo "$output" | awk '{print $1}'`
echo $output
if [ $profit -ne $correct_profit ]; then
  echo FAILED
fi
echo -n "httslf_randomstart_gpu_nr: "
time output=`./knapsack_gpu_randomstart_nr_httslf  < ${INSTANCE} | grep -v '^>' | grep -v '^$'`
profit=`echo "$output" | awk '{print $1}'`
echo $output
if [ $profit -ne $correct_profit ]; then
  echo FAILED
fi



times
echo -n "End: "
date

