#!/bin/sh
###############################################################################
#
# runall_gpu.sh - Run run_instances.py multiple times for means on GPU
#
# File:    runall_gpu.sh
# Author:  Alex Stivala
# Created: April 2009
#
#
# Run the run_intances.py script to execute GPU implementation multiple times
#
# Usage:
#  runall_gpu.sh  knapsack_options outdir 
#
#     knapsack_options are options to pass to knapsack_program
#
# Creates outdir if it does not exist. WARNING: will overwrite files in
# outdir.
#
# Requires run_instances.py to be in PATH
#
# $Id: runall_gpu_httslf.sh 4549 2013-01-12 22:28:33Z astivala $
# 
###############################################################################

MAXITERATIONS=10

TIMEGUARD=../utils/timeguard
TIMEOUT=25200

if [ $# -ne 2 ]; then
    echo "Usage: $0  knapsack_options outdir" >&2
    exit 1
fi

knapsack_options=$1
outdir=$2

if [ ! -d ${outdir} ]; then
    mkdir ${outdir}
fi



iter=1
while [ $iter -le $MAXITERATIONS ]; do
    /usr/bin/time $TIMEGUARD $TIMEOUT run_instances.py  knapsack_gpu_randomstart_nr_httslf "${knapsack_options}" instances > ${outdir}/instances_httslf_NA_${iter}.out 2> ${outdir}/instances_httslf_NA_${iter}.err
    iter=`expr $iter + 1`
done


/usr/bin/time $TIMEGUARD $TIMEOUT run_instances.py  knapsack_simple "${knapsack_options}" instances > ${outdir}/instances_simple.out 2> ${outdir}/instances_simple.err

