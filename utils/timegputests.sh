#!/bin/sh
###############################################################################
#
# timetests.sh - Run test harness programs and make table of times
#
# File:    timegputests.sh
# Author:  Alex Stivala
# Created: November 2012
#
#
# Run the test harness for a GPU impleemtnation, do not vary number of threads
# just do repetions, but make table for R read.table(,header=TRUE)
# compatilbe with that made by timtests.sh
#
# Usage:
#  timetests.sh  program
#
#     program is the test harness program to run  (eg iit_oahttsif_gpu_test)
#              (and writes a "elapsed time <x> ms" string to stdout where 
#               <x> is time in ms
#
#   output to stdout is table for reading in R
#
# $Id: timegputests.sh 4369 2012-11-20 01:54:33Z astivala $
# 
###############################################################################

# number of times to repeat each run
NUM_REPETITIONS=10


if [ $# -ne 1 ]; then
    echo "Usage: $0 program " 2>&1
    exit 1
fi

program=$1


echo "# Run as: $0 $*"
echo "# at: `date`"
echo "# by: `whoami`"
echo "# on: `uname -a`"

echo "threads iteration ms"

i=1
while [ $i -le $NUM_REPETITIONS ]; do
    ms=`${program}  ${numthreads} | grep "^elapsed time" | cut -d" " -f3`
    echo NA ${i} ${ms}
    i=`expr $i + 1`
done

