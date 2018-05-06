#!/bin/sh
###############################################################################
#
# summarize_instrument.sh - make table for R of instrumentation counts
#
# File:    summarize_instrument.sh
# Author:  Alex Stivala
# Created: May 2009
#
#
# sum up the reuse and hash counts  in each INSTRUMENT line, and output
# in format for R read.table(,header=TRUE)
#
# Usage:
#  summarize_instrument.sh
#
#    output is to stdout.
#    run from the output directory (reads files in cwd)
#
# Requires GNU gawk
#
# $Id: summarize_instrument.sh 4569 2013-01-19 23:14:19Z astivala $
# 
###############################################################################

grep '# on:' instances_oahttslf_1_1.out
echo "threads iter re hc hn"
for i in instances_oahttslf_*.out
do
  thread=`echo ${i} | cut -d '_' -f3`
  iter=`echo ${i} | cut -d '_' -f4| cut -d '.' -f1`
  echo -n "${thread} ${iter} "
  grep INSTRUMENT ${i} |  gawk '{
    hc = gensub(/.* hc=([0-9]*).*/,"\\1",1); 
    re = gensub(/.*,re=([0-9]*).*/,"\\1",1);
    hn = gensub(/.*,hn=([0-9]*).*/,"\\1",1);
    total_hc += hc; 
    total_re += re;
    total_hn += hn;
    } 
    END {print total_re,total_hc,total_hn}'
done



