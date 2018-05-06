###############################################################################
#
# plottestspeedup_iit_cpugpu.r - plot GPU/CPU speedup graphs from timetests.sh output
#
# File:    plottestspeedup_iit_cpugpu.r
# Author:  Alex Stivala
# Created: November 2012
#
# Plot GPU/CPU speedup graphs from timetests.sh output for different methods
# on on graph for each platform with results from the IIT style test harness.
#
# Usage:
#       R --vanilla --slave -f plottestspeedup_iit_cpugpu.r --args ADDS DELETES
#
#       ADDS is percentage of ADD operations
#       DELETES is percetnags of DELETE operations
#
#   Reads the rtab files iit_oahttslf_ADDS_DELETES.rtab etc.
#                  e.g.  iit_oahttslf_50_10.rtab
#
#
#    output is PostScript file named hostname.testspeedupiit_ADDS_DELETES.eps
#    where hostmame is name of the test machine.
#
# 
# Requires the gplots package from CRAN to draw error bars.
#
# $Id: plottestspeedup_iit_cpugpu.r 4320 2012-11-11 21:52:23Z astivala $
# 
###############################################################################

library(gplots)

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) {
  print("must have two args: ADDS DELETES\n")
  quit(save="no",status=1)
}

adds <- args[1]
deletes <- args[2]

#
# globals
#

colorvec=c('deepskyblue4','brown','red','turquoise','blue','purple','green','cyan','gray20','magenta','darkolivegreen2','midnightblue','magenta3','darkseagreen','violetred3','darkslategray3')
ltyvec=c(1,2,4,5,6,1,2,1,5,6,1,2,4,5,6,1,2)
pchvec=c(20,21,22,23,24,25)
# these must line up so the GPU output in gpu_fileprefixvec[i] corresponds
# to the CPU output for the same method in cpu_fileprefixvec[i]
cpu_fileprefixvec=c('iit_cpu_oahttslf','iit_cpu_httslf','iit_cpu_lockfreehashtable')
gpu_fileprefixvec=c('iit_gpu_oahttslf','iit_gpu_httslf','iit_gpu_lockfreehashtable')
# and the names line up here
namevec=c('open addressing','separate chaining','IIT')

hostnamevec=c('edward')


plottime <- function(hostname)
{
  epsfile <- paste(hostname,'testspeedupiit',sep='.')
  epsfile <- paste(epsfile, adds, deletes,sep='_')
  epsfile <- paste(epsfile,'eps',sep='.')

  # EPS suitable for inserting into LaTeX
  postscript(epsfile, onefile=FALSE,paper="special",horizontal=FALSE, 
             width = 9, height = 6)

  mean_speedup_by_thread  <- c()
  stddev_speedup_by_thread  <- c()
  mean_cputime_by_thread <- c()

  for (j in 1:length(cpu_fileprefixvec)) {

    cpu_rtabfile <- paste(cpu_fileprefixvec[j], adds, deletes,sep='_')
    cpu_rtabfile <- paste(cpu_rtabfile,hostname,'rtab',sep='.')
    cpu_timetab <- read.table(cpu_rtabfile,header=TRUE)
    maxthreads = max(cpu_timetab$threads)
    
    gpu_rtabfile <- paste(gpu_fileprefixvec[j], adds, deletes,sep='_')
    gpu_rtabfile <- paste(gpu_rtabfile,hostname,'rtab',sep='.')
    gpu_timetab <- read.table(gpu_rtabfile,header=TRUE)

    # we need to get mean for each value of threads for CPU implementation
    # but GPU only run with fixed set of threads

    cpu_times_by_thread_list <- lapply(c(1:maxthreads),
                                       FUN = function(i) subset(cpu_timetab,
                                                                threads == i)$ms)
    gpu_speedup_by_thread_vector <- (lapply(cpu_times_by_thread_list,
                                           FUN = function(v) v / gpu_timetab$ms))

                                         

    mean_speedup_by_thread <-  c(mean_speedup_by_thread,
                                 lapply(gpu_speedup_by_thread_vector, mean))

    stddev_speedup_by_thread <- c(stddev_speedup_by_thread,
                                 lapply(gpu_speedup_by_thread_vector, sd))


    print(cpu_times_by_thread_list)
    mean_cputime_by_thread = c(mean_cputime_by_thread,
                               lapply(cpu_times_by_thread_list, mean))
    print(mean_cputime_by_thread)
  }

  mean_speedup_by_thread <- unlist(mean_speedup_by_thread)
  stddev_speedup_by_thread <- unlist(stddev_speedup_by_thread)
#  print(mean_speedup_by_thread)
#  print(stddev_speedup_by_thread)

  n <- length(cpu_fileprefixvec)
  ciw <- qt(0.975, n) * stddev_speedup_by_thread / sqrt(n)

  # convert vector of speedups to matrix where each row is a different method
  gpu_speedup_over_cpu_threads_matrix <- matrix(mean_speedup_by_thread,
                                                nrow=n, byrow=TRUE)
  ci_matrix <- matrix(ciw, nrow=n, byrow=TRUE)

#  print(gpu_speedup_over_cpu_threads_matrix)
#  print(ci_matrix)


  mean_cputime_threads_matrix <- matrix(unlist(mean_cputime_by_thread),
                                        nrow=n, byrow=TRUE)
  print(mean_cputime_threads_matrix)

  # get the index in each method (row of matrix) i.e. number of threads
  # for which mean cputime is smallest (i.e. the number of threads on which
  # CPU perofmrance is best for that method)
  best_threads <- lapply(1:n, FUN = function(i) 
                                     which(mean_cputime_threads_matrix[i,] ==
                                           min(mean_cputime_threads_matrix[i,])))
  print (best_threads)
  ylim = c(0,30)
  barplot2(gpu_speedup_over_cpu_threads_matrix, xlab="CPU threads", 
             ylim=ylim,
             col=colorvec[1:n],
             beside = TRUE
    )
  legend('topright', fill=colorvec, legend=namevec,bty='n')
  # plot a 2nd time (add=TRUE) to use density to highlght bars for
  # speedup over number of threads on CPU with best performance
  # FIXME this double loop code is ugly and inefficient and not R-like, should somehow use an apply() function or other vecotrization
  density_vector <- rep(0, n*maxthreads)
  for (i in 1:maxthreads)
    for (j in 1:n)
      if (best_threads[[j]][1] == i)
        density_vector[i*n + j -n] = 10
  print(density_vector)
  mp<-barplot2(gpu_speedup_over_cpu_threads_matrix, xlab="CPU threads", 
             plot.ci = TRUE,
             ci.u = gpu_speedup_over_cpu_threads_matrix + ci_matrix, 
             ci.l = gpu_speedup_over_cpu_threads_matrix - ci_matrix,
             ylab="GPU speedup over CPU",
             ylim=ylim,
             names.arg = c(1:maxthreads),
             beside = TRUE,
               main=paste(adds, "% add, ", deletes, "% delete, ", 
                  100-(as.integer(adds)+as.integer(deletes)), "% search",sep=""),,
             add=TRUE,
             col=rep("black", n*maxthreads),
             density=density_vector
    )
  text(mp, gpu_speedup_over_cpu_threads_matrix + 3,
       format(gpu_speedup_over_cpu_threads_matrix, digits=2),
       xpd=TRUE)
  box()
  dev.off()
}

#
# main
#
  
for (hostname in hostnamevec) {
  plottime(hostname)
}

