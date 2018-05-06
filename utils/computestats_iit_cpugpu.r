###############################################################################
#
# computestats_iit_cpugpu.r - compute average times/speedups fort GPU/CPU on IIT test harness  from timetests.sh output
#
# File:    computestats_iit_cpugpu.r
# Author:  Alex Stivala
# Created: November 2012
#
# Compute GPU/CPU speedup and time stats from timetests.sh output for 
# different methods with results from the IIT style test harness.
#
# Usage:
#       R --vanilla --slave -f computestats_iit_cpugpu.r 
#
#
#   Reads the rtab files iit_oahttslf_ADDS_DELETES.rtab etc.
#                  e.g.  iit_oahttslf_50_10.rtab
#   in output directories edward_keys1000_output/ etc.
#
#
#    output is LaTeX table on stdout
#
# $Id: plottestspeedup_iit_cpugpu.r 4320 2012-11-11 21:52:23Z astivala $
# 
###############################################################################

library(gplots)


#
# globals
#

# these must line up so the GPU output in gpu_fileprefixvec[i] corresponds
# to the CPU output for the same method in cpu_fileprefixvec[i]
cpu_fileprefixvec=c('iit_cpu_oahttslf','iit_cpu_httslf','iit_cpu_lockfreehashtable')
gpu_fileprefixvec=c('iit_gpu_oahttslf','iit_gpu_httslf','iit_gpu_lockfreehashtable')
# and the names line up here
namevec=c('open addressing','separate chaining','IIT')

hostnamevec=c('edward')


# number of keys to run with
numkeys_vec=c(1000,10000,100000,1000000)

# percentages of ADD,DELETE operations
# (the remainder after ADD,DELETE are SEARCH).
opmix_matrix=rbind(c(30,0),c(10,10),c(20,20),c(40,40),c(50,10))

maxthreads <- 8
iterations <- 10

computestats <- function(hostname)
{

  overall_mean_best_cputimes <- c()
  overall_mean_gputimes <- c()
  overall_mean_speedup <- c()

  total_gputime_matrix <- mat.or.vec(length(cpu_fileprefixvec), iterations)
  total_best_cputime_matrix <- mat.or.vec(length(cpu_fileprefixvec), iterations)
  
  for (numkeys in numkeys_vec) {
    for (i in 1:dim(opmix_matrix)[1]) {
      adds <- opmix_matrix[i,1]
      deletes <- opmix_matrix[i,2]

      mean_speedup_by_thread  <- c()
      stddev_speedup_by_thread  <- c()
      mean_cputime_by_thread <- c()
      gputime_vector <- c()

      for (j in 1:length(cpu_fileprefixvec)) {

        rtabdir <- paste(hostname, "_keys", format(numkeys,scientific=F), 
                         "_output", sep='')
        cpu_rtabfile <- paste(cpu_fileprefixvec[j], adds, deletes,sep='_')
        cpu_rtabfile <- paste(cpu_rtabfile,hostname,'rtab',sep='.')
        cpu_rtabfile <- paste(rtabdir, cpu_rtabfile, sep='/')
        cpu_timetab <- read.table(cpu_rtabfile,header=TRUE)
        stopifnot(maxthreads == max(cpu_timetab$threads))
        stopifnot(iterations == max(cpu_timetab$iter))
        
        gpu_rtabfile <- paste(gpu_fileprefixvec[j], adds, deletes,sep='_')
        gpu_rtabfile <- paste(gpu_rtabfile,hostname,'rtab',sep='.')
        gpu_rtabfile <- paste(rtabdir, gpu_rtabfile, sep='/')
        gpu_timetab <- read.table(gpu_rtabfile,header=TRUE)
        stopifnot(iterations == max(gpu_timetab$iter))

        # we need to get mean for each value of threads for CPU implementation
        # but GPU only run with fixed set of threads

        cpu_times_by_thread_list <- lapply(c(1:maxthreads),
                                          FUN = function(x) subset(cpu_timetab,
                                                                    threads == x)$ms)

        gpu_speedup_by_thread_vector <- (lapply(cpu_times_by_thread_list,
                                              FUN = function(v) v / gpu_timetab$ms))

                                            

        mean_speedup_by_thread <-  c(mean_speedup_by_thread,
                                    lapply(gpu_speedup_by_thread_vector, mean))

        stddev_speedup_by_thread <- c(stddev_speedup_by_thread,
                                    lapply(gpu_speedup_by_thread_vector, sd))


        mean_cputime_by_thread <- c(mean_cputime_by_thread,
                                    lapply(cpu_times_by_thread_list, mean))

        gputime_vector <- c(gputime_vector, mean(gpu_timetab$ms))

        total_gputime_matrix[j,] <- total_gputime_matrix[j,] + gpu_timetab$ms

        best_thread <- which.min(lapply(cpu_times_by_thread_list, mean))

        ## print ( j)
        ## print(total_best_cputime_matrix[j,]) 
        ## print(best_thread) 
        total_best_cputime_matrix[j,] <- total_best_cputime_matrix[j,] +
          subset(cpu_timetab, threads == best_thread)$ms

      }
  

      mean_speedup_by_thread <- unlist(mean_speedup_by_thread)
      stddev_speedup_by_thread <- unlist(stddev_speedup_by_thread)


      n <- length(cpu_fileprefixvec)
      ciw <- qt(0.975, n) * stddev_speedup_by_thread / sqrt(n)

      # convert vector of speedups to matrix where each row is a different method
      gpu_speedup_over_cpu_threads_matrix <- matrix(mean_speedup_by_thread,
                                                    nrow=n, byrow=TRUE)
      ci_matrix <- matrix(ciw, nrow=n, byrow=TRUE)


      mean_cputime_threads_matrix <- matrix(unlist(mean_cputime_by_thread),
                                            nrow=n, byrow=TRUE)


      # get the index in each method (row of matrix) i.e. number of threads
      # for which mean cputime is smallest (i.e. the number of threads on which
      # CPU perofmrance is best for that method)
      best_threads <- lapply(1:n, FUN = function(i) 
                             which.min(mean_cputime_threads_matrix[i,]))

      #print(paste(numkeys, adds, deletes))
      for (k in 1:dim(gpu_speedup_over_cpu_threads_matrix)[1]) {
        #print(paste("     ",namevec[k], best_threads[[k]],
        #            mean_cputime_threads_matrix[k, best_threads[[k]]],
        #            gputime_vector[k],
        #            gpu_speedup_over_cpu_threads_matrix[k, best_threads[[k]]]))

        overall_mean_best_cputimes <- c(overall_mean_best_cputimes,
                            mean_cputime_threads_matrix[k, best_threads[[k]]])
        overall_mean_gputimes <- c(overall_mean_gputimes, gputime_vector[k])
        overall_mean_speedup <- c(overall_mean_speedup, 
                   gpu_speedup_over_cpu_threads_matrix[k, best_threads[[k]]])


      }
    }
  }


  # convert vector of speedups to matrix where each row is a different method
  overall_mean_best_cputimes_matrix <- matrix(overall_mean_best_cputimes,
                                              nrow=n, byrow=F)
  overall_mean_gputimes_matrix <- matrix(overall_mean_gputimes, nrow=n,
                                         byrow=F)
  overall_mean_speedup_matrix <- matrix(overall_mean_speedup, nrow=n,byrow=F)

  ## print(total_gputime_matrix)#XXX
  ## print(total_best_cputime_matrix)#XXX


  

  
  cat("\\begin{tabular}{lrrrrr} \n")
  cat("\\hline\n")
  cat("  & \\multicolumn{2}{c}{mean elapsed time(ms)} & \\\\ \n")
  cat("implementation & CPU & GPU  & speedup \\\\ \n")
  cat("\\hline\n")
  for (k in 1:n) {
      cputime_ciw <- qt(0.975, iterations) * sd(total_best_cputime_matrix[k,]) / sqrt(iterations)
      gputime_ciw <- qt(0.975, iterations) * sd(total_gputime_matrix[k,]) / sqrt(iterations)

    cat(sprintf("%20s & $%.0f \\pm %.1f$ & $%.0f \\pm %.1f$ & %.1f \\\\ \n",
                namevec[k],
                mean(total_best_cputime_matrix[k,]), cputime_ciw,
                mean(total_gputime_matrix[k,]), gputime_ciw,
                mean(total_best_cputime_matrix[k,]) / mean(total_gputime_matrix[k,])
               ))
  }
  cat("\\hline\n")
  cat("\\end{tabular}\n")
  #print(overall_mean_gputimes_matrix)
}

#
# main
#
  
for (hostname in hostnamevec) {
  computestats(hostname)
}

