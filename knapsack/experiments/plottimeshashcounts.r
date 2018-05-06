###############################################################################
#
# plottimeshashcounts.r - plot time and hash counts graphs of single experimetns
#
# File:    plottimeshashcounts.r
# Author:  Alex Stivala
# Created: January 2013
#
#
#
# Plot graphs of teims and hascounts for single instance epxeriments
#
# Usage:
#       R --vanilla --slave -f plottimeshashcounts.r
#
# $Id: plottimeshashcounts.r 2526 2009-06-15 06:23:41Z astivala $
# 
###############################################################################

#
# globals
#

colorvec=c('deepskyblue4','brown','red','turquoise','blue','purple','green','cyan','gray20','magenta','darkolivegreen2','midnightblue','magenta3','darkseagreen','violetred3','darkslategray3')
ltyvec=c(1,2,4,5,6,1,2,1,5,6,1,2,4,5,6,1,2)
pchvec=c(20,21,22,23,24,25)


randomstart_times <- read.table('thread_kerneltime_gpu_randomstart_nr_oahttslf.dat')
norandomstart_times <- read.table('thread_kerneltime_gpu_nr_oahttslf.dat')

randomstart_hashcounts <- read.table('thread_hashcount_gpu_randomstart_nr_oahttslf.dat')
norandomstart_hashcounts<- read.table('thread_hashcount_gpu_nr_oahttslf.dat')



time_filenamevec=c('thread_kerneltime_gpu_nr_oahttslf.dat','thread_kerneltime_gpu_randomstart_nr_oahttslf.dat')
# and the names line up here
time_namevec=c('without random starts', 'with random starts')
hashcount_filenamevec=c('thread_hashcount_gpu_nr_oahttslf.dat','thread_hashcount_gpu_randomstart_nr_oahttslf.dat')
# and the names line up here
hashcount_namevec=c('without random starts', 'with random starts')




time_ylim <- c(0, 70)
hashcount_ylim <- c(1, 220)

# EPS suitable for inserting into LaTeX
postscript('thread_kerneltime_gpu_nr_oahttslf.eps',
           onefile=FALSE,paper="special",horizontal=FALSE, 
           width = 9, height = 6)
plot(norandomstart_times$V1, norandomstart_times$V2/1000,
     xlab = "GPU threads", ylab ="knapsack kernel elapsed time (s)",
     ylim = time_ylim, main = "without random starts")

# EPS suitable for inserting into LaTeX
postscript('thread_kerneltime_gpu_randomstart_oahttslf.eps',
           onefile=FALSE,paper="special",horizontal=FALSE, 
           width = 9, height = 6)
plot(randomstart_times$V1, randomstart_times$V2/1000,
     xlab = "GPU threads", ylab ="knapsack kernel elapsed time (s)",
     ylim = time_ylim, main = "with random starts")


# EPS suitable for inserting into LaTeX
postscript('thread_hashcount_gpu_nr_oahttslf.eps',
           onefile=FALSE,paper="special",horizontal=FALSE, 
           width = 9, height = 6)
plot(norandomstart_hashcounts$V1, norandomstart_hashcounts$V2/norandomstart_hashcounts$V2[1],
     xlab = "GPU threads", 
     ylab ="increase in total computations (h/h1)",
     ylim = hashcount_ylim, main = "without random starts")

# EPS suitable for inserting into LaTeX
postscript('thread_hashcount_gpu_randomstart_nr_oahttslf.eps',
           onefile=FALSE,paper="special",horizontal=FALSE, 
           width = 9, height = 6)
plot(randomstart_hashcounts$V1, randomstart_hashcounts$V2/randomstart_hashcounts$V2[1],
     xlab = "GPU threads", 
     ylab ="increase in total computations (h/h1)",
     ylim = c(1,5), main = "with random starts")

# EPS suitable for inserting into LaTeX
postscript('gpu_thread_kerneltime.eps', onefile=FALSE,paper="special",
           horizontal=FALSE, width = 9, height = 6)
for (j in 1:length(time_filenamevec)) {
  rtabfile <- time_filenamevec[j]
  rtab <- read.table(rtabfile)
  if (j == 1) {
    plot(rtab$V1, rtab$V2/1000,
         xlab = "GPU threads", 
         ylab ="knapsack kernel elepased time (s)",
         ylim = time_ylim,
         col=colorvec[j], pch=pchvec[j])
  }
  else {
    points(rtab$V1, rtab$V2/1000, col=colorvec[j], pch=pchvec[j])
  }
#  lines(rtab$V1, rtab$V2/1000, col=colorvec[j], pch=pchvec[j])
}
legend('topright', col=colorvec, pch=pchvec, legend=time_namevec, bty='n')

# EPS suitable for inserting into LaTeX
postscript('gpu_thread_hashcountincrease.eps', onefile=FALSE,paper="special",
           horizontal=FALSE, width = 9, height = 6)
for (j in 1:length(time_filenamevec)) {
  rtabfile <- hashcount_filenamevec[j]
  rtab <- read.table(rtabfile)
  if (j == 1) {
    plot(rtab$V1, rtab$V2/rtab$V2[1],
         xlab = "GPU threads", 
         ylab ="increase in total computations (h/h1)",
         ylim = hashcount_ylim,,
         col=colorvec[j], pch=pchvec[j])
  }
  else {
    points(rtab$V1, rtab$V2/rtab$V2[1], col=colorvec[j], pch=pchvec[j])
  }
}
legend('topleft', col=colorvec, pch=pchvec, legend=time_namevec, bty='n')

dev.off()

