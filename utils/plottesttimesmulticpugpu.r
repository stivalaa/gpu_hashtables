###############################################################################
#
# plottesttimesmulticpugpu.r - plot elapsed time graph from timetests.sh output
#
# File:    plottesttimesmulticpugpu.r
# Author:  Alex Stivala
# Created: May 2009
#
# Plot elapsed time graphs from timetests.sh output for different meethods on
# one graph for each platform. This version is for the oahttslftest style
# tests, run on CPU and GPU for isntance output from
# timetests_gpu_edward_pbs_script.sh
#
# Usage:
#       R --vanilla --slave -f plottesttimesmulticpugpu.r
#
#
#   Reads the rtab files gpu_lockfreehashtable.edward.rtab,
#                        cpu_lockfreehasthtable.edward.rtab,
#                        gpu_oahttslf.edward.rtab,
#                        cpu_oahttslf.edward.rtab, etc.
#
#
#    output is PostScript file named hostname.testtimesmulticpugpu.eps
#    where hostmame is name of the test machine.
#
# 
# Requires the gplots package from CRAN to draw error bars.
#
# $Id: plottesttimesmulticpugpu.r 4320 2012-11-11 21:52:23Z astivala $
# 
###############################################################################

library(gplots)

#
# globals
#

colorvec=c('deepskyblue4','brown','red','turquoise','blue','purple','green','cyan','gray20','magenta','darkolivegreen2','midnightblue','magenta3','darkseagreen','violetred3','darkslategray3')
ltyvec=c(1,2,4,5,6,1,2,1,5,6,1,2,4,5,6,1,2)
pchvec=c(20,21,22,23,24,25)
namevec=c('CPU open addressing', 'CPU separate chaining' , 'CPU IIT', 'GPU open addressing', 'GPU separate chaining', 'GPU IIT' )
fileprefixvec=c('cpu_oahttslf',
                'cpu_httslf',
                'cpu_lockfreehashtable',
                'gpu_oahttslf',
                'gpu_httslf',
                'gpu_lockfreehashtable')


hostnamevec=c('edward')


plottime <- function(hostname)
{
  epsfile <- paste(hostname,'testtimesmulticpugpu',sep='.')
  epsfile <- paste(epsfile,'eps',sep='.')

  # EPS suitable for inserting into LaTeX
  postscript(epsfile, onefile=FALSE,paper="special",horizontal=FALSE, 
             width = 9, height = 6)

  for (j in 1:length(fileprefixvec)) {

    if (fileprefixvec[j] == 'tbbhashmap' && hostname == 'edward')
      break

    rtabfile <- fileprefixvec[j]
    rtabfile <- paste(rtabfile,hostname,'rtab',sep='.')

    timetab <- read.table(rtabfile,header=TRUE)


    maxthreads = max(timetab$threads)
    

    x <- subset(subset(timetab, threads > 0), iteration==1)$threads
    means <- c()
    stdev <- c()
    ciw <- c()
    for (i in x) { 
      # FIXME this code is ugly and inefficient, should use sapply or something
      means <-  c(means, mean(subset(timetab, threads==i)$ms))
      thisstdev <-  sqrt(var(subset(timetab, threads==i)$ms))
      stdev <-  c(stdev, thisstdev)
      n <- length(subset(timetab, threads==i)$ms)
      ciw <- c(ciw, qt(0.975, n) * thisstdev / sqrt(n))
    }

    # gpu results have NA for threads
    gpux <- subset(timetab, is.na(threads))
    if (length(gpux$iter) > 0) {
        means <-  c(means, mean(subset(timetab)$ms))
        thisstdev <-  sqrt(var(subset(timetab)$ms))
        stdev <-  c(stdev, thisstdev)
        n <- length(subset(timetab)$ms)
        ciw <- c(ciw, qt(0.975, n) * thisstdev / sqrt(n))
      x <- c(1:8); means <-rep(means,8); stdev <- rep(stdev,8); ciw <- rep(ciw,8) #dodgy, just hardocde for a straight line repeating same results for each thread value for GPU
    }



    print(hostname)
    print(fileprefixvec[j])
    print(x)
    print(means)
    
    plotCI(x, y=means, uiw=ciw, xlab="threads", ylab="elapsed time (ms)",
     log="y",
           ylim=c(1,100),
           xpd = TRUE,
           add = (j > 1),
           col=colorvec[j],pch=pchvec[j])
    lines(x,means,col=colorvec[j],lty=ltyvec[j])
  }
  legend('topright', col=colorvec, lty=ltyvec, pch=pchvec, legend=namevec,bty='n')  
  dev.off()
}

#
# main
#
  
for (hostname in hostnamevec) {
  plottime(hostname)
}

