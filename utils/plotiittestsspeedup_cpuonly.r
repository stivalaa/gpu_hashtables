###############################################################################
#
# plotiittestsspeedup_cpuonly.r - plot speedup graph from timetests.sh output
#
# File:    plotiittestsspeedup_cpuonly.r
# Author:  Alex Stivala
# Created: May 2009
#
# Plot speedup time graphs from timetests.sh output for different meethods on
# one graph for each platform.
#
# Usage:
#       R --vanilla --slave -f plotiittestsspeedup_cpuonly.r --args ADDS DELETES
#
#       ADDS is percentage of ADD operations
#       DELETES is percetnags of DELETE operations
#
#   Reads the rtab files iit_oahttslf_ADDS_DELETES.rtab etc.
#                  e.g.  iit_oahttslf_50_10.rtab
#
#
#    output is PostScript file named
#         hostname.testspeedupiit_cpuonly_ADDS_DELETES.eps
#    where hostmame is name of the test machine.
#
# 
# Requires the gplots package from CRAN to draw error bars.
#
# $Id: plotiittestsspeedup_cpuonly.r 4320 2012-11-11 21:52:23Z astivala $
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
namevec=c('open addressing', 'separate chaining', 'IIT')
fileprefixvec=c('iit_oahttslf','iit_httslf','iit_lockfreehashtable')

hostnamevec=c('edward')


plottime <- function(hostname)
{
  epsfile <- paste(hostname,'testspeedupiit_cpuonly',sep='.')
  epsfile <- paste(epsfile, adds, deletes,sep='_')
  epsfile <- paste(epsfile,'eps',sep='.')

  # EPS suitable for inserting into LaTeX
  postscript(epsfile, onefile=FALSE,paper="special",horizontal=FALSE, 
             width = 9, height = 6)

  for (j in 1:length(fileprefixvec)) {

    if (fileprefixvec[j] == 'tbbhashmap' && hostname == 'edward')
      break

    rtabfile <- paste(fileprefixvec[j], adds, deletes,sep='_')
    rtabfile <- paste(rtabfile,hostname,'rtab',sep='.')

    timetab <- read.table(rtabfile,header=TRUE)


    maxthreads = max(timetab$threads)
    

    basetime = mean(subset(timetab, threads==1)$ms)

    x <- subset(subset(timetab, iteration==1),threads > 0)$threads
    means <- c()
    stdev <- c()
    ciw <- c()
    for (i in x) { 
      # FIXME this code is ugly and inefficient, should use sapply or something
      means <-  c(means, mean(basetime/subset(timetab, threads==i)$ms))
      thisstdev <-  sqrt(var(basetime/subset(timetab, threads==i)$ms))
      stdev <-  c(stdev, thisstdev)
      n <- length(subset(timetab, threads==i)$ms)
      ciw <- c(ciw, qt(0.975, n) * thisstdev / sqrt(n))
    }

    maxspeedupvalue <- max(means)
    print(hostname)
    print(fileprefixvec[j])
    print(maxspeedupvalue)
    
    if (hostname == 'tango')
        maxspeedup <- 8
    else if (hostname == 'edward')
        maxspeedup = 16
    else if (hostname == 'edwardgpunode')
        maxspeedup <- 8
    else
        maxspeedup <- 4

    plotCI(x, y=means, uiw=ciw, xlab="threads", ylab="speedup",
           main=paste(adds, "% add, ", deletes, "% delete, ", 
             100-(as.integer(adds)+as.integer(deletes)), "% search",sep=""),
           ylim=c(0,maxspeedup),
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

