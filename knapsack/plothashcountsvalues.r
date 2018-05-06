###############################################################################
#
# plothascountsvalues.r - plot inertion and unique values bar graphs
#
# File:    plothascountsvalues.r
# Author:  Alex Stivala
# Created: January 2013
#
#
#
# Plot hashcount (insertion count) and unique value inserted (entries in
# hash table at end) frmo instrumentatino output summarized by
# summarize_instrument.sh
#
# Usage:
#       R --vanilla --slave -f plothascountsvalues.r 
#
#  Reads the rtab files teslam2070.instrumentrtab,  etc.
#  Output is  PostScript file knapsack_hashandvalucounts.eps 
#
# Requires the gplots package from CRAN to draw error bars.
#
# $Id: plothascountsvalues.r 2526 2009-06-15 06:23:41Z astivala $
# 
###############################################################################

library(gplots)

#
# globals
#

colorvec=c('deepskyblue4','brown','red','turquoise','blue','purple','green','cyan','gray20','magenta','darkolivegreen2','midnightblue','magenta3','darkseagreen','violetred3','darkslategray3')
ltyvec=c(1,2,4,5,6,1,2,1,5,6,1,2,4,5,6,1,2)
pchvec=c(20,21,22,23,24,25)
# these must line up so the GPU output in gpu_fileprefixvec[i] corresponds

maxthreads <- 8 # number of CPU threads to use

filenamevec <- c('edwardgpunode.instrument.rtab',
                 'edwardgpunode_randomstart.instrument.rtab', 
                 'teslam2070_norandomstart.instrument.rtab',  
                 'teslam2070.instrument.rtab')
namevec=c('CPU', 'CPU random starts', 'GPU', 'GPU random starts')

count_types=c('total computations (h/h1)','subproblems computed (v/v1)')
basefilename <- filenamevec[1] # contains threads==1 as baseline



plotcountsandvaluesincrease <- function()
{
  epsfile <- 'knapsack_hashandvaluecounts.eps'

  # EPS suitable for inserting into LaTeX
  postscript(epsfile, onefile=FALSE,paper="special",horizontal=FALSE, 
             width = 9, height = 6)


  basertab <- read.table(basefilename, header=T)
  basevaluecount <- mean(subset(basertab, threads==1)$hn)
  basehashcount <- mean(subset(basertab, threads==1)$hc)

  means <- c()
  stdev <- c()
  ciw <- c()
  for (j in 1:length(filenamevec)) {
    rtab <- read.table(filenamevec[j], header=TRUE)
    if (!is.na(pmatch('GPU', namevec[j])))
      isgpu <- T
    else
      isgpu <- F
    if (isgpu)  {
      means <- c(means,  mean(subset(rtab, is.na(threads))$hc/basehashcount) )
      means <- c(means,  mean(subset(rtab, is.na(threads))$hn/basevaluecount) )
    }
    else {
      means <- c(means,  mean(subset(rtab, threads == maxthreads)$hc/basehashcount) )
      means <- c(means,  mean(subset(rtab, threads == maxthreads)$hn/basevaluecount) )
    }
    this_hashstdev <-  sqrt(var(subset(rtab, is.na(threads))$hc/basehashcount)) 
    this_valuestdev <-  sqrt(var(subset(rtab, is.na(threads))$hn/basevaluecount)) 
    stdev <- c(stdev , this_hashstdev, this_valuestdev)
    n <- length(subset(rtab, is.na(threads))$hn)
    ciw <- c(ciw,  qt(0.975, n) * this_hashstdev / sqrt(n))
    ciw <- c(ciw,  qt(0.975, n) * this_valuestdev / sqrt(n))
   }
   mean_increase_matrix <- matrix(means, nrow = length(count_types), byrow=F)
   ci_matrix <- matrix(ciw,  nrow = length(count_types)   , byrow=F)

   ylim <- c(0,110)
   mp <- barplot2(mean_increase_matrix, #names.arg = namevec, 
            beside = TRUE,
            ylab="computation increase ratio",  ylim=ylim,
            plot.ci = TRUE,
            ci.u = mean_increase_matrix + ci_matrix,
            ci.l = mean_increase_matrix - ci_matrix,
            col=colorvec[1:2],
            names.arg=namevec
            )
   text(mp, mean_increase_matrix+05, format(means, digits=2), xpd=TRUE)
   legend('topright', fill=colorvec, legend=count_types,bty='n')
   box()
   dev.off()
}

#
# main
#
  
plotcountsandvaluesincrease()
