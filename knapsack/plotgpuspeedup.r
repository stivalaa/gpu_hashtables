###############################################################################
#
# plotgpuspeedup.r - plot GPU/CPU speedup graphs from timetests.sh output
#
# File:    plotgpuspeedup.r
# Author:  Alex Stivala
# Created: November 2012
#
# Plot GPU/CPU speedup bar graphs from summarize.sh output for different 
# variations of gpu knapsack implementation
#
# Usage:
#       R --vanilla --slave -f plotgpuspeedup.r 
#
#
#   Reads the rtab files teslam2070.rtab, teslam2070_httslf.rtab, etc.
#
#
#    output is PostScript file named knapsack_teslam2070speedup.eps
#
# 
# Requires the gplots package from CRAN to draw error bars.
#
# $Id: plotgpuspeedup.r 4320 2012-11-11 21:52:23Z astivala $
# 
###############################################################################

library(gplots)

#
# globals
#

colorvec=c('deepskyblue4','brown','red','turquoise','blue','purple','green','cyan','gray20','magenta','darkolivegreen2','midnightblue','magenta3','darkseagreen','violetred3','darkslategray3')
ltyvec=c(1,2,4,5,6,1,2,1,5,6,1,2,4,5,6,1,2)
pchvec=c(20,21,22,23,24,25)

filenamevec=c('teslam2070_norandomstart.rtab','teslam2070.rtab','teslam2070_httslf.rtab')
# and the names line up here
namevec=c('open addressing','open addressing + random starts','chaining + random starts')



plotspeedup <- function()
{
  epsfile <- 'knapsack_teslam2070speedup.eps'

  # EPS suitable for inserting into LaTeX
  postscript(epsfile, onefile=FALSE,paper="special",horizontal=FALSE, 
             width = 9, height = 6)


  means <- c()
  stdev <- c()
  ciw <- c()
  for (j in 1:length(filenamevec)) {
    timetab <- read.table(filenamevec[j], header=TRUE)
    basetime <- mean(subset(timetab, threads==0)$ms)
    means <- c(means,  mean(basetime/subset(timetab, is.na(threads))$ms) )
    thisstdev <-  sqrt(var(basetime/subset(timetab, is.na(threads))$ms)) 
    stdev <- c(stdev , thisstdev)
    n <- length(subset(timetab, is.na(threads))$ms)
    ciw <- c(ciw,  qt(0.975, n) * thisstdev / sqrt(n))
   }
   ylim <- c(0, 2.5)#max(means)+1)
   mp <- barplot2(means, names.arg = namevec, 
            beside = TRUE,
            ylab="GPU speedup over CPU baseline", ylim = ylim ,
            plot.ci = TRUE,
            ci.u = means + ciw, ci.l = means - ciw
            #col=colorvec[1:length(filenamevec)],
            )
   text(mp, means+0.1, format(means, digits=2), xpd=TRUE)
   box()
   dev.off()
}

#
# main
#
  
plotspeedup()

