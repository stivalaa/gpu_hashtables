###############################################################################
#
# File      : Makefile
# Author    : Alex Stivala (astivala)
# Created   : April 2009
#
# $Id: Makefile 4634 2013-01-31 03:17:42Z astivala $
#
# Makefile for building the threaded dp knapsack programs.
#
#
# The default target "all" will compile and link the program.
# The "clean" target will remove object and executable files.
# The "times" target will make table of times for tests
# The "graphs" target will make speedup graph from times, using R.
# The "runall" target will run all the challenge problems, on 1 to n
#   cores where n is number of cores on machine (WARNNIG: may take a long
#   time, only do this when you ened to re-run, idea is these are stored
#    in svn and can just rebuild tables/graphs with times/graphs target).
#
#
#
# set MODE=DEBUG to build with debugging and profiling on. Otherwise
# default is to build with optimizations on and no debug or profile.
#
# This Makefile uses implicit rules to compile .c files into .o files
# and some features specific to GNU Makee.
#
###############################################################################


# include the common.mk file from parent diretory, and override it with
# local.mk if it exists

include ../common.mk
-include ../local.mk


INCDIRS =  -I../utils
NBDSINC =  -I../nbds.0.4.3/include

CUDA_CPPFLAGS += $(INCDIRS)


# NVIDIA CUDA compute capability - we require 2.x for atomic CAS 64 bits,
# and recurisve device functions
GPU_ARCH = sm_20


CUDA_KERNELS  = 
CUDA_HOSTCODE = knapsack_gpu_iit.cu knapsack_gpu_oahttslf.cu knapsack_gpu_nr_oahttslf.cu knapsack_gpu_randomstart_nr_oahttslf.cu knapsack_randomstart_oahttslf.cu
CUDA_OBJS = 
CUDA_HOST_OBJS= $(CUDA_HOSTCODE:.cu=.o)  knapsack_gpu_randomstart_nr_httslf.o knapsack_gpu_nr_httslf.o


COMMONSRCS  = 
HTTSLFSRCS  = knapsack_threadcall.c knapsack_httslf.c
NBDSSRCS    = 

SRCS = $(COMMONSRCS) $(HTTSLFSRCS) $(NBDSSRCS)

COMMONOBJS    = $(COMMONSRCS:.c=.o)
HTTSLFOBJS    = $(HTTSLFSRCS:.c=.o)
NBDSOBJS      = $(NBDSSRCS:.c=.o)
NBDSLIBS      =  ../nbds.0.4.3/libnbds.a

OBJS = $(COMMONOBJS) $(HTTSLFOBJS) $(NBDSOBJS) $(CUDA_HOST_OBJS) $(CUDA_OBJS)

LIBS = ../utils/libbpautils_thread.a

CFLAGS += $(INCDIRS)

HOSTNAME = ${shell hostname | cut -d. -f1}
R       = R --vanilla --slave

GRAPHS =  mundara.hashcount.eps\
          mundara.eps mungera.eps mundara.instrument.eps \
          mungera_norandomization_instrument.eps mungera_norandomization.instrument.eps \
		  mundara_norandomization_instrument.eps mundara_norandomization.instrument.eps \
		  mungera.instrument.eps tango.eps tango_norandomization.eps \
			edward.eps edwardgpunode.eps \
			edward_nonrecursive.eps edwardgpunode_nonrecursive.eps \
			knapsack_teslam2070speedup.eps \
			knapsack_hashandvaluecounts.eps


RTABS = $(GRAPHS:.eps=.rtab)

BASETIMES = mundara.basetime mungera.basetime tango.basetime edward.basetime \
           edwardgpunode.basetime

all: knapsack_oahttslf knapsack_httslf knapsack_simple knapsack_threadcall \
     knapsack_diverge_oahttslf \
     knapsack_gpu_iit knapsack_gpu_oahttslf knapsack_nr_oahttslf \
		 knapsack_gpu_nr_oahttslf  knapsack_gpu_nr_httslf \
		 knapsack_randomstart_oahttslf \
		 knapsack_gpu_randomstart_nr_oahttslf knapsack_gpu_randomstart_nr_httslf \
    knapsack_gpu_randomstart_nr_profitability_oahttslf knapsack_gpu_randomstart_nr_profitability_httslf \
		 knapsack_gpu_nr_oahttslf_random123  knapsack_gpu_nr_httslf_random123 


times: $(RTABS)
graphs: $(GRAPHS)
basetimes: $(BASETIMES)

knapsack_simple: knapsack_simple.o ../utils/libbpautils_nothread.a
	$(LD) -o $@ $^  $(LDFLAGS) $(LDLIBPATH) 

knapsack_httslf: $(COMMONOBJS) knapsack_httslf.o
	$(LD) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(PTHREAD_LDFLAGS)

knapsack_oahttslf: $(COMMONOBJS) knapsack_oahttslf.o ../utils/oahttslf.o
	$(LD) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(PTHREAD_LDFLAGS)

knapsack_threadcall: $(COMMONOBJS) knapsack_threadcall.o
	$(LD) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(PTHREAD_LDFLAGS)

knapsack_simple.o: knapsack_simple.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCS) -c -o $@ $<

knapsack_diverge_oahttslf: $(COMMONOBJS) knapsack_diverge_oahttslf.o ../utils/oahttslf.o
	$(LD) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(PTHREAD_LDFLAGS)

knapsack_randomstart_oahttslf: $(COMMONOBJS) knapsack_randomstart_oahttslf.o ../utils/oahttslf.o
	$(LD) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(PTHREAD_LDFLAGS)

knapsack_gpu_iit: knapsack_gpu_iit.o 
	$(NVCC) -o $@ $^ -arch $(GPU_ARCH) $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

knapsack_gpu_iit.o : knapsack_gpu_iit.cu
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -arch $(GPU_ARCH) -o $@ $<

knapsack_gpu_oahttslf: knapsack_gpu_oahttslf.o
	$(NVCC) -o $@ $^ -arch $(GPU_ARCH) $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

knapsack_gpu_oahttslf.o : knapsack_gpu_oahttslf.cu   ../utils/oahttslf_gpu_kernel.cu
	# MUST have USE_GOOD_HASH for dp with open addressing - incredibly slow otherwise
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -DUSE_GOOD_HASH -arch $(GPU_ARCH) -o $@ $<

knapsack_nr_oahttslf: $(COMMONOBJS) knapsack_nr_oahttslf.o ../utils/oahttslf.o
	$(LD) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(PTHREAD_LDFLAGS)

knapsack_gpu_nr_oahttslf: knapsack_gpu_nr_oahttslf.o
	$(NVCC) -o $@ $^ -arch $(GPU_ARCH) $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

knapsack_gpu_nr_oahttslf.o : knapsack_gpu_nr_oahttslf.cu   ../utils/oahttslf_gpu_kernel.cu
	# MUST have USE_GOOD_HASH for dp with open addressing - incredibly slow otherwise
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -DUSE_GOOD_HASH -arch $(GPU_ARCH) -o $@ $<

knapsack_gpu_nr_oahttslf_random123: knapsack_gpu_nr_oahttslf_random123.o
	$(NVCC) -o $@ $^ -arch $(GPU_ARCH) $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

knapsack_gpu_nr_oahttslf_random123.o : knapsack_gpu_nr_oahttslf.cu   ../utils/oahttslf_gpu_kernel.cu
	# MUST have USE_GOOD_HASH for dp with open addressing - incredibly slow otherwise
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -DUSE_RANDOM123 -DUSE_GOOD_HASH -arch $(GPU_ARCH) -o $@ $<

knapsack_gpu_nr_httslf: knapsack_gpu_nr_httslf.o
	$(NVCC) -o $@ $^ -arch $(GPU_ARCH) $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

knapsack_gpu_nr_httslf.o : knapsack_gpu_nr_oahttslf.cu   ../utils/httslf_gpu_kernel.cu
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -DUSE_HTTSLF -DUSE_GOOD_HASH -arch $(GPU_ARCH) -o $@ $<

knapsack_gpu_nr_httslf_random123: knapsack_gpu_nr_httslf_random123.o
	$(NVCC) -o $@ $^ -arch $(GPU_ARCH) $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

knapsack_gpu_nr_httslf_random123.o : knapsack_gpu_nr_oahttslf.cu   ../utils/httslf_gpu_kernel.cu
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -DUSE_RANDOM123 -DUSE_HTTSLF -DUSE_GOOD_HASH -arch $(GPU_ARCH) -o $@ $<

knapsack_gpu_randomstart_nr_oahttslf: knapsack_gpu_randomstart_nr_oahttslf.o
	$(NVCC) -o $@ $^ -arch $(GPU_ARCH) $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

knapsack_gpu_randomstart_nr_oahttslf.o : knapsack_gpu_randomstart_nr_oahttslf.cu   ../utils/oahttslf_gpu_kernel.cu
	# MUST have USE_GOOD_HASH for dp with open addressing - incredibly slow otherwise
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -DUSE_GOOD_HASH -arch $(GPU_ARCH) -o $@ $<

knapsack_gpu_randomstart_nr_httslf: knapsack_gpu_randomstart_nr_httslf.o
	$(NVCC) -o $@ $^ -arch $(GPU_ARCH) $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

knapsack_gpu_randomstart_nr_httslf.o : knapsack_gpu_randomstart_nr_oahttslf.cu   ../utils/httslf_gpu_kernel.cu
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -DUSE_HTTSLF -DUSE_GOOD_HASH -arch $(GPU_ARCH) -o $@ $<




knapsack_gpu_randomstart_nr_profitability_oahttslf: knapsack_gpu_randomstart_nr_profitability_oahttslf.o
	$(NVCC) -o $@ $^ -arch $(GPU_ARCH) $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

knapsack_gpu_randomstart_nr_profitability_oahttslf.o : knapsack_gpu_randomstart_nr_profitability.cu   ../utils/oahttslf_gpu_kernel.cu
	# MUST have USE_GOOD_HASH for dp with open addressing - incredibly slow otherwise
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -DUSE_GOOD_HASH -arch $(GPU_ARCH) -o $@ $<

knapsack_gpu_randomstart_nr_profitability_httslf: knapsack_gpu_randomstart_nr_profitability_httslf.o
	$(NVCC) -o $@ $^ -arch $(GPU_ARCH) $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

knapsack_gpu_randomstart_nr_profitability_httslf.o : knapsack_gpu_randomstart_nr_profitability.cu   ../utils/httslf_gpu_kernel.cu
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -DUSE_HTTSLF -DUSE_GOOD_HASH -arch $(GPU_ARCH) -o $@ $<




gen2: gen2.c
	$(CC) -O -o gen2 gen2.c -lm


runall: runall.sh ../utils/numcores
	runall.sh  "" $(HOSTNAME)_output `../utils/numcores`


%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCS) $(PTHREAD_CFLAGS) -c -o $@ $<

depend: $(SRCS) $(HDRS)
	$(MAKEDEPEND) $(INCDIRS) $(NBDSINC) $(SRCS) $(HDRS) > $(DEPENDFILE)


knapsack_hashandvaluecounts.eps: plothashcountsvalues.r edwardgpunode.instrument.rtab teslam2070_norandomstart.instrument.rtab edwardgpunode_randomstart.instrument.rtab teslam2070.instrument.rtab
	$(R) -f plothashcountsvalues.r

knapsack_teslam2070speedup.eps: plotgpuspeedup.r teslam2070.rtab teslam2070_httslf.rtab teslam2070_norandomstart.rtab
	$(R) -f plotgpuspeedup.r

%.instrument.eps: %.instrument.rtab
	@echo "$< used in doc only, not here"

%.hashcount.eps: %.instrument.rtab
	$(R) -f plothashcount.r --args $<

%.instrument_retry.rtab: summarize_instrument_retry.sh
	( cd $*_instrument_retry_output >/dev/null ; ../summarize_instrument_retry.sh ) > $@

%.instrument.rtab: summarize_instrument.sh
	( cd $*_instrument_output >/dev/null ; ../summarize_instrument.sh ) > $@

%-tabtype2.eps: %.rtab %.httslf.rtab plotspeeduptabtype2.r
	$(R) -f plotspeeduptabtype2.r --args $< 

%.eps: %.rtab ../mosp/plotspeedup.r 
	$(R) -f ../mosp/plotspeedup.r --args $<


%.timeserrtab: %.rtab
	$(R) -f ../mosp/mktimeserrtab.r --args $< > $@


%.diverge.rtab: summarize.sh mkrtab.sh
	( cd $*_diverge_output ; ../summarize.sh ) | mkrtab.sh > $@

%.httslf.rtab: summarize_httslf.sh mkrtab.sh
	( cd $*_httslf_output ; ../summarize_httslf.sh ) | mkrtab.sh > $@

%.rtab: summarize.sh mkrtab.sh
	( cd $*_output ; ../summarize.sh ) | mkrtab.sh > $@

%.basetime: %_output
	grep '^[0-9]' $*_output/instances_simple.out | awk 'BEGIN {min=9999;max=0;} {sum += $$5 ; if ($$5 < min) min = $$5 ; if ($$5 > max) max = $$5;} END {printf("total %.0f\nmin  %.0f\nmax  %.0f\n", sum/1000,min/1000,max/1000)}' > $@


# make the TAGS file for EMACS
TAGS:
	$(MAKETAGS) $(SRCS) $(HDRS)


clean:
	$(RM) $(OBJS) knapsack_simple.o knapsack_oahttslf.o knapsack_threadcall.o
	$(RM) knapsack_httslf knapsack_simple knapsack_oahttslf knapsack_threadcall
	$(RM) knapsack_diverge_oahttslf knapsack_diverge_oahttslf.o
	$(RM) knapsack_randomstart_httslf.o
	$(RM) knapsack_nr_oahttslf knapsack_gpu_randomstart_nr_httslf knapsack_gpu_randomstart_nr_oahttslf knapsack_gpu_nr_httslf
	$(RM) gen2
	$(RM) knapsack_gpu_iit knapsack_gpu_oahttslf
	$(RM) knapack_nr_oahttslf knapsack_nr_oahttslf.o knapsack_gpu_nr_oahttslf
	$(RM) knapsack_gpu_nr_oahttslf_random123 knapsack_gpu_nr_httslf_random123

realclean:
	$(RM) $(GRAPHS) $(RTABS) $(BASETIMES)

include $(DEPENDFILE)


