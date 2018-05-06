###############################################################################
#
# File      : Makefile
# Author    : Alex Stivala (astivala)
# Created   : June 2007
#
# $Id: Makefile 4556 2013-01-13 02:20:29Z astivala $
#
# Makefile for building the the C utils library containing lock-free hash
# table etc.
#
#
# The default target "all" will compile and build the library.
# The "test" target will build test harness executables.
# The "runtests" target will run the test harness executables
# The "clean" target will remove object and executable files and libraries.
# The "times" target will make table of times for tests
# The "graphs" target will make speedup graph from times, using R.
# The "tables" target will make speedup tables from times, usinr R.
#
# set MODE=DEBUG to build with debugging and profiling on. Otherwise
# default is to build with optimizations on and no debug or profile.
#
# This Makefile uses implicit rules to compile .c files into .o files
# and some features specific to GNU Make.
#
#
# In order to build the test that use the Intel Threading Building Blocks
# for Open Source,
# these need to first be installed, download from
#   http://www.threadingbuildingblocks.org/
# then set env var TBB21_INSTALL_DIR to the install directory and
# source the tbbvars.sh (or tbbvars.csh) script in the appropriate 
# architecture/OS subdirctory to setup environment variables for building
# with the Intel Threading Building Blocks library.
# Developed with version 2.1 (20080605 Open Source Linux version).
# 
###############################################################################


# include the common.mk file from parent diretory, and override it with
# local.mk if it exists

include ../common.mk
-include ../local.mk


CPPFLAGS += -DALLOW_DELETE -DUSE_GOOD_HASH -DUSE_THREAD_CP_ALLOC
CXXFLAGS += -DALLOW_DELETE -DUSE_GOOD_HASH -DUSE_THREAD_CP_ALLOC
NVCFLAGS += -DALLOW_DELETE -DUSE_GOOD_HASH
IIT_PARAMETERS = -DNUM_ITEMS=100000 -DMAX_NUM_THREADS=32 -DFACTOR=1 -DALLOW_DELETE

INCDIRS =  


CUDA_CPPFLAGS += $(INCDIRS)


# NVIDIA CUDA compute capability - we require 2.x for atomic CAS 64 bits,
# and recurisve device functions
GPU_ARCH = sm_20


CUDA_KERNELS  = oahttslf_gpu_kernel.cu httslf_gpu_kernel.cu 
CUDA_HOSTCODE = oahttslf_gpu_test.cu iit_oahttslf_gpu_test.cu lockfreeht_gpu_test.cu httslf_gpu_test.cu iit_httslf_gpu_test.cu verify_httslf_gpu.cu
CUDA_OBJS = 
CUDA_HOST_OBJS= $(CUDA_HOSTCODE:.cu=.o)


LIB_THREAD_SRCS = bpautils.c httslf.c cellpool.c threadcellpool.c
LIB_NOTHREAD_SRCS = bpautils.c ht.c cellpool.c 

TEST_SRCS =  httest.c httslftest.c oahttslftest.c
OTHER_SRCS = oahttslf.c ht.c
SRCS = $(LIB_THREAD_SRCS) $(LIB_NOTHREAD_SRCS) $(TEST_SRCS) $(OTHER_SRCS)

LIB_THREAD_OBJS  = $(LIB_THREAD_SRCS:.c=.o)
LIB_NOTHREAD_OBJS = $(LIB_NOTHREAD_SRCS:.c=.nothread.o)

TEST_OBJS = $(TEST_SRCS:.c=.o)
TEST_OBJS +=  iit_oahttslf_test.o   iit_httslf_test.o cpphttslf.o
OTHER_OBJS = $(OTHER_SRCS:.c=.o)
OBJS = $(LIB_NOTHREAD_OBJS) $(LIB_THREAD_OBJS) $(TEST_OBJS) $(OTHER_OBJS) \
       $(CUDA_HOST_OBJS) $(CUDA_OBJS)


CFLAGS += $(INCDIRS) 

TEST_EXES = httest httslftest oahttslftest iit_oahttslf_test iit_httslf_test \
            oahttslf_gpu_test iit_oahttslf_gpu_test lockfreeht_gpu_test \
           httslf_gpu_test iit_httslf_gpu_test verify_httslf_gpu

LIBS = libbpautils_thread.a libbpautils_nothread.a

HOSTNAME = ${shell hostname | cut -d. -f1}

GRAPHS = charikar.eps mundara.eps mungera.eps node00.eps \
         tbbhashmap.charikar.eps tbbhashmap.node00.eps \
		 charikar.testspeedupmulti.eps node00.testspeedupmulti.eps \
         mundara.testspeedupmulti2.eps \
         mungera.testspeedupmulti2.eps \
         tango.eps 

R       = R --vanilla --slave


all: libbpautils_thread.a libbpautils_nothread.a tests gprof-helper.so \
     numcores timeguard

oahttslf_gpu_test: oahttslf_gpu_test.o 
	$(NVCC) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

oahttslf_gpu_test.o : oahttslf_gpu_test.cu oahttslf_gpu_kernel.cu
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -arch $(GPU_ARCH) -o $@ $<

iit_oahttslf_gpu_test: iit_oahttslf_gpu_test.o 
	$(NVCC) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

iit_oahttslf_gpu_test.o : iit_oahttslf_gpu_test.cu oahttslf_gpu_kernel.cu
	$(NVCC) -c $(NVCFLAGS) $(IIT_PARAMETERS) $(CUDA_CPPFLAGS) -arch $(GPU_ARCH) -o $@ $<

lockfreeht_gpu_test: lockfreeht_gpu_test.o 
	$(NVCC) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

lockfreeht_gpu_test.o : lockfreeht_gpu_test.cu $(IIT_INCDIR)/LockFreeHashTableValues_kernel.cu
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -arch $(GPU_ARCH) -o $@ $<

httslf_gpu_test: httslf_gpu_test.o 
	$(NVCC) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

httslf_gpu_test.o : httslf_gpu_test.cu httslf_gpu_kernel.cu 
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -arch $(GPU_ARCH) -o $@ $<

iit_httslf_gpu_test: iit_httslf_gpu_test.o 
	$(NVCC) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

iit_httslf_gpu_test.o : iit_httslf_gpu_test.cu httslf_gpu_kernel.cu
	$(NVCC) -c $(NVCFLAGS) $(IIT_PARAMETERS) $(CUDA_CPPFLAGS) -arch $(GPU_ARCH) -o $@ $<

verify_httslf_gpu: verify_httslf_gpu.o 
	$(NVCC) -o $@ $^ $(LIBS) $(LDFLAGS) $(LDLIBPATH) $(CUDA_LDFLAGS)

verify_httslf_gpu.o : verify_httslf_gpu.cu httslf_gpu_kernel.cu 
	$(NVCC) -c $(NVCFLAGS) $(CUDA_CPPFLAGS) -arch $(GPU_ARCH) -o $@ $<

tests: $(TEST_EXES)

runtests: tests
	for i in $(TEST_EXES) ; do  \
          $$i ; \
	  if [ $$? -ne 0  ]; then  \
	    echo $$i  FAILED ; \
          else  \
            echo $$i  PASSED ; \
          fi ; \
        done 

times: $(HOSTNAME).rtab $(HOSTNAME).oahttslftest.rtab

graphs: $(GRAPHS) oahttslftest.edward.eps \
		edward.testtimesmulticpugpu.eps edward.testspeedupcpugpu.eps  \
		iit_graphs

tables: iitspeeduptable.tex

libbpautils_thread.a: $(LIB_THREAD_OBJS)
	$(AR) rv $@ $^

libbpautils_nothread.a: $(LIB_NOTHREAD_OBJS)
	$(AR) rv $@ $^


tbbhashmaptest: tbbhashmaptest.o tbbhashmap.o bpautils.o
	$(CXX) $(CXXFLAGS) $(PTHREAD_CFLAGS) -o $@ $^ -ltbbmalloc -ltbb 

%.testspeedupmulti.eps: plottestspeedupmulti.r
	$(R) -f plottestspeedupmulti.r

%.testspeedupmulti2.eps: plottestspeedupmulti2.r
	$(R) -f plottestspeedupmulti2.r

iitlockfreehttest.edward.eps: iitlockfreehttest.edward.rtab
	$(R) -f plottestspeedup_baseline1thread.r --args $<

oahttslftest.edward.eps: oahttslftest.edward.rtab
	$(R) -f plottestspeedup_baseline1thread.r --args $<

iit_graphs:
	for dir in edward_keys*_output; do (cd $$dir ; make ); done

timegraphs:
	for i in  iit_cpu_oahttslf_*_*.rtab; do adds=`echo $$i | cut -d_ -f4`; deletes=`echo $$i | cut -d_ -f5  | cut -d. -f1`; $(R) -f plottesttimesmulti.r --args $$adds $$deletes; done

gpuspeedupgraphs:
	for i in  iit_cpu_oahttslf_*_*.rtab; do adds=`echo $$i | cut -d_ -f4`; deletes=`echo $$i | cut -d_ -f5  | cut -d. -f1`; $(R) -f plottestspeedup_iit_cpugpu.r --args $$adds $$deletes; done

iitcpuspeedupgraphs:
	for i in  iit_oahttslf_*_*.rtab; do adds=`echo $$i | cut -d_ -f3`; deletes=`echo $$i | cut -d_ -f4  | cut -d. -f1`; $(R) -f plotiittestsspeedup_cpuonly.r --args $$adds $$deletes; done

iitspeeduptable.tex: computestats_iit_cpugpu.r
	$(R) -f computestats_iit_cpugpu.r | fgrep -v 'WARNING: ignoring environment value of R_HOME' > $@

edward.testtimesmulticpugpu.eps: plottesttimesmulticpugpu.r gpu_lockfreehashtable.edward.rtab cpu_lockfreehashtable.edward.rtab gpu_oahttslf.edward.rtab cpu_oahttslf.edward.rtab 
	$(R) -f plottesttimesmulticpugpu.r

edward.testspeedupcpugpu.eps: plottestspeedup_cpugpu.r cpu_lockfreehashtable.edward.rtab cpu_oahttslf.edward.rtab gpu_lockfreehashtable.edward.rtab gpu_oahttslf.edward.rtab
	$(R) -f plottestspeedup_cpugpu.r


%.eps: %.rtab plottestspeedup.r 
	$(R) -f plottestspeedup.r --args $<

%.oahttslftest.rtab: #timetests.sh $(TEST_EXES) numcores
	numcores=`numcores` ; \
	timetests.sh httest oahttslftest $$numcores > $@

%.rtab: #timetests.sh $(TEST_EXES) numcores
	numcores=`numcores` ; \
	timetests.sh httest httslftest $$numcores > $@


%.nothread.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCS) -c -o $@ $<

oahttslf.o: oahttslf.c $(INLINE_ASM)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCS) $(PTHREAD_CFLAGS) $(INLINE_ASM) -c -o $@ $<




iit_oahttslf_test.o: iit_oahttslf_test.cpp
	$(CXX) $(CXXFLAGS) $(PTHREAD_CFLAGS) $(IIT_PARAMETERS) -c -o $@ $<

iit_oahttslf_test: iit_oahttslf_test.o oahttslf.o bpautils.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(PTHREAD_LDFLAGS) $(LDLIBPATH) $(LDLIBS)

iit_httslf_test.o: iit_httslf_test.cpp
	$(CXX) $(CXXFLAGS) $(PTHREAD_CFLAGS) $(IIT_PARAMETERS) -c -o $@ $<

iit_httslf_test: iit_httslf_test.o cpphttslf.o threadcellpool.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(PTHREAD_LDFLAGS) $(LDLIBPATH) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(PTHREAD_CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCS) $(PTHREAD_CFLAGS) -c -o $@ $<

httest: httest.o libbpautils_nothread.a
	$(LD) -o $@ $^ libbpautils_nothread.a $(LDFLAGS) $(LDLIBPATH) $(LDLIBS)

httslftest: httslftest.o libbpautils_thread.a
	$(LD) -o $@ $^  libbpautils_thread.a $(LDFLAGS) $(PTHREAD_LDFLAGS) $(LDLIBPATH) $(LDLIBS)

oahttslftest: oahttslftest.o oahttslf.o bpautils.o
	$(LD) -o $@ $^ $(LDFLAGS) $(PTHREAD_LDFLAGS) $(LDLIBPATH) $(LDLIBS)

nomemorytest: nomemorytest.o oahttslf.o bpautils.o
	$(LD) -o $@ $^ $(LDFLAGS) $(PTHREAD_LDFLAGS) $(LDLIBPATH) $(LDLIBS)

simpletest: simpletest.o bpautils.o
	$(LD) -o $@ $^ $(LDFLAGS) $(PTHREAD_LDFLAGS) $(LDLIBPATH) $(LDLIBS)

numcores: numcores.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $<

timeguard: timeguard.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $<

gprof-helper.so: gprof-helper.c
	gcc -shared -fPIC $(LDFLAGS) gprof-helper.c -o gprof-helper.so -lpthread -ldl

depend: $(SRCS) $(HDRS)
	$(MAKEDEPEND) $(INCDIRS) $(NBDSINC) $(SRCS) $(HDRS) > $(DEPENDFILE)


# make the TAGS file for EMACS
tags: TAGS
TAGS:
	$(MAKETAGS) $(SRCS) $(HDRS)


clean:
	$(RM) $(OBJS)
	$(RM) $(LIBS) $(TEST_EXES)
	$(RM) gprof-helper.so numcores timeguard
	$(RM) tbbhashmaptest.o tbbhashmap.o tbbhashmaptest

realclean: clean
	$(RM) $(HOSTNAME).rtab $(HOSTNAME).eps

include $(DEPENDFILE)


