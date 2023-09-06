# kmeans algorithm makefile
COPTS = -g -Wall -O3 -std=c99 
GCC = gcc $(COPTS)
MPICC = mpicc $(COPTS)
OMPCC = $(GCC) -fopenmp
NVCC = nvcc -g -O3

SHELL  = /bin/bash
CWD    = $(shell pwd | sed 's/.*\///g')

PROGRAMS = \
	kmeans_serial \
	kmeans_mpi \
	kmeans_omp \
	kmeans_cuda \

all : $(PROGRAMS)

############################################################
SHELL  = /bin/bash
CWD    = $(shell pwd | sed 's/.*\///g')

clean:
	rm -f $(PROGRAMS) *.o

################################################################################
# kmeans_serial
kmeans_serial : kmeans_serial.c
	$(GCC) -o $@ $^ -lm

################################################################################
# kmeans_mpi
kmeans_mpi : kmeans_mpi.c
	$(MPICC) -o $@ $^ -lm

################################################################################
# kmeans_omp
kmeans_omp : kmeans_omp.c
	$(OMPCC) -o $@ $^ -lm

################################################################################
# kmeans_cuda
kmeans_cuda : kmeans_cuda.cu kmeans_util.cu
	$(NVCC) -o $@ $^ -lm
