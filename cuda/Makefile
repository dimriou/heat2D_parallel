CUDA_INSTALL_PATH = /opt/cuda
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common/inc/
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart
CLEAN_FILES = cuda cuda_heat2D.o cuda_heat2D_gpu.o

cuda: cuda_heat2D.o cuda_heat2D_gpu.o
	${CC} ${LFLAGS} -o $@ $^
	cp $@ ../release

cuda_heat2D.o: cuda_heat2D.c
	${CC} -c ${FLAGS} -o $@ $^

cuda_heat2D_gpu.o: cuda_heat2D_gpu.cu
	${NVCC} ${NVFLAGS} -DUNIX -c $^ -o $@

clean:
	\rm -f $(CLEAN_FILES)
	\rm -f ../release/simple_add

rebuild: clean simple_add

