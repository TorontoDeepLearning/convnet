##################################
# Set path to dependencies.
INCLUDE := -I../deps/eigen

openmp=yes
openblas=no
OPENBLAS_LIB=$(HOME)/OpenBLAS/lib
OPENBLAS_INC=$(HOME)/OpenBLAS/include
###################################

CXX = g++
CXXFLAGS := -O3 -std=c++0x -fPIC -shared
LIB_NAME := libeigenmat.so

ifeq ($(openblas), yes)
	CPPFLAGS += -DUSE_OPENBLAS -I$(OPENBLAS_INC)
	LIBFLAGS += -L$(OPENBLAS_LIB)
	LINKFLAGS += -lopenblas
endif

ifeq ($(openmp), yes)
	CPPFLAGS += -DUSE_OPENMP
	CXXFLAGS += -fopenmp
	LINKFLAGS += -lgomp
endif

all: eigenmat.cc cpumat_conv.cc
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDE) -o $(LIB_NAME) $(LINKFLAGS) eigenmat.cc cpumat_conv.cc

clean:
	rm -rf $(LIB_NAME)
