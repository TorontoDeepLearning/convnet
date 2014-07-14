##################################
# Set path to dependencies.
# Path to protocol buffers, hdf5.
# OPENBLAS needs to be set only if openblas=yes.
INC=$(HOME)/local/include
LIB=$(HOME)/local/lib

openmp=yes
openblas=no
OPENBLAS=$(HOME)/OpenBLAS
#####################################
CXX = g++

BIN=../bin
DEPS=../deps
PROTO=../proto

CPPFLAGS = -I$(DEPS) -I$(INC)
LIBFLAGS = -L$(LIB)
LINKFLAGS = -ljpeg -lX11 -lpthread -lprotobuf -lhdf5
CXXFLAGS = -O2 -std=c++0x -mtune=native -Wall -Wno-unused-result -Wno-sign-compare

ifeq ($(openblas), yes)
	CPPFLAGS += -DUSE_OPENBLAS -I$(OPENBLAS)
	LIBFLAGS += -L$(OPENBLAS)
	LINKFLAGS += -lopenblas
endif

ifeq ($(openmp), yes)
	CPPFLAGS += -DUSE_OPENMP
	LINKFLAGS += -lgomp
	CXXFLAGS += -fopenmp
endif

all : $(BIN)/extract_representation_cpu

$(BIN)/extract_representation_cpu: convnet_config.pb.o cpuconv.o convnet_cpu.o extract_representation_cpu.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

%.o: %.cc %.h
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

%.o: %.cc
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

convnet_config.pb.o : $(PROTO)/convnet_config.proto
	protoc -I=$(PROTO) --cpp_out=./ $(PROTO)/convnet_config.proto
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) convnet_config.pb.cc -o $@

clean:
	rm -rf *.o $(BIN)/extract_representation_cpu convnet_config.pb.*
