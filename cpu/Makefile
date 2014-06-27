##################################
# Set path to dependencies.
# Path to CImg.
CIMG = $(HOME)/CImg

# Path to protocol buffers, hdf5.
INC=$(HOME)/local/include
LIB=$(HOME)/local/lib
#####################################
CXX = g++

BIN=../bin

CPPFLAGS = -I$(CIMG) -I$(INC)
LIBFLAGS = -L$(LIB)
LINKFLAGS = -ljpeg -lX11 -lpthread -lprotobuf -lhdf5 -lgomp
CXXFLAGS = -O2 -std=c++0x -mtune=native -Wall -Wno-unused-result -Wno-sign-compare -fopenmp

all : $(BIN)/extract_representation_cpu

$(BIN)/extract_representation_cpu: convnet_config.pb.o cpuconv.o convnet_cpu.o extract_representation_cpu.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

%.o: %.cc %.h
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

%.o: %.cc
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

convnet_config.pb.o : convnet_config.proto
	protoc -I=./ --cpp_out=./ convnet_config.proto
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) convnet_config.pb.cc -o $@

clean:
	rm -rf *.o $(BIN)/extract_representation_cpu convnet_config.pb.*
