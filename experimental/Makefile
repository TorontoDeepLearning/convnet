
CIMG = $(HOME)/CImg
INC=$(HOME)/local/include
LIB=$(HOME)/local/lib
CPPFLAGS = -I$(CIMG) -I$(INC)
LIBFLAGS = -L$(LIB)
LINKFLAGS = -ljpeg -lX11 -lpthread -lprotobuf -lhdf5 -lgomp
CXXFLAGS = -O2 -std=c++0x -mtune=native -Wall -Wno-unused-result -Wno-sign-compare -fopenmp

all : test_cpuconv extract_features

test_cpuconv: cpuconv.o test_cpuconv.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

extract_features: convnet_config.pb.o cpuconv.o convnet_cpu.o extract_features.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

%.o: %.cc %.h
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

%.o: %.cc
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

convnet_config.pb.o : convnet_config.proto
	protoc -I=./ --cpp_out=./ convnet_config.proto
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) convnet_config.pb.cc -o $@

clean:
	rm -rf *.o test_cpuconv extract_features
