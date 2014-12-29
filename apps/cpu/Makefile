##################################
# Set path to dependencies.
# Path to protocol buffers, hdf5.
# OPENBLAS needs to be set only if openblas=yes.
INC=$(HOME)/local/include
LIB=$(HOME)/local/lib
LOCAL_BIN=$(HOME)/local/bin

USE_MPI=no
openmp=yes
openblas=no
OPENBLAS_LIB=$(HOME)/OpenBLAS/lib
OPENBLAS_INC=$(HOME)/OpenBLAS/include
#####################################

CXX = g++

ROOT=../../
SRC=$(ROOT)/src
APP=$(ROOT)/apps
OBJ=$(ROOT)/obj
BIN=$(ROOT)/bin
PROTO=$(ROOT)/proto
PYT=$(ROOT)/py
DEPS=$(ROOT)/deps

LIBFLAGS = -L$(LIB) -L$(ROOT)/eigenmat
CPPFLAGS = -I$(DEPS) -I$(INC) -I$(ROOT)/eigenmat -I$(SRC)
LINKFLAGS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lhdf5 -leigenmat -ljpeg -lX11 -lpthread -lprotobuf
CXXFLAGS = -O2 -std=c++0x -mtune=native -Wall -Wno-unused-result -Wno-sign-compare

ifeq ($(USE_MPI), yes)
	CPPFLAGS += -DUSE_MPI
	CXX = mpic++.mpich2
endif

ifeq ($(openblas), yes)
	CPPFLAGS += -DUSE_OPENBLAS -I$(OPENBLAS_INC)
	LIBFLAGS += -L$(OPENBLAS_LIB)
	LINKFLAGS += -lopenblas
endif

ifeq ($(openmp), yes)
	CPPFLAGS += -DUSE_OPENMP
	LINKFLAGS += -lgomp
	CXXFLAGS += -fopenmp
endif

EDGES_SRC := $(wildcard $(SRC)/*_edge.cc)
EDGES_OBJS := $(OBJ)/optimizer.o $(OBJ)/edge.o $(OBJ)/edge_with_weight.o $(patsubst $(SRC)/%.cc, $(OBJ)/%.o, $(EDGES_SRC))
DATAHANDLER_SRC := $(SRC)/image_iterators.cc $(SRC)/video_iterators.cc $(SRC)/datahandler.cc $(SRC)/datawriter.cc
DATAHANDLER_OBJS := $(OBJ)/image_iterators.o $(OBJ)/video_iterators.o $(OBJ)/datahandler.o $(OBJ)/datawriter.o
COMMONOBJS = $(OBJ)/convnet_config.pb.o $(OBJ)/util.o $(OBJ)/CPUMatrix.o $(OBJ)/loss_functions.o $(OBJ)/layer.o $(DATAHANDLER_OBJS) $(EDGES_OBJS)
TARGETS := $(BIN)/extract_representation_cpu

all : $(OBJ)/convnet_config.pb.o $(TARGETS)

$(BIN)/extract_representation_cpu: $(COMMONOBJS) $(OBJ)/convnet.o $(OBJ)/multigpu_convnet.o $(OBJ)/extract_representation.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(OBJ)/%.o: $(SRC)/%.cc
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(OBJ)/%.o: $(APP)/%.cc
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(OBJ)/convnet_config.pb.o : $(PROTO)/convnet_config.proto
	$(LOCAL_BIN)/protoc -I=$(PROTO) --cpp_out=$(SRC) --python_out=$(PYT) $(PROTO)/convnet_config.proto
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(SRC)/convnet_config.pb.cc -o $@

clean:
	rm -rf $(OBJ)/*.o $(TARGETS) $(SRC)/convnet_config.pb.* $(PYT)/convnet_config_pb2.py
