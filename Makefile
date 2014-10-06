##################################
# Set path to dependencies.
# Path to protocol buffers, hdf5.
INC=$(HOME)/local/include
LIB=$(HOME)/local/lib
LOCAL_BIN = $(HOME)/local/bin

# CUDA.
CUDA_INC=/pkgs_local/cuda-5.5/include
CUDA_LIB=/pkgs_local/cuda-5.5/lib64
USE_MPI=yes
#####################################

CXX = g++
LIBFLAGS = -L$(LIB) -L$(CUDA_LIB) -L/u/nitish/convnet/cudamat
CPPFLAGS = -I$(INC) -I$(CUDA_INC) -I$(SRC) -Ideps
LINKFLAGS = -lhdf5 -ljpeg -lX11 -lpthread -lprotobuf -lcublas -ldl -lgomp -lcudamat -lcudamat_conv -lcudart -Wl,-rpath='$$ORIGIN/../cudamat'
CXXFLAGS = -O2 -std=c++0x -mtune=native -Wall -Wno-unused-result -Wno-sign-compare -fopenmp

ifeq ($(USE_MPI), yes)
	CPPFLAGS += -DUSE_MPI
	CXX = mpic++.mpich2
endif

SRC=src
OBJ=obj
BIN=bin
EDGES_SRC := $(wildcard $(SRC)/*_edge.cc)
EDGES_OBJS := $(OBJ)/edge.o $(OBJ)/edge_with_weight.o $(patsubst $(SRC)/%.cc, $(OBJ)/%.o, $(EDGES_SRC)) $(OBJ)/optimizer.o
DATAHANDLER_SRC := $(SRC)/image_iterators.cc $(SRC)/datahandler.cc $(SRC)/datawriter.cc
DATAHANDLER_OBJS := $(OBJ)/image_iterators.o $(OBJ)/datahandler.o $(OBJ)/datawriter.o
COMMONOBJS = $(OBJ)/convnet_config.pb.o $(OBJ)/matrix.o $(DATAHANDLER_OBJS) $(OBJ)/layer.o $(OBJ)/util.o $(EDGES_OBJS)
TARGETS := $(BIN)/train_convnet $(BIN)/train_convnet_data_parallel $(BIN)/run_grad_check $(BIN)/extract_representation $(BIN)/jpeg2hdf5 $(BIN)/test_mpi

all : $(OBJ)/convnet_config.pb.o $(TARGETS)

$(BIN)/test_mpi: $(COMMONOBJS) $(OBJ)/convnet.o $(OBJ)/test_mpi.o
	mpic++.mpich2 $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(OBJ)/test_mpi.o: $(SRC)/test_mpi.cc
	mpic++.mpich2 -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(BIN)/train_convnet_data_parallel: $(COMMONOBJS) $(OBJ)/convnet.o  $(OBJ)/train_convnet_data_parallel.o 
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(BIN)/train_convnet: $(COMMONOBJS) $(OBJ)/convnet.o  $(OBJ)/train_convnet.o $(OBJ)/multigpu_convnet.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(BIN)/extract_representation: $(COMMONOBJS) $(OBJ)/convnet.o $(OBJ)/multigpu_convnet.o $(OBJ)/extract_representation.o 
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(BIN)/run_grad_check: $(COMMONOBJS) $(OBJ)/convnet.o $(OBJ)/grad_check.o $(OBJ)/run_grad_check.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(BIN)/jpeg2hdf5: $(OBJ)/image_iterators.o $(OBJ)/jpeg2hdf5.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(OBJ)/%.o: $(SRC)/%.cc $(SRC)/%.h
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(OBJ)/%.o: $(SRC)/%.cc
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(OBJ)/convnet_config.pb.o : proto/convnet_config.proto
	$(LOCAL_BIN)/protoc -I=proto --cpp_out=$(SRC) --python_out=$(SRC) proto/convnet_config.proto
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(SRC)/convnet_config.pb.cc -o $@

clean:
	rm -rf $(OBJ)/*.o $(TARGETS) $(SRC)/convnet_config.pb.*
