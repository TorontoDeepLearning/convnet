##################################
# Set path to dependencies.
# Path to protocol buffers, hdf5.
# OPENBLAS needs to be set only if openblas=yes.
INC=$(HOME)/local/include
LIB=$(HOME)/local/lib
LOCAL_BIN=$(HOME)/local/bin

# CUDA.
CUDA_ROOT=/pkgs_local/cuda-5.5

USE_MPI=no
USE_CUDA=yes
openmp=yes
openblas=no
USE_GEMM_KERNELS=yes
OPENBLAS_LIB=$(HOME)/OpenBLAS/lib
OPENBLAS_INC=$(HOME)/OpenBLAS/include
#####################################

CUDA_INC=$(CUDA_ROOT)/include
CUDA_LIB=$(CUDA_ROOT)/lib64
CUDAMAT_DIR=$(CURDIR)/cudamat

CXX = g++

SRC=src
APP=apps
OBJ=obj
OBJ_CPU=obj/cpu
BIN=bin
PROTO=proto
PYT=py
DEPS=deps

LIBFLAGS = -L$(LIB) -L$(CURDIR)/eigenmat
CPPFLAGS_COMMON = -I$(DEPS) -I$(INC) -I$(CURDIR)/eigenmat -I$(SRC)
CPPFLAGS = $(CPPFLAGS_COMMON)
CPPFLAGS_CPU = $(CPPFLAGS_COMMON) -DUSE_GEMM
LINKFLAGS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lhdf5 -leigenmat -ljpeg -lX11 -lpthread -lprotobuf -ldl
CXXFLAGS = -O2 -std=c++0x -mtune=native -Wall -Wno-unused-result -Wno-sign-compare

EDGES_SRC := $(wildcard $(SRC)/*_edge.cc)
EDGES_OBJS := $(OBJ)/optimizer.o $(OBJ)/edge.o $(OBJ)/edge_with_weight.o $(patsubst $(SRC)/%.cc, $(OBJ)/%.o, $(EDGES_SRC))
DATAHANDLER_OBJS := $(OBJ)/image_iterators.o $(OBJ)/video_iterators.o $(OBJ)/datahandler.o $(OBJ)/datawriter.o
COMMONOBJS_COMMON = $(OBJ)/convnet_config.pb.o $(OBJ)/util.o $(OBJ)/loss_functions.o $(OBJ)/layer.o $(DATAHANDLER_OBJS) $(EDGES_OBJS)
COMMONOBJS = $(COMMONOBJS_COMMON)
COMMONOBJS_CPU = $(patsubst $(OBJ)/%.o, $(OBJ_CPU)/%.o, $(COMMONOBJS_COMMON))
COMMONOBJS += $(OBJ)/matrix.o
COMMONOBJS_CPU += $(OBJ_CPU)/CPUMatrix.o
TARGETS := $(BIN)/image2hdf5 $(BIN)/video2hdf5 $(BIN)/extract_representation_cpu $(BIN)/train_convnet_cpu

ifeq ($(USE_MPI), yes)
	CPPFLAGS_COMMON += -DUSE_MPI
	CXX = mpic++.mpich2
endif

ifeq ($(openblas), yes)
	CPPFLAGS_COMMON += -DUSE_OPENBLAS -I$(OPENBLAS_INC)
	LIBFLAGS += -L$(OPENBLAS_LIB)
	LINKFLAGS += -lopenblas
endif

ifeq ($(openmp), yes)
	CPPFLAGS_COMMON += -DUSE_OPENMP
	LINKFLAGS += -lgomp
	CXXFLAGS += -fopenmp
endif

ifeq ($(USE_CUDA), yes)

ifeq ($(USE_GEMM_KERNELS), yes)
	CPPFLAGS += -DUSE_GEMM
	LINKFLAGS += -lcudamat_conv_gemm
else
	LINKFLAGS += -lcudamat_conv
endif

	LIBFLAGS += -L$(CUDA_LIB) -L$(CUDAMAT_DIR)
	CPPFLAGS += -I$(CUDA_INC) -DUSE_CUDA
	LINKFLAGS += -lcublas -lcudamat -lcudart -Wl,-rpath=$(CUDAMAT_DIR) -Wl,-rpath=$(LIB) -Wl,-rpath=$(CUDA_LIB)
	TARGETS += $(BIN)/train_convnet $(BIN)/train_convnet_data_parallel $(BIN)/run_grad_check $(BIN)/extract_representation $(BIN)/compute_mean $(BIN)/test_data_handler
endif

all : $(TARGETS)

$(BIN)/train_convnet_data_parallel: $(COMMONOBJS) $(OBJ)/convnet.o $(OBJ)/train_convnet_data_parallel.o 
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(BIN)/train_convnet: $(COMMONOBJS) $(OBJ)/convnet.o $(OBJ)/train_convnet.o $(OBJ)/multigpu_convnet.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(BIN)/compute_mean: $(COMMONOBJS) $(OBJ)/compute_mean.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(BIN)/test_data_handler: $(COMMONOBJS) $(OBJ)/test_data_handler.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(BIN)/extract_representation: $(COMMONOBJS) $(OBJ)/convnet.o $(OBJ)/multigpu_convnet.o $(OBJ)/extract_representation.o 
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(BIN)/run_grad_check: $(COMMONOBJS) $(OBJ)/convnet.o $(OBJ)/grad_check.o $(OBJ)/run_grad_check.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS) $^ -o $@ $(LINKFLAGS)

$(BIN)/train_convnet_cpu: $(COMMONOBJS_CPU) $(OBJ_CPU)/convnet.o $(OBJ_CPU)/train_convnet.o $(OBJ_CPU)/multigpu_convnet.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS_CPU) $^ -o $@ $(LINKFLAGS)

$(BIN)/image2hdf5: $(COMMONOBJS_CPU) $(OBJ_CPU)/image_iterators.o $(OBJ_CPU)/image2hdf5.o $(OBJ_CPU)/util.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS_CPU) $^ -o $@ $(LINKFLAGS)

$(BIN)/video2hdf5: $(COMMONOBJS_CPU) $(OBJ_CPU)/video_iterators.o $(OBJ_CPU)/video2hdf5.o $(OBJ_CPU)/util.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS_CPU) $^ -o $@ $(LINKFLAGS)

$(BIN)/extract_representation_cpu: $(COMMONOBJS_CPU) $(OBJ_CPU)/convnet.o $(OBJ_CPU)/multigpu_convnet.o $(OBJ_CPU)/extract_representation.o
	$(CXX) $(LIBFLAGS) $(CPPFLAGS_CPU) $^ -o $@ $(LINKFLAGS)

$(OBJ_CPU)/%.o: $(SRC)/%.cc
	$(CXX) -c $(CPPFLAGS_CPU) $(CXXFLAGS) $< -o $@

$(OBJ_CPU)/%.o: $(APP)/%.cc
	$(CXX) -c $(CPPFLAGS_CPU) $(CXXFLAGS) $< -o $@

$(OBJ)/%.o: $(SRC)/%.cc
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(OBJ)/%.o: $(APP)/%.cc
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(OBJ)/convnet_config.pb.o : $(PROTO)/convnet_config.proto
	$(LOCAL_BIN)/protoc -I=$(PROTO) --cpp_out=$(SRC) --python_out=$(PYT) $(PROTO)/convnet_config.proto
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(SRC)/convnet_config.pb.cc -o $@

$(OBJ_CPU)/convnet_config.pb.o : $(PROTO)/convnet_config.proto
	$(LOCAL_BIN)/protoc -I=$(PROTO) --cpp_out=$(SRC) --python_out=$(PYT) $(PROTO)/convnet_config.proto
	$(CXX) -c $(CPPFLAGS_CPU) $(CXXFLAGS) $(SRC)/convnet_config.pb.cc -o $@

clean:
	rm -rf $(OBJ)/*.o $(OBJ_CPU)/*.o $(TARGETS) $(SRC)/convnet_config.pb.* $(PYT)/convnet_config_pb2.py
