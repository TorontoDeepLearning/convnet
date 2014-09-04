##################################
# Set path to dependencies.
# CUDA.
CUDA_LIB=/pkgs_local/cuda-5.5/lib64
###################################

NVCC = nvcc
FLAGS = -O3 --use_fast_math -v \
		    -gencode=arch=compute_20,code=sm_20 \
		    -gencode=arch=compute_30,code=sm_30 \
		    -gencode=arch=compute_35,code=sm_35 \
				--compiler-options '-fPIC' --shared

CUDA_CONV_SRC := $(wildcard cudamat_conv_*.cu)

all : libcudamat.so libcudamat_conv.so

libcudamat.so: cudamat.cu cudamat_kernels.cu cudamat.cuh cudamat_kernels.cuh
	$(NVCC) $(FLAGS) -o $@ cudamat.cu cudamat_kernels.cu -lcublas -L$(CUDA_LIB)

libcudamat_conv.so: $(CUDA_CONV_SRC) cudamat_conv.cuh cudamat.cuh
	$(NVCC) $(FLAGS) -o $@ $(CUDA_CONV_SRC) -lcublas -L$(CUDA_LIB)
	
clean:
	rm -rf libcudamat.so libcudamat_conv.so
