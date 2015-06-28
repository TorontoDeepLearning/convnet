##################################
# Set path to dependencies.
# CUDA.
CUDA_ROOT=/u/nitish/local/cuda-6.5
#CUDA_ROOT=/pkgs_local/cuda-5.5
###################################

CUDA_LIB=$(CUDA_ROOT)/lib64
CUDA_BIN=$(CUDA_ROOT)/bin

NVCC = $(CUDA_BIN)/nvcc
FLAGS = -O3 --use_fast_math -v \
		    -gencode=arch=compute_20,code=sm_20 \
		    -gencode=arch=compute_30,code=sm_30 \
		    -gencode=arch=compute_35,code=sm_35 \
				--compiler-options '-fPIC' --shared -Xlinker -rpath -Xlinker $(CUDA_LIB)

CUDA_CONV_SRC := cudamat_conv_filteracts.cu cudamat_conv_imgacts.cu cudamat_conv_weightacts.cu cudamat_conv_others.cu cudamat_conv_util.cu

#all : libcudamat.so libcudamat_conv.so libcudamat_conv_gemm.so
all : libcudamat.so libcudamat_conv_gemm.so

libcudamat.so: cudamat.cu cudamat_kernels.cu cudamat.cuh cudamat_kernels.cuh
	$(NVCC) $(FLAGS) -o $@ cudamat.cu cudamat_kernels.cu -lcublas -L$(CUDA_LIB)

libcudamat_conv.so: $(CUDA_CONV_SRC) cudamat_conv.cuh cudamat.cuh
	$(NVCC) $(FLAGS) -o $@ $(CUDA_CONV_SRC) -lcublas -L$(CUDA_LIB)

libcudamat_conv_gemm.so: cudamat_conv_gemm.cu cudamat_conv3d_gemm.cu cudamat_conv_gemm.cuh cudamat.cuh
	$(NVCC) $(FLAGS) -o $@ cudamat_conv_gemm.cu cudamat_conv3d_gemm.cu -lcublas -L$(CUDA_LIB)

clean:
	rm -rf libcudamat.so libcudamat_conv.so libcudamat_conv_gemm.so
