/** Kernels for convUp, convDown, convOutp, maxpool, avgpool, maxpoolundo,
 *  avgpoolundo.
 *  These kernels are 10-20% slower than cuda-convnet2, but have no constraints
 *  on number of channels and support rectangular images and rectangular kernels.
 *  They use cublasSgemm for convUp, convDown, convOutp.
 *  Data layout : Column-major
 *  data : (num_images, image_size_x, image_size_y, num_input_channels)
 *  filters : (num_output_channels, kernel_size_x, kernel_size_y, num_input_channels)
 */

#include "cudamat_conv_gemm.cuh"
#define getLastCudaError(msg)   __getLastCudaError (msg, __FILE__, __LINE__)
size_t free_space_ = 0;
void EstimateFreeSpace() {
  /*
  size_t total;
  cudaMemGetInfo(&free_space_, &total);
  //free_space_ >>= 1;
  if (free_space_ > 1>>20) {
    free_space_ -= 1 << 20; // Just remove 1 MB. This seems to work!
  }
  */
  // How to get free contiguous space ?
  free_space_ = MAX_MEMORY_BYTES;
}

inline bool check_cublas_error() {
  cublasStatus status = cublasGetError();
  return status != CUBLAS_STATUS_SUCCESS;
}

__device__ inline float square(float a) {
  return a * a;
}
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
 cudaError_t err = cudaGetLastError();
 if (cudaSuccess != err) {
  fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
      file, line, errorMessage, (int)err, cudaGetErrorString(err));
  exit(EXIT_FAILURE);
 }
}

void _Scale(cudamat* mat, float scale) {
  if (scale == 0) {
    cudaMemset(mat->data_device, 0, sizeof(float) * mat->size[0] * mat->size[1]);
  } else if (scale != 1) {
    cublasSscal(mat->size[0] * mat->size[1], scale, mat->data_device, 1);
  }
}

class AvgPooler {
 public:
  __device__ inline float operator()(const float a, const float b) const {
    return a + b;
  }
  __device__ inline float getBaseValue() const {
    return 0;
  }
  __device__ inline float output(const float a, const int regionSize) const {
    return a / regionSize;
  }
};

class MaxPooler {
 public:
  __device__ inline float operator()(const float a, const float b) const {
    return fmaxf(a, b);
  }
  __device__ inline float getBaseValue() const {
    return -2e38; 
  }
  __device__ inline float output(const float a, const int regionSize) const {
    return a;
  }
};

__global__ void kExpand(float *images, float* targets,
                        int num_images, int num_input_channels,
                        int image_size_y, int image_size_x,
                        int num_modules_y, int num_modules_x,
                        int kernel_size_y, int kernel_size_x,
                        int padding_y, int padding_x,
                        int stride_y, int stride_x,
                        int num_modules_batch, int module_id_offset) {
  int color = blockIdx.y;
  int src_module_id = module_id_offset + blockIdx.x;
  int dst_module_id = blockIdx.x;

  int module_id_x = src_module_id % num_modules_x;
  int module_id_y = src_module_id / num_modules_x;
  int startX = module_id_x * stride_x + padding_x;
  int startY = module_id_y * stride_y + padding_y;
  int Y, X;
  long target_id, source_id;
  images += num_images * image_size_x * image_size_y * color;
  targets += num_images * (dst_module_id + num_modules_batch * (kernel_size_y * kernel_size_x * color));
  for (int y = 0; y < kernel_size_y; y++) {
    Y = startY + y;
    for (int x = 0; x < kernel_size_x; x++) {
      X = startX + x;
      target_id = num_images * num_modules_batch * (x + kernel_size_x * y);
      source_id = num_images * (X + image_size_x * Y);
      if (X < 0 || X >= image_size_x || Y < 0 || Y >= image_size_y) {
        for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
          targets[target_id + im] = 0;
        }
      } else {
        for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
          targets[target_id + im] = images[source_id + im];
        }
      }
      __syncthreads();
    }
  }
}

template <class Pooler>
__global__ void kPool(float* images, float* targets,
                      int num_images, int num_input_channels,
                      int image_size_y, int image_size_x, int image_size_t,
                      int num_modules_y, int num_modules_x, int num_modules_t,
                      int kernel_size_y, int kernel_size_x, int kernel_size_t,
                      int padding_y, int padding_x, int padding_t,
                      int stride_y, int stride_x, int stride_t,
                      float scaleOutput, Pooler pooler) {
  const int color = blockIdx.y;
  const int num_colors = gridDim.y;
  const int num_modules = num_modules_y * num_modules_x * num_modules_t;

  long source_id, target_id;
  images  += num_images * image_size_x * image_size_y * color;
  targets += num_images * num_modules_x * num_modules_y * color;
  for (int module_id = blockIdx.x; module_id < num_modules; module_id += gridDim.x) {
    int module_id_x = module_id % num_modules_x;
    int module_id_y = (module_id / num_modules_x) % num_modules_y;
    int module_id_t = (module_id / num_modules_x) / num_modules_y;
    int startX = module_id_x * stride_x + padding_x;
    int startY = module_id_y * stride_y + padding_y;
    int startT = module_id_t * stride_t + padding_t;
    int endX = startX + kernel_size_x;
    int endY = startY + kernel_size_y;
    int endT = startT + kernel_size_t;
    target_id = num_images * (module_id_x + num_modules_x * (module_id_y + num_modules_y * num_colors * module_id_t));
    startX = MAX(startX, 0);
    startY = MAX(startY, 0);
    startT = MAX(startT, 0);
    endY   = MIN(endY  , image_size_y);
    endX   = MIN(endX  , image_size_x);
    endT   = MIN(endT  , image_size_t);
    int regionSize = (endX - startX) * (endY - startY) * (endT - startT);
    for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
      float val = pooler.getBaseValue();
      for (int T = startT; T < endT; T++) {
        for (int Y = startY; Y < endY; Y++) {
          for (int X = startX; X < endX; X++) {
            source_id = num_images * (X + image_size_x * (Y + image_size_y * num_colors * T));
            val = pooler(val, images[source_id + im]);
          }
        }
      }
      targets[target_id + im] = scaleOutput * pooler.output(val, regionSize);
    }
  }
  __syncthreads();
}


__global__ void kAvgPoolUndo(float *derivs, float* targets,
                             int num_images, int num_input_channels,
                             int image_size_y, int image_size_x, int image_size_t,
                             int num_modules_y, int num_modules_x, int num_modules_t,
                             int kernel_size_y, int kernel_size_x, int kernel_size_t,
                             int padding_y, int padding_x, int padding_t,
                             int stride_y, int stride_x, int stride_t, float scaleOutput) {
  const int color = blockIdx.y;
  const int num_colors = gridDim.y;
  const int num_modules = num_modules_y * num_modules_x * num_modules_t;

  long source_id, target_id;
  derivs  += num_images * num_modules_x * num_modules_y * color;
  targets += num_images * image_size_x * image_size_y * color;
  for (int module_id = blockIdx.x; module_id < num_modules; module_id += gridDim.x) {
    int module_id_x = module_id % num_modules_x;
    int module_id_y = (module_id / num_modules_x) % num_modules_y;
    int module_id_t = (module_id / num_modules_x) / num_modules_y;
    int startX = module_id_x * stride_x + padding_x;
    int startY = module_id_y * stride_y + padding_y;
    int startT = module_id_t * stride_t + padding_t;
    int endX = startX + kernel_size_x;
    int endY = startY + kernel_size_y;
    int endT = startT + kernel_size_t;
    source_id = num_images * (module_id_x + num_modules_x * (module_id_y + num_modules_y * num_colors * module_id_t));
    startX = MAX(startX, 0);
    startY = MAX(startY, 0);
    startT = MAX(startT, 0);
    endY   = MIN(endY  , image_size_y);
    endX   = MIN(endX  , image_size_x);
    endT   = MIN(endT  , image_size_t);
    int regionSize = (endX - startX) * (endY - startY) * (endT - startT);

    for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
      float val = scaleOutput * derivs[source_id + im] / regionSize;
      for (int T = startT; T < endT; T++) {
        for (int Y = startY; Y < endY; Y++) {
          for (int X = startX; X < endX; X++) {
            target_id = num_images * (X + image_size_x * (Y + image_size_y * num_colors * T)) + im;
            atomicAdd(&targets[target_id], val);
            __syncthreads();
          }
        }
      }
    }
  }
}

__global__ void kMaxPoolUndo(float * images, float *derivs, float* maxes, float* targets,
                        int num_images, int num_input_channels,
                        int image_size_y, int image_size_x, int image_size_t,
                        int num_modules_y, int num_modules_x, int num_modules_t,
                        int kernel_size_y, int kernel_size_x, int kernel_size_t,
                        int padding_y, int padding_x, int padding_t,
                        int stride_y, int stride_x, int stride_t, float scaleOutput) {
  const int color = blockIdx.y;
  const int num_colors = gridDim.y;
  const int num_modules = num_modules_y * num_modules_x * num_modules_t;

  long source_id, target_id;
  derivs  += num_images * num_modules_x * num_modules_y * color;
  maxes   += num_images * num_modules_x * num_modules_y * color;
  targets += num_images * image_size_x * image_size_y * color;
  images  += num_images * image_size_x * image_size_y * color;
  for (int module_id = blockIdx.x; module_id < num_modules; module_id += gridDim.x) {
    int module_id_x = module_id % num_modules_x;
    int module_id_y = (module_id / num_modules_x) % num_modules_y;
    int module_id_t = (module_id / num_modules_x) / num_modules_y;
    int startX = module_id_x * stride_x + padding_x;
    int startY = module_id_y * stride_y + padding_y;
    int startT = module_id_t * stride_t + padding_t;
    int endX = startX + kernel_size_x;
    int endY = startY + kernel_size_y;
    int endT = startT + kernel_size_t;
    source_id = num_images * (module_id_x + num_modules_x * (module_id_y + num_modules_y * num_colors * module_id_t));
    startX = MAX(startX, 0);
    startY = MAX(startY, 0);
    startT = MAX(startT, 0);
    endY   = MIN(endY  , image_size_y);
    endX   = MIN(endX  , image_size_x);
    endT   = MIN(endT  , image_size_t);
    for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
      float val = scaleOutput * derivs[source_id + im];
      for (int T = startT; T < endT; T++) {
        for (int Y = startY; Y < endY; Y++) {
          for (int X = startX; X < endX; X++) {
            target_id = num_images * (X + image_size_x * (Y + image_size_y * num_colors * T)) + im;
            if (images[target_id] == maxes[source_id + im]) {
              atomicAdd(&targets[target_id], val);
            }
            __syncthreads();
          }
        }
      }
    }
  }
}

__global__ void kContract(float *expanded_data, float* targets,
                          int num_images, int num_input_channels,
                          int image_size_y, int image_size_x,
                          int num_modules_y, int num_modules_x,
                          int kernel_size_y, int kernel_size_x,
                          int padding_y, int padding_x,
                          int stride_y, int stride_x,
                          int num_modules_batch, int module_id_offset) {
  int color = blockIdx.y;
  int dst_module_id = module_id_offset + blockIdx.x;
  int src_module_id = blockIdx.x;

  int module_id_x = dst_module_id % num_modules_x;
  int module_id_y = dst_module_id / num_modules_x;
  int startX = module_id_x * stride_x + padding_x;
  int startY = module_id_y * stride_y + padding_y;
  int Y, X;
  long target_id, source_id;
  targets += num_images * image_size_x * image_size_y * color;
  expanded_data  += num_images * (src_module_id + num_modules_batch * (kernel_size_y * kernel_size_x * color));
  for (int y = 0; y < kernel_size_y; y++) {
    Y = startY + y;
    for (int x = 0; x < kernel_size_x; x++) {
      X = startX + x;
      source_id = num_images * num_modules_batch * (x + kernel_size_x * y);
      target_id = num_images * (X + image_size_x * Y);
      if (X < 0 || X >= image_size_x || Y < 0 || Y >= image_size_y) {
        // do nothing.
      } else {
        for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
          atomicAdd(&targets[target_id + im], expanded_data[source_id + im]);
          __syncthreads();
        }
      }
    }
  }
}

__global__ void kWriteRows(float* data, float* target, int num_images,
                           int num_modules, int num_modules_batch,
                           int module_id_offset, float beta) {
  int c = blockIdx.y;
  int src_module_id = blockIdx.x;
  int dst_module_id = module_id_offset + blockIdx.x;

  data += num_images * (src_module_id + c * num_modules_batch);
  target += num_images * (dst_module_id + c * num_modules);

  for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
    target[im] = beta * data[im];
  }
}

__global__ void kReadRows(float* data, float* target,
                          int num_images, int num_modules,
                          int num_modules_batch, int module_id_offset) {
  int c = blockIdx.y;
  int src_module_id = module_id_offset + blockIdx.x;
  int dst_module_id = blockIdx.x;

  data += num_images * (src_module_id + c * num_modules);
  target += num_images * (dst_module_id + c * num_modules_batch);

  for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
    target[im] = data[im];
  }
}


__global__ void kWriteRowsMult(float* data, float* target,
                               int num_images, int num_modules,
                               int num_modules_batch, int module_id_offset,
                               float alpha, float beta) {
  int c = blockIdx.y;
  int src_module_id = blockIdx.x;
  int dst_module_id = module_id_offset + blockIdx.x;

  data += num_images * (src_module_id + c * num_modules_batch);
  target += num_images * (dst_module_id + c * num_modules);

  for (int im = threadIdx.x; im < num_images; im += blockDim.x) {
    target[im] = alpha * target[im] + beta * data[im];
  }
}

__global__ void kCrossMapDenoms(float* data, float* denoms,
                                int num_locs, int batch_locs, int batch_offset,
                                float addScale, float powScale,
                                int num_filters, int k, bool blocked) {
  long loc_id = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  if (batch_offset + loc_id < num_locs) {
    data   += batch_offset + loc_id;
    denoms += loc_id;
    int prev_start = 0, prev_end = 0, start, end;
    float sum = 0;
    for (int j = 0; j < num_filters; j++) {
      start = blocked ? (j / k) * k : -k/2 + j;
      end = MIN(num_filters, start + k);
      start = MAX(0, start);
      for (int i = prev_start; i < start; i++) {
        sum -= square(data[i * num_locs]);
      }
      for (int i = prev_end; i < end; i++) {
        sum += square(data[i * num_locs]);
      }
      denoms[j * batch_locs] = __powf(1 + addScale * sum, -powScale - 1);
      prev_start = start;
      prev_end = end;
    }
  }
}

__global__ void kCrossMapRNorm(float* data, float* target,
                               int num_locs, float addScale, float powScale,
                               int num_filters, int k, bool blocked) {
  long loc_id = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  if (loc_id < num_locs) {
    data   += loc_id;
    target += loc_id;
    float sum = 0;
    int prev_start = 0, prev_end = 0, start, end;
    for (int j = 0; j < num_filters; j++) {
      start = blocked ? (j / k) * k : -k/2 + j;
      end = MIN(num_filters, start + k);
      start = MAX(0, start);
      for (int i = prev_start; i < start; i++) {
        sum -= square(data[i * num_locs]);
      }
      for (int i = prev_end; i < end; i++) {
        sum += square(data[i * num_locs]);
      }
      target[j * num_locs] = data[j * num_locs] * __powf(1 + addScale * sum, -powScale);
      prev_start = start;
      prev_end = end;
    }
  }
}

__global__ void kCrossMapRNormUndo(float* data, float* deriv, float* denoms, float* target,
                                   int num_locs, int batch_locs, int batch_offset, float addScale, float powScale,
                                   int num_filters, int k, bool blocked) {
  long loc_id = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
  if (batch_offset + loc_id < num_locs) {
    data   += batch_offset + loc_id;
    target += batch_offset + loc_id;
    deriv  += batch_offset + loc_id;
    denoms += loc_id;
    float sum = 0;
    int prev_start = 0, prev_end = 0, start, end;
    for (int j = 0; j < num_filters; j++) {
      start = blocked ? (j / k) * k : -k + k/2 + j + 1;
      end = MIN(num_filters, start + k);
      start = MAX(0, start);
      for (int i = prev_start; i < start; i++) {
        sum -= deriv[i * num_locs] * data[i * num_locs] * denoms[i * batch_locs];
      }
      for (int i = prev_end; i < end; i++) {
        sum += deriv[i * num_locs] * data[i * num_locs] * denoms[i * batch_locs];
      }
      target[j * num_locs] = deriv[j * num_locs] * __powf(denoms[j * batch_locs], powScale / (powScale + 1)) -
                             2 * addScale * powScale * data[j * num_locs] * sum;
      prev_start = start;
      prev_end = end;
    }
  }
}

void _convUpGemm(cudamat* images, cudamat* filters, cudamat* targets,
                Shape4D images_shape, Shape4D filters_shape,
                Shape4D targets_shape, ConvDesc conv_desc,
                float scaleTargets, float scaleOutput, bool conv) {
    
    int num_input_channels   = conv_desc.num_input_channels;
    int num_output_channels  = conv_desc.num_output_channels;
    int kernel_size_y        = conv_desc.kernel_size_y;
    int kernel_size_x        = conv_desc.kernel_size_x;
    int stride_y             = conv_desc.stride_y;
    int stride_x             = conv_desc.stride_x;
    int padding_y            = conv_desc.padding_y;
    int padding_x            = conv_desc.padding_x;
    int input_channel_begin  = conv_desc.input_channel_begin;
    int input_channel_end    = conv_desc.input_channel_end;
    int output_channel_begin = conv_desc.output_channel_begin;
    int output_channel_end   = conv_desc.output_channel_end;
    int num_groups           = conv_desc.num_groups;

    if (output_channel_end == 0) output_channel_end = num_output_channels;
    if (input_channel_end == 0) input_channel_end = num_input_channels;

    int num_output_channels2 = targets_shape.shape[3];
    int num_modules_y        = targets_shape.shape[2];
    int num_modules_x        = targets_shape.shape[1];
    int num_images           = targets_shape.shape[0];

    int num_input_channels2  = images_shape.shape[3];
    int image_size_y         = images_shape.shape[2];
    int image_size_x         = images_shape.shape[1];
    int num_images2          = images_shape.shape[0];

    int num_input_channels3  = filters_shape.shape[3];
    int kernel_size_y2       = filters_shape.shape[2];
    int kernel_size_x2       = filters_shape.shape[1];
    int num_output_channels3 = filters_shape.shape[0];

    int num_modules          = num_modules_y * num_modules_x;
    int filterModuleMult     = conv ? 1 : num_modules;
  
    // Consistency checks. 
    assert (num_images == num_images2);
    assert (num_output_channels == num_output_channels2);
    assert (output_channel_end - output_channel_begin == num_output_channels3);
    assert (num_input_channels == num_input_channels2);
    assert (input_channel_end - input_channel_begin == num_input_channels3 / filterModuleMult);
    assert (num_images == images->size[0]);
    assert (num_images == targets->size[0]);
    assert (num_output_channels3 == filters->size[0]);
    assert (image_size_y * image_size_x * num_input_channels == images->size[1]);
    assert (num_modules_y * num_modules_x * num_output_channels == targets->size[1]);
    assert (kernel_size_y * kernel_size_x * num_input_channels3 * filterModuleMult == filters->size[1]);
    assert (kernel_size_y == kernel_size_y2);
    assert (kernel_size_x == kernel_size_x2);
    assert (num_input_channels % num_groups == 0);
    assert (num_groups == 1);
    assert (input_channel_begin  >= 0);
    assert (output_channel_begin >= 0);
    assert (input_channel_end    <= num_input_channels);
    assert (output_channel_end   <= num_output_channels);
    assert (input_channel_begin  <= input_channel_end);
    assert (output_channel_begin <= output_channel_end);
    num_input_channels = input_channel_end - input_channel_begin;
    num_output_channels = output_channel_end - output_channel_begin;
    assert(num_input_channels  > 0);
    assert(num_output_channels > 0);
    float* w = filters->data_device;
    float* images_data = images->data_device + input_channel_begin * image_size_y * image_size_x * num_images;
    float* targets_data = targets->data_device + output_channel_begin * num_modules * num_images;

    int input_size = kernel_size_y * kernel_size_x * num_input_channels;
    int num_threads_x = MIN(num_images, NUM_THREADS_PER_BLOCK);
    
    float *expanded_images = NULL, *expanded_target = NULL;
    int num_modules_batch;
    
    int input_memory_size  = num_images * input_size * sizeof(float);
    int output_memory_size = num_images * num_output_channels * sizeof(float);
    if (free_space_ == 0) EstimateFreeSpace();
    int max_batch_size = free_space_ / (input_memory_size + output_memory_size);
    max_batch_size = MIN(max_batch_size, num_modules / filterModuleMult);
    max_batch_size = MIN(max_batch_size, MAX_BLOCKS_X);
    //max_batch_size = MAX(max_batch_size, 1);
    //printf("Free space %ld max batch size %d\n", free_space, max_batch_size);

    cudaError_t err1, err2;
    err1 = cudaMalloc((void**)&expanded_images,  max_batch_size * input_memory_size);
    err2 = cudaMalloc((void**)&expanded_target, max_batch_size * output_memory_size);
    if (cudaSuccess != err1 || cudaSuccess != err2) {
      printf("Could not allocate memory.\n");
      num_modules_batch = 1;
    } else {
      num_modules_batch = max_batch_size;
    }

    int num_iter = DIVUP(num_modules, num_modules_batch);

    int module_id_start = 0;
    for (int i = 0; i < num_iter; i++) {
      int this_num_modules_batch = MIN(num_modules_batch, num_modules - module_id_start);
      //printf("Step %d num_modules %d\n", i, this_num_modules_batch);

      dim3 threads(num_threads_x);
      dim3 blocks = dim3(this_num_modules_batch, input_channel_end - input_channel_begin);
      kExpand<<<blocks, threads>>>(images_data, expanded_images,
                                   num_images, num_input_channels,
                                   image_size_y, image_size_x,
                                   num_modules_y, num_modules_x,
                                   kernel_size_y, kernel_size_x,
                                   padding_y, padding_x,
                                   stride_y, stride_x,
                                   this_num_modules_batch, module_id_start);
      if (!conv) w += num_output_channels * input_size;
      cublasSgemm('n', 't', 
                  num_images * this_num_modules_batch, num_output_channels,
                  input_size,
                  1, expanded_images, num_images * this_num_modules_batch,
                  w, num_output_channels,
                  0, expanded_target, num_images * this_num_modules_batch);

      dim3 blocks2 = dim3(this_num_modules_batch, num_output_channels);
      if (scaleTargets == 0) {
        kWriteRows<<<blocks2, threads>>>(expanded_target, targets_data,
                                         num_images, num_modules,
                                         this_num_modules_batch, module_id_start,
                                         scaleOutput);
      } else {
        kWriteRowsMult<<<blocks2, threads>>>(expanded_target, targets_data,
                                         num_images, num_modules,
                                         this_num_modules_batch, module_id_start,
                                         scaleTargets, scaleOutput);
      }
      module_id_start += this_num_modules_batch;
    }
    cudaFree(expanded_images);
    cudaFree(expanded_target);
    getLastCudaError("convUpGemm: kernel execution failed");
}

void _convDownGemm(cudamat* derivs, cudamat* filters, cudamat* targets,
                Shape4D derivs_shape, Shape4D filters_shape,
                Shape4D targets_shape, ConvDesc conv_desc,
                float scaleTargets, float scaleOutput, bool conv) {
    
    int num_input_channels   = conv_desc.num_input_channels;
    int num_output_channels  = conv_desc.num_output_channels;
    int kernel_size_y        = conv_desc.kernel_size_y;
    int kernel_size_x        = conv_desc.kernel_size_x;
    int stride_y             = conv_desc.stride_y;
    int stride_x             = conv_desc.stride_x;
    int padding_y            = conv_desc.padding_y;
    int padding_x            = conv_desc.padding_x;
    int input_channel_begin  = conv_desc.input_channel_begin;
    int input_channel_end    = conv_desc.input_channel_end;
    int output_channel_begin = conv_desc.output_channel_begin;
    int output_channel_end   = conv_desc.output_channel_end;
    int num_groups           = conv_desc.num_groups;
    if (output_channel_end == 0) output_channel_end = num_output_channels;
    if (input_channel_end == 0) input_channel_end = num_input_channels;

    int num_output_channels2 = derivs_shape.shape[3];
    int num_modules_y        = derivs_shape.shape[2];
    int num_modules_x        = derivs_shape.shape[1];
    int num_images           = derivs_shape.shape[0];

    int num_input_channels2  = targets_shape.shape[3];
    int image_size_y         = targets_shape.shape[2];
    int image_size_x         = targets_shape.shape[1];
    int num_images2          = targets_shape.shape[0];

    int num_input_channels3  = filters_shape.shape[3];
    int kernel_size_y2       = filters_shape.shape[2];
    int kernel_size_x2       = filters_shape.shape[1];
    int num_output_channels3 = filters_shape.shape[0];

    int num_modules          = num_modules_y * num_modules_x;
    int filterModuleMult     = conv ? 1 : num_modules;
  
    // Consistency checks. 
    assert (num_images == num_images2);
    assert (num_output_channels == num_output_channels2);
    assert (output_channel_end - output_channel_begin == num_output_channels3);
    assert (num_input_channels == num_input_channels2);
    assert (input_channel_end - input_channel_begin == num_input_channels3 / filterModuleMult);
    assert (num_images2 == targets->size[0]);
    assert (num_images == derivs->size[0]);
    assert (num_output_channels3 == filters->size[0]);
    assert (image_size_y * image_size_x * num_input_channels2 == targets->size[1]);
    assert (num_modules_y * num_modules_x * num_output_channels2 == derivs->size[1]);
    assert (kernel_size_y * kernel_size_x * num_input_channels3 * filterModuleMult == filters->size[1]);
    assert (kernel_size_y == kernel_size_y2);
    assert (kernel_size_x == kernel_size_x2);
    assert (num_input_channels % num_groups == 0);
    assert (num_groups == 1);
    assert (input_channel_begin  >= 0);
    assert (output_channel_begin >= 0);
    assert (input_channel_end    <= num_input_channels);
    assert (output_channel_end   <= num_output_channels);
    assert (input_channel_begin  <= input_channel_end);
    assert (output_channel_begin <= output_channel_end);
    num_input_channels = input_channel_end - input_channel_begin;
    num_output_channels = output_channel_end - output_channel_begin;
    assert(num_input_channels  > 0);
    assert(num_output_channels > 0);
    float* w = filters->data_device;
    float* derivs_data = derivs->data_device + output_channel_begin * num_modules * num_images;
    float* targets_data = targets->data_device + input_channel_begin * image_size_y * image_size_x * num_images;
    
    int input_size = kernel_size_y * kernel_size_x * num_input_channels;
    int num_threads_x = MIN(num_images, NUM_THREADS_PER_BLOCK);
    float *expanded_target = NULL, *expanded_derivs = NULL;
    int num_modules_batch;

    int input_memory_size  = num_images * input_size * sizeof(float);
    int output_memory_size = num_images * num_output_channels * sizeof(float);
    if (free_space_ == 0) EstimateFreeSpace();
    int max_batch_size = free_space_ / (input_memory_size + output_memory_size);
    max_batch_size = MIN(max_batch_size, num_modules / filterModuleMult);
    max_batch_size = MIN(max_batch_size, MAX_BLOCKS_X);
    max_batch_size = MAX(max_batch_size, 1);

    cudaError_t err1, err2;
    err1 = cudaMalloc((void**)&expanded_target,  max_batch_size * input_memory_size);
    err2 = cudaMalloc((void**)&expanded_derivs, max_batch_size * output_memory_size);
    if (cudaSuccess != err1 || cudaSuccess != err2) {
      printf("Out of memory\n");
      /*
      if (cudaSuccess == err1) cudaFree(expanded_target);
      if (cudaSuccess == err2) cudaFree(expanded_derivs);
      err1 = cudaMalloc((void**)&expanded_target,  input_memory_size);
      err2 = cudaMalloc((void**)&expanded_derivs, output_memory_size);
      if (cudaSuccess != err1 || cudaSuccess != err2) {
        printf("Out of memory on GPU! %s \n", cudaGetErrorString(err1));
        printf("Out of memory on GPU! %s \n", cudaGetErrorString(err2));
      }
      */
      num_modules_batch = 1;
    } else {
      num_modules_batch = max_batch_size;
    }

    int num_iter = DIVUP(num_modules, num_modules_batch);
   
    _Scale(targets, scaleTargets); 

    int module_id_start = 0;
    for (int i = 0; i < num_iter; i++) {
      int this_num_modules_batch = MIN(num_modules_batch, num_modules - module_id_start);
      //printf("Step %d num_modules %d\n", i, this_num_modules_batch);

      dim3 blocks = dim3(this_num_modules_batch, num_output_channels);
      dim3 threads(num_threads_x);
      kReadRows<<<blocks, threads>>>(derivs_data, expanded_derivs,
                                     num_images, num_modules,
                                     this_num_modules_batch, module_id_start);
      if (!conv) w += num_output_channels * input_size;
      cublasSgemm('n', 'n', 
                  num_images * this_num_modules_batch, kernel_size_x * kernel_size_y * num_input_channels,
                  num_output_channels,
                  scaleOutput, expanded_derivs, num_images * this_num_modules_batch,
                  w, num_output_channels,
                  0, expanded_target, num_images * this_num_modules_batch);

      if (check_cublas_error()) {
        printf("Error in dot or before it.\n");
      }
      dim3 blocks2 = dim3(this_num_modules_batch, num_input_channels);
      kContract<<<blocks2, threads>>>(expanded_target, targets_data,
                                   num_images, num_input_channels,
                                   image_size_y, image_size_x,
                                   num_modules_y, num_modules_x,
                                   kernel_size_y, kernel_size_x,
                                   padding_y, padding_x,
                                   stride_y, stride_x,
                                   this_num_modules_batch, module_id_start);
      module_id_start += this_num_modules_batch;
    }
    cudaFree(expanded_derivs);
    cudaFree(expanded_target);
    getLastCudaError("convDownGemm: kernel execution failed");
}

void _convOutpGemm(cudamat* images, cudamat* derivs, cudamat* targets,
              Shape4D images_shape, Shape4D derivs_shape, Shape4D targets_shape,
              ConvDesc conv_desc, float scaleTargets, float scaleOutput, bool conv) {

    int num_input_channels   = conv_desc.num_input_channels;
    int num_output_channels  = conv_desc.num_output_channels;
    int kernel_size_y        = conv_desc.kernel_size_y;
    int kernel_size_x        = conv_desc.kernel_size_x;
    int stride_y             = conv_desc.stride_y;
    int stride_x             = conv_desc.stride_x;
    int padding_y            = conv_desc.padding_y;
    int padding_x            = conv_desc.padding_x;
    int input_channel_begin  = conv_desc.input_channel_begin;
    int input_channel_end    = conv_desc.input_channel_end;
    int output_channel_begin = conv_desc.output_channel_begin;
    int output_channel_end   = conv_desc.output_channel_end;
    int num_groups           = conv_desc.num_groups;
    if (output_channel_end == 0) output_channel_end = num_output_channels;
    if (input_channel_end == 0) input_channel_end = num_input_channels;

    int num_output_channels2 = derivs_shape.shape[3];
    int num_modules_y        = derivs_shape.shape[2];
    int num_modules_x        = derivs_shape.shape[1];
    int num_images           = derivs_shape.shape[0];

    int num_input_channels2  = images_shape.shape[3];
    int image_size_y         = images_shape.shape[2];
    int image_size_x         = images_shape.shape[1];
    int num_images2          = images_shape.shape[0];

    int num_input_channels3Mult  = targets_shape.shape[3];
    int kernel_size_y2       = targets_shape.shape[2];
    int kernel_size_x2       = targets_shape.shape[1];
    int num_output_channels3 = targets_shape.shape[0];

    int num_modules          = num_modules_y * num_modules_x;
    int filterModuleMult     = conv ? 1 : num_modules;
  
    // Consistency checks. 
    assert (num_images == num_images2);
    assert (num_output_channels == num_output_channels2);
    assert (output_channel_end - output_channel_begin == num_output_channels3);
    assert (num_input_channels == num_input_channels2);
    assert (input_channel_end - input_channel_begin == num_input_channels3Mult / filterModuleMult);
    assert (num_images2 == images->size[0]);
    assert (num_images == derivs->size[0]);
    assert (num_output_channels3 == targets->size[0]);
    assert (image_size_y * image_size_x * num_input_channels2 == images->size[1]);
    assert (num_modules_y * num_modules_x * num_output_channels2 == derivs->size[1]);
    assert (kernel_size_y2 * kernel_size_x2 * num_input_channels3Mult == targets->size[1]);
    assert (kernel_size_y == kernel_size_y2);
    assert (kernel_size_x == kernel_size_x2);
    assert (num_input_channels % num_groups == 0);
    assert (num_groups == 1);
    assert (input_channel_begin  >= 0);
    assert (output_channel_begin >= 0);
    assert (input_channel_end    <= num_input_channels);
    assert (output_channel_end   <= num_output_channels);
    assert (input_channel_begin  <= input_channel_end);
    assert (output_channel_begin <= output_channel_end);
    if (output_channel_end == 0) output_channel_end = num_output_channels;
    if (input_channel_end == 0) input_channel_end = num_input_channels;
    num_input_channels = input_channel_end - input_channel_begin;
    num_output_channels = output_channel_end - output_channel_begin;
    assert(num_input_channels  > 0);
    assert(num_output_channels > 0);
    float* dw = targets->data_device;
    float* images_data = images->data_device + input_channel_begin * image_size_y * image_size_x * num_images;
    float* derivs_data = derivs->data_device + output_channel_begin * num_modules * num_images;
    
    int input_size = kernel_size_y * kernel_size_x * num_input_channels;
    int num_threads_x = MIN(num_images, NUM_THREADS_PER_BLOCK);
    
    float *expanded_images = NULL, *expanded_derivs = NULL;
    int num_modules_batch;

    int input_memory_size  = num_images * input_size * sizeof(float);
    int output_memory_size = num_images * num_output_channels * sizeof(float);
    if (free_space_ == 0) EstimateFreeSpace();
    int max_batch_size = free_space_ / (input_memory_size + output_memory_size);
    max_batch_size = MIN(max_batch_size, num_modules / filterModuleMult);
    max_batch_size = MIN(max_batch_size, MAX_BLOCKS_X);
    max_batch_size = MAX(max_batch_size, 1);

    cudaError_t err1, err2;
    err1 = cudaMalloc((void**)&expanded_images,  max_batch_size * input_memory_size);
    err2 = cudaMalloc((void**)&expanded_derivs, max_batch_size * output_memory_size);
    if (cudaSuccess != err1 || cudaSuccess != err2) {
      printf("Out of memory.\n");
      num_modules_batch = 1;
    } else {
      num_modules_batch = max_batch_size;
    }

    int num_iter = DIVUP(num_modules, num_modules_batch);

    _Scale(targets, scaleTargets);

    int module_id_start = 0;
    dim3 threads(num_threads_x);
    for (int i = 0; i < num_iter; i++) {
      int this_num_modules_batch = MIN(num_modules_batch, num_modules - module_id_start);
      //printf("Step %d num_modules %d\n", i, this_num_modules_batch);

      dim3 blocks = dim3(this_num_modules_batch, num_output_channels);
      kReadRows<<<blocks, threads>>>(derivs_data, expanded_derivs,
                                     num_images, num_modules,
                                     this_num_modules_batch, module_id_start);
      dim3 blocks2 = dim3(this_num_modules_batch, num_input_channels);
      kExpand<<<blocks2, threads>>>(images_data, expanded_images,
                                   num_images, num_input_channels,
                                   image_size_y, image_size_x,
                                   num_modules_y, num_modules_x,
                                   kernel_size_y, kernel_size_x,
                                   padding_y, padding_x,
                                   stride_y, stride_x,
                                   this_num_modules_batch, module_id_start);
      if (!conv) dw += num_output_channels * input_size;
      cublasSgemm('t', 'n', 
                  num_output_channels,
                  input_size,
                  num_images * this_num_modules_batch,
                  scaleOutput, expanded_derivs, num_images * this_num_modules_batch,
                  expanded_images, num_images * this_num_modules_batch,
                  1, dw, num_output_channels);
      if (check_cublas_error()) {
        printf("Error in dot or before it.\n");
      }
      module_id_start += this_num_modules_batch;
    }
    cudaFree(expanded_derivs);
    cudaFree(expanded_images);
    getLastCudaError("convOutpGemm: kernel execution failed");
}

template <class Pooler>
void _convPoolGemm(cudamat* images, cudamat* targets,
                Shape4D images_shape, Shape4D targets_shape,
                ConvDesc conv_desc, float scaleTargets, float scaleOutput, Pooler pooler) {
    
    int num_input_channels   = conv_desc.num_input_channels;
    int num_output_channels  = conv_desc.num_output_channels;
    int kernel_size_y        = conv_desc.kernel_size_y;
    int kernel_size_x        = conv_desc.kernel_size_x;
    int kernel_size_t        = conv_desc.kernel_size_t;
    int stride_y             = conv_desc.stride_y;
    int stride_x             = conv_desc.stride_x;
    int stride_t             = conv_desc.stride_t;
    int padding_y            = conv_desc.padding_y;
    int padding_x            = conv_desc.padding_x;
    int padding_t            = conv_desc.padding_t;

    int num_output_channels2 = targets_shape.shape[3];
    int num_modules_y        = targets_shape.shape[2];
    int num_modules_x        = targets_shape.shape[1];
    int num_images           = targets_shape.shape[0];

    int num_input_channels2  = images_shape.shape[3];
    int image_size_y         = images_shape.shape[2];
    int image_size_x         = images_shape.shape[1];
    int num_images2          = images_shape.shape[0];

    int image_size_t         = num_input_channels2 / num_input_channels;
    int num_modules_t        = num_output_channels2 / num_output_channels;
    int num_modules          = num_modules_y * num_modules_x * num_modules_t;
  
    // Consistency checks. 
    assert (num_images == num_images2);
    assert (num_input_channels2 % image_size_t == 0);
    assert (num_input_channels == num_input_channels2 / image_size_t);
    assert (num_output_channels2 % num_modules_t == 0);
    assert (num_output_channels == num_output_channels2 / num_modules_t);
    assert (num_images == images->size[0]);
    assert (num_images == targets->size[0]);
    assert (image_size_t * image_size_y * image_size_x * num_input_channels == images->size[1]);
    assert (num_modules_t * num_modules_y * num_modules_x * num_output_channels == targets->size[1]);

    _Scale(targets, scaleTargets);

    dim3 threads(NUM_THREADS_PER_BLOCK);
    int num_blocks_x = MIN(MAX_BLOCKS_X, num_modules);
    dim3 blocks = dim3(num_blocks_x, num_input_channels);
    kPool<<<blocks, threads>>>(images->data_device, targets->data_device,
                               num_images, num_input_channels,
                               image_size_y, image_size_x, image_size_t,
                               num_modules_y, num_modules_x, num_modules_t,
                               kernel_size_y, kernel_size_x, kernel_size_t,
                               padding_y, padding_x, padding_t,
                               stride_y, stride_x, stride_t, scaleOutput,
                               pooler);
    getLastCudaError("convLocalPool: kernel execution failed");
}

void _avgPoolUndoGemm(cudamat* derivs, cudamat* targets,
                Shape4D derivs_shape, Shape4D targets_shape,
                ConvDesc conv_desc, float scaleTargets, float scaleOutput) {
 
    int num_input_channels   = conv_desc.num_input_channels;
    int num_output_channels  = conv_desc.num_output_channels;
    int kernel_size_y        = conv_desc.kernel_size_y;
    int kernel_size_x        = conv_desc.kernel_size_x;
    int kernel_size_t        = conv_desc.kernel_size_t;
    int stride_y             = conv_desc.stride_y;
    int stride_x             = conv_desc.stride_x;
    int stride_t             = conv_desc.stride_t;
    int padding_y            = conv_desc.padding_y;
    int padding_x            = conv_desc.padding_x;
    int padding_t            = conv_desc.padding_t;

    int num_output_channels2 = derivs_shape.shape[3];
    int num_modules_y        = derivs_shape.shape[2];
    int num_modules_x        = derivs_shape.shape[1];
    int num_images           = derivs_shape.shape[0];

    int num_input_channels2  = targets_shape.shape[3];
    int image_size_y         = targets_shape.shape[2];
    int image_size_x         = targets_shape.shape[1];
    int num_images2          = targets_shape.shape[0];

    int image_size_t         = num_input_channels2 / num_input_channels;
    int num_modules_t        = num_output_channels2 / num_output_channels;
    int num_modules          = num_modules_y * num_modules_x * num_modules_t;
  
    // Consistency checks. 
    assert (num_images == num_images2);
    assert (num_input_channels2 % image_size_t == 0);
    assert (num_input_channels == num_input_channels2 / image_size_t);
    assert (num_output_channels2 % num_modules_t == 0);
    assert (num_output_channels == num_output_channels2 / num_modules_t);
    assert (num_images == targets->size[0]);
    assert (num_images == derivs->size[0]);
    assert (image_size_t * image_size_y * image_size_x * num_input_channels == targets->size[1]);
    assert (num_modules_t * num_modules_y * num_modules_x * num_output_channels == derivs->size[1]);

    _Scale(targets, scaleTargets);

    dim3 threads(NUM_THREADS_PER_BLOCK);
    int num_blocks_x = MIN(MAX_BLOCKS_X, num_modules);
    dim3 blocks = dim3(num_blocks_x, num_input_channels);
    kAvgPoolUndo<<<blocks, threads>>>(derivs->data_device, targets->data_device,
                               num_images, num_input_channels,
                               image_size_y, image_size_x, image_size_t,
                               num_modules_y, num_modules_x, num_modules_t,
                               kernel_size_y, kernel_size_x, kernel_size_t,
                               padding_y, padding_x, padding_t,
                               stride_y, stride_x, stride_t, scaleOutput);
    getLastCudaError("avgPoolUndo: kernel execution failed");
}

void _maxPoolUndoGemm(cudamat* images, cudamat* derivs, cudamat* maxes,
                      cudamat* targets, Shape4D targets_shape,
                      Shape4D derivs_shape, ConvDesc conv_desc,
                      float scaleTargets, float scaleOutput) {
    
    int num_input_channels   = conv_desc.num_input_channels;
    int num_output_channels  = conv_desc.num_output_channels;
    int kernel_size_y        = conv_desc.kernel_size_y;
    int kernel_size_x        = conv_desc.kernel_size_x;
    int kernel_size_t        = conv_desc.kernel_size_t;
    int stride_y             = conv_desc.stride_y;
    int stride_x             = conv_desc.stride_x;
    int stride_t             = conv_desc.stride_t;
    int padding_y            = conv_desc.padding_y;
    int padding_x            = conv_desc.padding_x;
    int padding_t            = conv_desc.padding_t;

    int num_output_channels2 = derivs_shape.shape[3];
    int num_modules_y        = derivs_shape.shape[2];
    int num_modules_x        = derivs_shape.shape[1];
    int num_images           = derivs_shape.shape[0];

    int num_input_channels2  = targets_shape.shape[3];
    int image_size_y         = targets_shape.shape[2];
    int image_size_x         = targets_shape.shape[1];
    int num_images2          = targets_shape.shape[0];

    int image_size_t         = num_input_channels2 / num_input_channels;
    int num_modules_t        = num_output_channels2 / num_output_channels;
    int num_modules          = num_modules_y * num_modules_x * num_modules_t;
  
    // Consistency checks. 
    assert (num_images == num_images2);
    assert (num_input_channels2 % image_size_t == 0);
    assert (num_input_channels == num_input_channels2 / image_size_t);
    assert (num_output_channels2 % num_modules_t == 0);
    assert (num_output_channels == num_output_channels2 / num_modules_t);
    assert (num_images == targets->size[0]);
    assert (num_images == derivs->size[0]);
    assert (image_size_t * image_size_y * image_size_x * num_input_channels == targets->size[1]);
    assert (num_modules_t * num_modules_y * num_modules_x * num_output_channels == derivs->size[1]);


    _Scale(targets, scaleTargets);

    dim3 threads(NUM_THREADS_PER_BLOCK);
    int num_blocks_x = MIN(MAX_BLOCKS_X, num_modules);
    dim3 blocks = dim3(num_blocks_x, num_input_channels);
    kMaxPoolUndo<<<blocks, threads>>>(images->data_device, derivs->data_device,
                               maxes->data_device, targets->data_device,
                               num_images, num_input_channels,
                               image_size_y, image_size_x, image_size_t,
                               num_modules_y, num_modules_x, num_modules_t,
                               kernel_size_y, kernel_size_x, kernel_size_t,
                               padding_y, padding_x, padding_t,
                               stride_y, stride_x, stride_t, scaleOutput);
    getLastCudaError("maxPoolUndo: kernel execution failed");
}

void _CrossMapRNorm(cudamat* images, cudamat* targets, int num_filters, int sizeF, float addScale, float powScale, bool blocked) {
  int num_locs = (images->size[0] * images->size[1]) / num_filters;
  int num_blocks = DIVUP(num_locs, NUM_THREADS_PER_BLOCK);
  kCrossMapRNorm<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(images->data_device, targets->data_device,
                 num_locs, addScale, powScale, num_filters, sizeF, blocked);
  getLastCudaError("_CrossMapRNorm: kernel execution failed");
}

void _CrossMapRNormUndo(cudamat* outGrads, cudamat* images, cudamat* targets,
                        int num_filters, int sizeF, float addScale,
                        float powScale, bool blocked) {
  int num_locs = (images->size[0] * images->size[1]) / num_filters;
  int batch_offset = 0;

  float *denoms;
  if (free_space_ == 0) EstimateFreeSpace();
  int max_batch_size = free_space_ / (sizeof(float) * num_filters);
  max_batch_size = MIN(num_locs, max_batch_size);
  max_batch_size = MIN(num_locs, MAX_BLOCKS_X);
  cudaError_t err;
  err = cudaMalloc((void**)&denoms, max_batch_size * num_filters * sizeof(float));
  if (cudaSuccess != err) {
    printf("Out of memory on GPU!\n");
  }
  int num_batches = DIVUP(num_locs, max_batch_size);
  for (int i = 0; i < num_batches; i++) {
    int batch_size = MIN(max_batch_size, num_locs - batch_offset);
    int num_blocks = DIVUP(batch_size, NUM_THREADS_PER_BLOCK);
    kCrossMapDenoms<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(images->data_device, denoms, num_locs, batch_size,
                    batch_offset, addScale, powScale, num_filters, sizeF, blocked);

    kCrossMapRNormUndo<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(images->data_device, outGrads->data_device, denoms,
                       targets->data_device, num_locs, batch_size, batch_offset,
                       addScale, powScale, num_filters, sizeF, blocked);
    batch_offset += batch_size;
  }

  cudaFree(denoms);
  getLastCudaError("_CrossMapRNormUndo: kernel execution failed");
}

#ifdef __cplusplus
extern "C" {
#endif

void convUpGemm(cudamat* images, cudamat* filters, cudamat* targets,
                Shape4D* images_shape, Shape4D* filters_shape,
                Shape4D* targets_shape, ConvDesc conv_desc,
                float scaleTargets) {
  /*
  printf("image shape %d %d %d %d\n", images_shape->shape[0], images_shape->shape[1], images_shape->shape[2], images_shape->shape[3]);
  printf("filters shape %d %d %d %d\n", filters_shape->shape[0], filters_shape->shape[1], filters_shape->shape[2], filters_shape->shape[3]);
  printf("targets shape %d %d %d %d\n", targets_shape->shape[0], targets_shape->shape[1], targets_shape->shape[2], targets_shape->shape[3]);
  printf("Convolution : kernel_size_y %d kernel_size_x %d stride_y %d stride_x %d padding_y %d padding_x %d num_input_channels %d num_output_channels %d num_groups %d\n",
          conv_desc.kernel_size_y, conv_desc.kernel_size_x, conv_desc.stride_x, conv_desc.stride_y, conv_desc.padding_y, conv_desc.padding_x, conv_desc.num_input_channels,
          conv_desc.num_output_channels, conv_desc.num_groups);
          */
  _convUpGemm(images, filters, targets, *images_shape, *filters_shape,
              *targets_shape, conv_desc, scaleTargets, 1.0, true);
}
void convDownGemm(cudamat* derivs, cudamat* filters, cudamat* targets,
              Shape4D* derivs_shape, Shape4D* filters_shape,
              Shape4D* targets_shape, ConvDesc conv_desc, float scaleTargets) {
  _convDownGemm(derivs, filters, targets, *derivs_shape, *filters_shape,
                *targets_shape, conv_desc, scaleTargets, 1.0, true);
}

void convOutpGemm(cudamat* images, cudamat* derivs, cudamat* targets,
              Shape4D* images_shape, Shape4D* derivs_shape, Shape4D* targets_shape,
              ConvDesc conv_desc, float scaleTargets, float scaleOutput) {
  _convOutpGemm(images, derivs, targets, *images_shape, *derivs_shape,
              *targets_shape, conv_desc, scaleTargets, scaleOutput, true);
}

void localUpGemm(cudamat* images, cudamat* filters, cudamat* targets,
                Shape4D* images_shape, Shape4D* filters_shape,
                Shape4D* targets_shape, ConvDesc conv_desc,
                float scaleTargets) {
  _convUpGemm(images, filters, targets, *images_shape, *filters_shape,
              *targets_shape, conv_desc, scaleTargets, 1.0, false);
}
void localDownGemm(cudamat* derivs, cudamat* filters, cudamat* targets,
              Shape4D* derivs_shape, Shape4D* filters_shape,
              Shape4D* targets_shape, ConvDesc conv_desc, float scaleTargets) {
  _convDownGemm(derivs, filters, targets, *derivs_shape, *filters_shape,
                *targets_shape, conv_desc, scaleTargets, 1.0, false);
}

void localOutpGemm(cudamat* images, cudamat* derivs, cudamat* targets,
              Shape4D* images_shape, Shape4D* derivs_shape, Shape4D* targets_shape,
              ConvDesc conv_desc, float scaleTargets, float scaleOutput) {
  _convOutpGemm(images, derivs, targets, *images_shape, *derivs_shape,
              *targets_shape, conv_desc, scaleTargets, scaleOutput, false);
}

void MaxPoolGemm(cudamat* images, cudamat* targets, Shape4D* images_shape,
             Shape4D* targets_shape, ConvDesc conv_desc, float scaleTargets, float scaleOutput){
  MaxPooler pooler;
  _convPoolGemm<MaxPooler>(images, targets, *images_shape, *targets_shape,
                           conv_desc, scaleTargets, scaleOutput, pooler);
}

void AvgPoolGemm(cudamat* images, cudamat* targets, Shape4D* images_shape,
             Shape4D* targets_shape, ConvDesc conv_desc, float scaleTargets, float scaleOutput){
  AvgPooler pooler;
  _convPoolGemm<AvgPooler>(images, targets, *images_shape, *targets_shape,
                           conv_desc, scaleTargets, scaleOutput, pooler);
}

void MaxPoolUndoGemm(cudamat* images, cudamat* maxGrads, cudamat* maxActs,
                 cudamat* targets, Shape4D* images_shape, Shape4D* maxGrads_shape,
                 ConvDesc conv_desc, float scaleTargets) {
  _maxPoolUndoGemm(images, maxGrads, maxActs, targets, *images_shape,
                   *maxGrads_shape, conv_desc, scaleTargets, 1);
}

void AvgPoolUndoGemm(cudamat* avgGrads, cudamat* targets, Shape4D* avgGrads_shape,
                 Shape4D* targets_shape, ConvDesc conv_desc, float scaleTargets) {
  _avgPoolUndoGemm(avgGrads, targets, *avgGrads_shape, *targets_shape, conv_desc,
                   scaleTargets, 1);
}

void UpSampleGemm(cudamat* images, cudamat* targets, Shape4D* images_shape,
              Shape4D* targets_shape, int factor, float scaleTargets) { 
  ConvDesc conv_desc;
  conv_desc.kernel_size_y = factor;
  conv_desc.kernel_size_x = factor;
  conv_desc.stride_y = factor;
  conv_desc.stride_x = factor;
  conv_desc.padding_y = 0;
  conv_desc.padding_x = 0;
  conv_desc.num_input_channels = images_shape->shape[3];
  conv_desc.num_output_channels = targets_shape->shape[3];
  conv_desc.num_groups = 1;
  _avgPoolUndoGemm(images, targets, *images_shape, *targets_shape, conv_desc,
                   scaleTargets, factor * factor);
}

void DownSampleGemm(cudamat* images, cudamat* targets, Shape4D* images_shape, Shape4D* targets_shape, int factor) {
  AvgPooler pooler = AvgPooler();
  ConvDesc conv_desc;
  conv_desc.kernel_size_y = factor;
  conv_desc.kernel_size_x = factor;
  conv_desc.stride_y = factor;
  conv_desc.stride_x = factor;
  conv_desc.padding_y = 0;
  conv_desc.padding_x = 0;
  conv_desc.num_input_channels = images_shape->shape[3];
  conv_desc.num_output_channels = targets_shape->shape[3];
  conv_desc.num_groups = 1;
  _convPoolGemm<AvgPooler>(images, targets, *images_shape, *targets_shape,
                           conv_desc, 0, 1, pooler);
}

void ResponseNormCrossMapGemm(
  cudamat* images, cudamat* targets, int num_filters, int sizeF, float addScale,
  float powScale, bool blocked) {
  _CrossMapRNorm(images, targets, num_filters, sizeF, addScale, powScale, blocked);
}

void ResponseNormCrossMapUndoGemm(
  cudamat* outGrads, cudamat* inputs, cudamat* targets, int num_filters,
  int sizeF, float addScale, float powScale, bool blocked) {
  _CrossMapRNormUndo(outGrads, inputs, targets, num_filters, sizeF, addScale,
                     powScale, blocked);
}
void Scale(cudamat* mat, float scale) {
  _Scale(mat, scale);
}
#ifdef __cplusplus
}
#endif
