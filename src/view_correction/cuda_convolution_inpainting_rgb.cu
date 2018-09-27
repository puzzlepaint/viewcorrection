// Copyright 2018 ETH Zürich
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "view_correction/cuda_convolution_inpainting_rgb.cuh"

#include <cub/cub.cuh>
#include <glog/logging.h>

#include "view_correction/cuda_util.h"
#include "view_correction/helper_math.h"

namespace view_correction {

#define kIterationsPerKernelCall 4

const int kBlockWidth = 32;
const int kBlockHeight = 32;

#define kSqrt2 1.4142135623731f

template<int block_size_x, int block_size_y>
__global__ void RGBConvolutionInpaintingInitializeVariablesKernel(
    int grid_dim_x,
    CUDABuffer_<uchar4> input,
    CUDABuffer_<uchar4> output,
    CUDABuffer_<uint16_t> block_coordinates) {
  const int width = output.width();
  const int height = output.height();
  
  const int kBlockOutputSizeX = block_size_x - 2 * kIterationsPerKernelCall;
  const int kBlockOutputSizeY = block_size_y - 2 * kIterationsPerKernelCall;
  unsigned int x = blockIdx.x * kBlockOutputSizeX + threadIdx.x - kIterationsPerKernelCall;
  unsigned int y = blockIdx.y * kBlockOutputSizeY + threadIdx.y - kIterationsPerKernelCall;
  
  const bool kOutput =
      threadIdx.x >= kIterationsPerKernelCall &&
      threadIdx.y >= kIterationsPerKernelCall &&
      threadIdx.x < block_size_x - kIterationsPerKernelCall &&
      threadIdx.y < block_size_y - kIterationsPerKernelCall &&
      x < width &&
      y < height;
  
  bool thread_is_active = false;
  if (kOutput) {
    output(y, x) = input(y, x);
    thread_is_active = (input(y, x).w == 0);
  }
  
  typedef cub::BlockReduce<
      int, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y> BlockReduceInt;
  __shared__ typename BlockReduceInt::TempStorage int_storage;
  int num_active_threads = BlockReduceInt(int_storage).Sum(thread_is_active ? 1 : 0);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    block_coordinates(0, blockIdx.x + blockIdx.y * grid_dim_x) = num_active_threads;
  }
}

__forceinline__ __device__ float4 uchar4ToFloat4(const uchar4& input) {
  return make_float4(input.x, input.y, input.z, input.w);
}

__forceinline__ __device__ uchar4 float4ToUchar4(const float4& input) {
  return make_uchar4(input.x, input.y, input.z, input.w);
}

template<int block_size_x, int block_size_y, bool check_convergence>
__global__ void RGBConvolutionInpaintingKernel(
    CUDABuffer_<uint16_t> block_coordinates,
    CUDABuffer_<uchar4> input,
    CUDABuffer_<uint8_t> max_change,
    float max_change_rate_threshold,
    CUDABuffer_<uchar4> output) {
  const int x = max(0, min(output.width() - 1, block_coordinates(0, 2 * blockIdx.x + 0) + threadIdx.x - kIterationsPerKernelCall));
  const int y = max(0, min(output.height() - 1, block_coordinates(0, 2 * blockIdx.x + 1) + threadIdx.y - kIterationsPerKernelCall));
  
  const bool kIsPixelToInpaint = (input(y, x).w == 0);
  const bool kOutput =
      threadIdx.x >= kIterationsPerKernelCall &&
      threadIdx.y >= kIterationsPerKernelCall &&
      threadIdx.x < block_size_x - kIterationsPerKernelCall &&
      threadIdx.y < block_size_y - kIterationsPerKernelCall &&
      block_coordinates(0, 2 * blockIdx.x + 0) + threadIdx.x - kIterationsPerKernelCall < output.width() &&
      block_coordinates(0, 2 * blockIdx.x + 1) + threadIdx.y - kIterationsPerKernelCall < output.height();
  
  // Load inputs into private or shared memory.
  // NOTE: Could also use uchar4 here to save memory, but this would introduce some additional conversions.
  __shared__ float4 color_shared[block_size_x * block_size_y];
  int shared_mem_index = threadIdx.x + block_size_x * threadIdx.y;
  color_shared[shared_mem_index] = uchar4ToFloat4(output(y, x));
  
  // Wait for shared memory to be loaded.
  __syncthreads();
  
#pragma unroll
  for (int i = 0; i < kIterationsPerKernelCall; ++ i) {
    float4 result = make_float4(0, 0, 0, 0);
    float weight = 0;
    float pixel_weight;
    if (kIsPixelToInpaint &&
        threadIdx.x > 0 &&
        threadIdx.y > 0 &&
        threadIdx.x < block_size_x - 1 &&
        threadIdx.y < block_size_y - 1) {
      pixel_weight =
          (y > 0 && x > 0 && color_shared[shared_mem_index - 1 - block_size_x].w > 0) *
          0.073235f;
      result += pixel_weight * color_shared[shared_mem_index - 1 - block_size_x];
      weight += pixel_weight;
      
      pixel_weight =
          (y > 0 && color_shared[shared_mem_index - block_size_x].w > 0) *
          0.176765f;
      result += pixel_weight * color_shared[shared_mem_index - block_size_x];
      weight += pixel_weight;
      
      pixel_weight =
          (y > 0 && x < output.width() - 1 && color_shared[shared_mem_index + 1 - block_size_x].w > 0) *
          0.073235f;
      result += pixel_weight * color_shared[shared_mem_index + 1 - block_size_x];
      weight += pixel_weight;
      
      pixel_weight =
          (x > 0 && color_shared[shared_mem_index - 1].w > 0) *
          0.176765f;
      result += pixel_weight * color_shared[shared_mem_index - 1];
      weight += pixel_weight;
      
      pixel_weight =
          (x < output.width() - 1 && color_shared[shared_mem_index + 1].w > 0) *
          0.176765f;
      result += pixel_weight * color_shared[shared_mem_index + 1];
      weight += pixel_weight;
      
      pixel_weight =
          (y < output.height() - 1 && x > 0 && color_shared[shared_mem_index - 1 + block_size_x].w > 0) *
          0.073235f;
      result += pixel_weight * color_shared[shared_mem_index - 1 + block_size_x];
      weight += pixel_weight;
      
      pixel_weight =
          (y < output.height() - 1 && color_shared[shared_mem_index + block_size_x].w > 0) *
          0.176765f;
      result += pixel_weight * color_shared[shared_mem_index + block_size_x];
      weight += pixel_weight;
      
      pixel_weight =
          (y < output.height() - 1 && x < output.width() - 1 && color_shared[shared_mem_index + 1 + block_size_x].w > 0) *
          0.073235f;
      result += pixel_weight * color_shared[shared_mem_index + 1 + block_size_x];
      weight += pixel_weight;
      
      // Version without explicit handling of uninitialized values:
//       result = 0.073235f * color_shared[shared_mem_index - 1 - block_size_x] +
//                0.176765f * color_shared[shared_mem_index - block_size_x] +
//                0.073235f * color_shared[shared_mem_index + 1 - block_size_x] +
//                0.176765f * color_shared[shared_mem_index - 1] +
//                0 +
//                0.176765f * color_shared[shared_mem_index + 1] +
//                0.073235f * color_shared[shared_mem_index - 1 + block_size_x] +
//                0.176765f * color_shared[shared_mem_index + block_size_x] +
//                0.073235f * color_shared[shared_mem_index + 1 + block_size_x];
    }
    __syncthreads();
    
    float4 new_color = (1.0f / weight) * result;
    
    // Convergence test.
    float change = 0;
    if (check_convergence && kOutput && kIsPixelToInpaint && i == kIterationsPerKernelCall - 1) {
      change = max(max(fabs((new_color.x - color_shared[shared_mem_index].x) / color_shared[shared_mem_index].x),
                       fabs((new_color.y - color_shared[shared_mem_index].y) / color_shared[shared_mem_index].y)),
                   fabs((new_color.z - color_shared[shared_mem_index].z) / color_shared[shared_mem_index].z));
    }
    if (check_convergence) {
      typedef cub::BlockReduce<
          int, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y> BlockReduceInt;
      __shared__ typename BlockReduceInt::TempStorage int_storage;
      int active_pixels = BlockReduceInt(int_storage).Sum(change > max_change_rate_threshold);
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        max_change(0, blockIdx.x) = (active_pixels > 0) ? 1 : 0;
      }
    }
    
    if (kIsPixelToInpaint && weight > 0) {
      color_shared[shared_mem_index] = new_color;
    }
    __syncthreads();
  }
  
  if (kOutput && kIsPixelToInpaint) {
    output(y, x) = float4ToUchar4(color_shared[shared_mem_index] + make_float4(0.5f, 0.5f, 0.5f, 0.5f));
  }
}

template<int block_size_x, int block_size_y, bool check_convergence>
__global__ void
__launch_bounds__(32*32, 1)
RGBConvolutionInpaintingKernelWithWeighting(
    CUDABuffer_<uint16_t> block_coordinates,
    CUDABuffer_<uchar4> input,
    cudaTextureObject_t gradient_magnitude_div_sqrt2,
    CUDABuffer_<uint8_t> max_change,
    float max_change_rate_threshold,
    CUDABuffer_<uchar4> output) {
  __shared__ float4 color_shared[block_size_x * block_size_y];
  __shared__ float weights_shared[block_size_x * block_size_y];
  
  // NOTE: According to the internet, using volatile puts it into a register.
  volatile unsigned int threadIdx_x = threadIdx.x;
  volatile unsigned int threadIdx_y = threadIdx.y;
  volatile int shared_mem_index;
  __shared__ float base_weight[block_size_x * block_size_y];
  bool kIsPixelToInpaint;
  bool kOutput;
  
  {
    const int raw_x = block_coordinates(0, 2 * blockIdx.x + 0) + threadIdx_x - kIterationsPerKernelCall;
    const int raw_y = block_coordinates(0, 2 * blockIdx.x + 1) + threadIdx_y - kIterationsPerKernelCall;
    
    const bool kInImage =
        raw_x >= 0 &&
        raw_y >= 0 &&
        raw_x < output.width() &&
        raw_y < output.height();
    kIsPixelToInpaint = kInImage && (input(raw_y, raw_x).w == 0);
    kOutput =
        threadIdx_x >= kIterationsPerKernelCall &&
        threadIdx_y >= kIterationsPerKernelCall &&
        threadIdx_x < block_size_x - kIterationsPerKernelCall &&
        threadIdx_y < block_size_y - kIterationsPerKernelCall &&
        kInImage && kIsPixelToInpaint;
    
    // Load inputs into private or shared memory.
    // NOTE: Could also use uchar4 here to save memory, but this would introduce some additional conversions.
    shared_mem_index = threadIdx_x + block_size_x * threadIdx_y;
    if (kInImage) {
      color_shared[shared_mem_index] = uchar4ToFloat4(output(raw_y, raw_x));
    }
    base_weight[shared_mem_index] = (kInImage ? 1 : 0) *  1.f / (1.f + 50.f * tex2D<uchar>(gradient_magnitude_div_sqrt2, raw_x, raw_y) * kSqrt2 / 255.f);
    weights_shared[shared_mem_index] = base_weight[shared_mem_index] * (color_shared[shared_mem_index].w > 0);
  }
  
  // Wait for shared memory to be loaded.
  __syncthreads();
  
#pragma unroll
  for (int i = 0; i < kIterationsPerKernelCall; ++ i) {
    float4 new_color = make_float4(0, 0, 0, 0);
    if (kIsPixelToInpaint &&
        threadIdx_x > 0 &&
        threadIdx_y > 0 &&
        threadIdx_x < block_size_x - 1 &&
        threadIdx_y < block_size_y - 1) {
      float weight = 0;
      float pixel_weight =
          0.073235f * weights_shared[shared_mem_index - 1 - block_size_x];
      new_color += pixel_weight * color_shared[shared_mem_index - 1 - block_size_x];
      weight += pixel_weight;
      
      pixel_weight =
          0.176765f * weights_shared[shared_mem_index - block_size_x];
      new_color += pixel_weight * color_shared[shared_mem_index - block_size_x];
      weight += pixel_weight;
      
      pixel_weight =
          0.073235f * weights_shared[shared_mem_index + 1 - block_size_x];
      new_color += pixel_weight * color_shared[shared_mem_index + 1 - block_size_x];
      weight += pixel_weight;
      
      pixel_weight =
          0.176765f * weights_shared[shared_mem_index - 1];
      new_color += pixel_weight * color_shared[shared_mem_index - 1];
      weight += pixel_weight;
      
      pixel_weight =
          0.176765f * weights_shared[shared_mem_index + 1];
      new_color += pixel_weight * color_shared[shared_mem_index + 1];
      weight += pixel_weight;
      
      pixel_weight =
          0.073235f * weights_shared[shared_mem_index - 1 + block_size_x];
      new_color += pixel_weight * color_shared[shared_mem_index - 1 + block_size_x];
      weight += pixel_weight;
      
      pixel_weight =
          0.176765f * weights_shared[shared_mem_index + block_size_x];
      new_color += pixel_weight * color_shared[shared_mem_index + block_size_x];
      weight += pixel_weight;
      
      pixel_weight =
          0.073235f * weights_shared[shared_mem_index + 1 + block_size_x];
      new_color += pixel_weight * color_shared[shared_mem_index + 1 + block_size_x];
      weight += pixel_weight;
      
      // Version without explicit handling of uninitialized values:
      // (And without weights):
//       result = 0.073235f * color_shared[shared_mem_index - 1 - block_size_x] +
//                0.176765f * color_shared[shared_mem_index - block_size_x] +
//                0.073235f * color_shared[shared_mem_index + 1 - block_size_x] +
//                0.176765f * color_shared[shared_mem_index - 1] +
//                0 +
//                0.176765f * color_shared[shared_mem_index + 1] +
//                0.073235f * color_shared[shared_mem_index - 1 + block_size_x] +
//                0.176765f * color_shared[shared_mem_index + block_size_x] +
//                0.073235f * color_shared[shared_mem_index + 1 + block_size_x];
      
      new_color = (1.0f / weight) * new_color;
    }
    __syncthreads();
    
    // Convergence test.
    if (check_convergence && i == kIterationsPerKernelCall - 1) {
      float change = 0;
      if (kOutput) {
        change = max(max(fabs((new_color.x - color_shared[shared_mem_index].x) / color_shared[shared_mem_index].x),
                        fabs((new_color.y - color_shared[shared_mem_index].y) / color_shared[shared_mem_index].y)),
                    fabs((new_color.z - color_shared[shared_mem_index].z) / color_shared[shared_mem_index].z));
      }
      
      typedef cub::BlockReduce<
          int, block_size_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS, block_size_y> BlockReduceInt;
      __shared__ typename BlockReduceInt::TempStorage int_storage;
      // "change" now becomes "active_pixel_count".
      change = BlockReduceInt(int_storage).Sum(change > max_change_rate_threshold);
      if (threadIdx_x == 0 && threadIdx_y == 0) {
        max_change(0, blockIdx.x) = (change > 0) ? 1 : 0;
      }
    }
    
    if (kIsPixelToInpaint && new_color.w > 0) {
      color_shared[shared_mem_index] = new_color;
      if (i < kIterationsPerKernelCall - 1) {
        weights_shared[shared_mem_index] = base_weight[shared_mem_index] * (new_color.w > 0);
      }
    }
    if (i < kIterationsPerKernelCall - 1) {
      __syncthreads();
    }
  }
  
  if (kOutput) {
    const int x = block_coordinates(0, 2 * blockIdx.x + 0) + threadIdx_x - kIterationsPerKernelCall;
    const int y = block_coordinates(0, 2 * blockIdx.x + 1) + threadIdx_y - kIterationsPerKernelCall;
    if (x >= 0 &&
        y >= 0 &&
        x < output.width() &&
        y < output.height()) {
      output(y, x) = float4ToUchar4(color_shared[shared_mem_index] + make_float4(0.5f, 0.5f, 0.5f, 0.5f));
    }
  }
}

int InpaintImageWithConvolutionCUDA(
    cudaStream_t stream,
    bool use_weighting,
    int max_num_iterations,
    float max_change_rate_threshold,
    cudaTextureObject_t gradient_magnitude_div_sqrt2,
    const CUDABuffer<uchar4>& input,
    CUDABuffer<uint8_t>* max_change,
    CUDABuffer<uchar4>* output,
    CUDABuffer<uint16_t>* block_coordinates,
    uint32_t* pixel_to_inpaint_count) {
  const int width = output->width();
  const int height = output->height();
  
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  
  const int kBlockOutputSizeX = kBlockWidth - 2 * kIterationsPerKernelCall;
  const int kBlockOutputSizeY = kBlockHeight - 2 * kIterationsPerKernelCall;
  dim3 grid_dim(cuda_util::GetBlockCount(width, kBlockOutputSizeX),
                cuda_util::GetBlockCount(height, kBlockOutputSizeY));
  
  // Initialize variables.
  CHECK_EQ(kBlockWidth, 32);
  CHECK_EQ(kBlockHeight, 32);
  RGBConvolutionInpaintingInitializeVariablesKernel<32, 32><<<grid_dim, block_dim, 0, stream>>>(
      grid_dim.x, input.ToCUDA(), output->ToCUDA(), block_coordinates->ToCUDA());
  CHECK_CUDA_NO_ERROR();
  
  uint16_t* block_activity = new uint16_t[grid_dim.x * grid_dim.y];
  block_coordinates->DownloadPartAsync(0, grid_dim.x * grid_dim.y * sizeof(uint16_t), stream, block_activity);
  cudaStreamSynchronize(stream);
  int active_block_count = 0;
  *pixel_to_inpaint_count = 0;
  uint16_t* block_coordinates_cpu = new uint16_t[2 * grid_dim.x * grid_dim.y];
  for (size_t y = 0; y < grid_dim.y; ++ y) {
    for (size_t x = 0; x < grid_dim.x; ++ x) {
      if (block_activity[x + y * grid_dim.x] > 0) {
        block_coordinates_cpu[2 * active_block_count + 0] = x * kBlockOutputSizeX;
        block_coordinates_cpu[2 * active_block_count + 1] = y * kBlockOutputSizeY;
        ++ active_block_count;
        *pixel_to_inpaint_count += block_activity[x + y * grid_dim.x];
      }
    }
  }
  delete[] block_activity;
  if (active_block_count == 0) {
    delete[] block_coordinates_cpu;
    LOG(INFO) << "Color inpainting converged after iteration: 0";
    return 0;
  }
  block_coordinates->UploadPartAsync(0, 2 * active_block_count * sizeof(uint16_t), stream, block_coordinates_cpu);
  
  uint8_t* max_change_cpu = new uint8_t[grid_dim.x * grid_dim.y];
  
  // Run convolution iterations.
  int i = 0;
  int last_convergence_check_iteration = -9999;
  for (i = 0; i < max_num_iterations; i += kIterationsPerKernelCall) {
    const bool check_convergence = (i - last_convergence_check_iteration >= 25);
    
    dim3 grid_dim_active(active_block_count);
    CHECK_EQ(kBlockWidth, 32);
    CHECK_EQ(kBlockHeight, 32);
    if (use_weighting) {
      if (check_convergence) {
        RGBConvolutionInpaintingKernelWithWeighting<32, 32, true><<<grid_dim_active, block_dim, 0, stream>>>(
            block_coordinates->ToCUDA(),
            input.ToCUDA(),
            gradient_magnitude_div_sqrt2,
            max_change->ToCUDA(),
            max_change_rate_threshold,
            output->ToCUDA());
      } else {
        RGBConvolutionInpaintingKernelWithWeighting<32, 32, false><<<grid_dim_active, block_dim, 0, stream>>>(
            block_coordinates->ToCUDA(),
            input.ToCUDA(),
            gradient_magnitude_div_sqrt2,
            max_change->ToCUDA(),
            max_change_rate_threshold,
            output->ToCUDA());
      }
    } else {
      if (check_convergence) {
        RGBConvolutionInpaintingKernel<32, 32, true><<<grid_dim_active, block_dim, 0, stream>>>(
            block_coordinates->ToCUDA(),
            input.ToCUDA(),
            max_change->ToCUDA(),
            max_change_rate_threshold,
            output->ToCUDA());
      } else {
        RGBConvolutionInpaintingKernel<32, 32, false><<<grid_dim_active, block_dim, 0, stream>>>(
            block_coordinates->ToCUDA(),
            input.ToCUDA(),
            max_change->ToCUDA(),
            max_change_rate_threshold,
            output->ToCUDA());
      }
    }
    
    if (check_convergence) {
      max_change->DownloadPartAsync(0, active_block_count * sizeof(uint8_t), stream, max_change_cpu);
      cudaStreamSynchronize(stream);
      int new_active_block_count = 0;
      for (int j = 0, end = active_block_count; j < end; j ++) {
        if (max_change_cpu[j]) {
          ++ new_active_block_count;
        }
      }
      if (new_active_block_count == 0) {
        i += kIterationsPerKernelCall;  // For correct iteration count logging.
        break;
      }
      last_convergence_check_iteration = i;
    }
  }
  
  delete[] max_change_cpu;
  delete[] block_coordinates_cpu;
  CHECK_CUDA_NO_ERROR();
  
  if (i < max_num_iterations) {
    LOG(INFO) << "Color inpainting converged after iteration: " << i;
  } else {
    LOG(WARNING) << "Color inpainting used maximum iteration count: " << i;
  }
  return i;
}

}
