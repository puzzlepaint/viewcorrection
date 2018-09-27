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

#include "view_correction/cuda_tv_inpainting_functions.cuh"

#include <cub/cub.cuh>
#include <glog/logging.h>

#include "view_correction/cuda_util.h"
#include "view_correction/helper_math.h"

namespace view_correction {

constexpr float kSqrt2 = 1.4142135623731f;
constexpr float kHuberEpsilon = 0.01f;  
constexpr float kCellChangeThreshold = 10e-6f;
constexpr float kDualIntToFloat = 2.f / 32767;  // std::numeric_limits<int16_t>::max();

// One iteration is one dual (D) and one primal (P) step. Theoretically one
// could make this more fine-grained to for example also do DPD and PDP, but it
// would complicate the code unnecessarily.
constexpr int kIterationsPerKernelCall = 4;

__global__ void TVInpaintingInitializeVariablesKernel(
    int grid_dim_x,
    bool kUseSingleKernel,
    float depth_input_scaling_factor,
    cudaTextureObject_t depth_map_input,
    CUDABuffer_<bool> tv_flag,
    CUDABuffer_<bool> tv_dual_flag,
    CUDABuffer_<int16_t> tv_dual_x,
    CUDABuffer_<int16_t> tv_dual_y,
    CUDABuffer_<float> tv_u,
    CUDABuffer_<float> tv_u_bar,
    CUDABuffer_<uint16_t> block_coordinates) {
  const int width = tv_u.width();
  const int height = tv_u.height();
  
  unsigned int x;
  unsigned int y;
  if (kUseSingleKernel) {
    const int kBlockOutputSizeX = 32 - 2 * kIterationsPerKernelCall;
    const int kBlockOutputSizeY = 32 - 2 * kIterationsPerKernelCall;
    x = blockIdx.x * kBlockOutputSizeX + threadIdx.x - kIterationsPerKernelCall;
    y = blockIdx.y * kBlockOutputSizeY + threadIdx.y - kIterationsPerKernelCall;
  } else {
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
  }
  const bool kOutput =
      threadIdx.x >= kIterationsPerKernelCall &&
      threadIdx.y >= kIterationsPerKernelCall &&
      threadIdx.x < 32 - kIterationsPerKernelCall &&
      threadIdx.y < 32 - kIterationsPerKernelCall &&
      x < width &&
      y < height;

  bool thread_is_active = false;
  if (kOutput) {
    tv_dual_x(y, x) = 0;
    tv_dual_y(y, x) = 0;
    const float depth_input = depth_input_scaling_factor * tex2D<float>(depth_map_input, x, y);
    tv_flag(y, x) = (depth_input == 0);
    thread_is_active =
        (depth_input == 0 ||
         (x > 0 && tex2D<float>(depth_map_input, x - 1, y) == 0) ||
         (y > 0 && tex2D<float>(depth_map_input, x, y - 1) == 0) ||
         (x < width - 1 && tex2D<float>(depth_map_input, x + 1, y) == 0) ||
         (y < height - 1 && tex2D<float>(depth_map_input, x, y + 1) == 0));
    tv_dual_flag(y, x) = thread_is_active;
    tv_u(y, x) = depth_input;
    tv_u_bar(y, x) = depth_input;
  }
  
  typedef cub::BlockReduce<
      int, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 32> BlockReduceInt;
  __shared__ typename BlockReduceInt::TempStorage int_storage;
  int num_active_threads = BlockReduceInt(int_storage).Reduce(thread_is_active ? 1 : 0, cub::Sum());
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    reinterpret_cast<uint8_t*>(block_coordinates.address())[blockIdx.x + blockIdx.y * grid_dim_x] = (num_active_threads > 0) ? 1 : 0;
  }
}

// set up all block activities (primal and dual) for all blocks
// that overlap with the inpainting area
__global__ void InitBlockActivationsFromInpaintRegionKernel(
  cudaTextureObject_t d_f,  
  const int width,
  const int height,
  CUDABuffer_<unsigned char> d_block_activities)
{
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    bool blockIsEmpty = true;
    // for all threads in the block
    for (int j=0; j<blockDim.y; ++j) {
      for (int i=0; i<blockDim.x; ++i) {
        const int x = blockIdx.x * blockDim.x + i;
        const int y = blockIdx.y * blockDim.y + j;
        if (x < width && y < height && tex2D<float>(d_f, x, y) == 0) {
          blockIsEmpty = false;
          break;
        }
      }
      if (!blockIsEmpty) break;
    }
    
    if (blockIsEmpty) d_block_activities(blockIdx.y, blockIdx.x) = 0;
    else d_block_activities(blockIdx.y, blockIdx.x) = 3;

  } // if (threadIdx.x == 0 && threadIdx.y == 0)
}

// checks the convergence of individual blocks and keeps track of the block boundary
// in order to steer deactivation and reactivation of block updates
__device__ void UpdateBlockActivations(    
    const float local_value,
    const float prev_value,
    const float cell_change_threshold,
    const unsigned char activity_flag,
    volatile float *sdata,
    CUDABuffer_<unsigned char> d_block_activities) {
  const float diff = local_value != 0 ? 
      fabs(local_value-prev_value)/fabs(local_value) : 
      (prev_value != 0 ? fabs(local_value-prev_value)/fabs(prev_value) : 0);

  sdata[threadIdx.x + blockDim.x*threadIdx.y] = diff;   // save value to shared memory

  __syncthreads();

  // reduction code to compute column sums of shared memory in parallel
  float sum = 0;
  float lsum, rsum, tsum = 0, bsum = 0;
  if (threadIdx.y == 0) {
    for (int j=0; j<blockDim.y; ++j) {
      const float value = sdata[threadIdx.x + blockDim.x*j];
      if (j == 0) tsum += value;
      if (j == blockDim.y-1) bsum += value;
      sum += value;
    }
    if (threadIdx.x == 0) lsum = sum;
    if (threadIdx.x == blockDim.x-1) rsum = sum;
    sdata[threadIdx.x] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    // compute final sum for the whole warp
    sum = 0;
    for (int j=0; j<blockDim.x; ++j) sum += sdata[j];
    
    // unset activity flag if converged (i.e. change was very small)
    if (sum < cell_change_threshold*blockDim.x*blockDim.y) {      
      d_block_activities(blockIdx.y, blockIdx.x) &= ~activity_flag;  // unset flag
    }
  } // if (threadIdx.x == 0 && threadIdx.y == 0)

  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    // reactivate neighboring blocks if necessary
    if (lsum >= cell_change_threshold*blockDim.y && blockIdx.x > 0)
      d_block_activities(blockIdx.y, blockIdx.x-1) |= activity_flag;
    if (rsum >= cell_change_threshold*blockDim.y && blockIdx.x < gridDim.x-1)
      d_block_activities(blockIdx.y, blockIdx.x+1) |= activity_flag;
    if (tsum >= cell_change_threshold*blockDim.x && blockIdx.y > 0)
      d_block_activities(blockIdx.y-1, blockIdx.x) |= activity_flag;
    if (bsum >= cell_change_threshold*blockDim.x && blockIdx.y < gridDim.y-1)
      d_block_activities(blockIdx.y+1, blockIdx.x) |= activity_flag;
  } // if (threadIdx.x == 0 && threadIdx.y == 0)
}


// performs primal update and extrapolation step:
// u^{k+1} = u^k + tau* div(p^{k+1})
// \bar{u}^{k+1} = 2*u^{k+1} - u^k
template<bool check_convergence, bool block_adaptive>
__global__ void TVInpaintingPrimalStepKernel(
    const float cell_change_threshold,
    CUDABuffer_<bool> d_tv_flag,
    CUDABuffer_<int16_t> d_dualTVX,
    CUDABuffer_<int16_t> d_dualTVY,
    CUDABuffer_<float> d_u,
    CUDABuffer_<float> d_u_bar,
    CUDABuffer_<float> d_m,
    CUDABuffer_<unsigned char> d_block_activities) {
  if (block_adaptive) {
    // check block activity
    if (d_block_activities(blockIdx.y, blockIdx.x) == 0) return;
  }

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // this will accumulate the update step for the primal variable
  float update = 0;
  // this will accumulate all row entries of the linear operator for the preconditioned step width
  float rowSum = 0;
  float u = 0;

  // only update within the inpainting region (f == 0)
  if (x < d_u.width() && y < d_u.height() && d_tv_flag(y, x)) {    
    // compute divergence update of dualTV - Neumann boundary conditions,
    // keep track of row sum for preconditioning
    update += kDualIntToFloat * (d_dualTVX(y, x) + d_dualTVY(y, x));
    rowSum += 2;
    if (x > 0) {
      update -= kDualIntToFloat * d_dualTVX(y, x - 1);
      rowSum++;
    }
    if (y > 0) {
      update -= kDualIntToFloat * d_dualTVY(y - 1, x);
      rowSum++;
    }

    constexpr float kPrimalStepWidth = 1.f;
    const float tau = kPrimalStepWidth / rowSum;
    u = d_u(y, x);

    update = u + tau * update;

    // primal proximal point extrapolation
    constexpr float kGamma = 0.1f;
    update += kGamma * (update - u);

    d_u(y, x) = update;
    d_u_bar(y, x) = 2 * update - u;
    
    if (check_convergence) {
      d_m(y, x) = fabs((update - u) / u);
    }
  }

  if (block_adaptive) {
    extern __shared__ float sdata[];
    UpdateBlockActivations(update, u, cell_change_threshold, 1, 
      sdata, d_block_activities);
  }
}


// performs dual update step
// p^{k=1} = \Pi_{|p|<=g} [ p^k + \sigma * \nabla \bar{u}^k ]
// p^{k=1} = \Pi_{|p|<=g} [ (p^k + \sigma * \nabla \bar{u}^k) / (1+\sigma*huberEpsilon) ]
template<bool use_weighting, bool block_adaptive>
__global__ void TVInpaintingDualStepKernel(
    const float huber_epsilon,
    const float cell_change_threshold,
    CUDABuffer_<bool> d_tv_dual_flag,
    CUDABuffer_<float> d_u,
    cudaTextureObject_t d_tvWeight,
    CUDABuffer_<int16_t> d_dualTVX,
    CUDABuffer_<int16_t> d_dualTVY,
    CUDABuffer_<unsigned char> d_block_activities) {
  if (block_adaptive) {
    // check block activity
    if (d_block_activities(blockIdx.y, blockIdx.x) == 0) return;
  }

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  float resultX = 0, resultY = 0;
  float dualTVX = 0, dualTVY = 0;
  if (x < d_u.width() && y < d_u.height() && d_tv_dual_flag(y, x)) {
    // update using the gradient of u
    constexpr float kDualStepWidth = 1.f;
    const float huberFactor = 1.0f / (1.0f + kDualStepWidth * 0.5f * huber_epsilon);
    const float u = d_u(y, x);
    dualTVX = kDualIntToFloat * d_dualTVX(y, x);
    dualTVY = kDualIntToFloat * d_dualTVY(y, x);
    resultX =
        huberFactor * (dualTVX + kDualStepWidth * 0.5f *
        ( (x < d_u.width() - 1) ? (d_u(y, x + 1)  - u) : 0 ));
    resultY =
        huberFactor * (dualTVY + kDualStepWidth * 0.5f *
        ( (y < d_u.height() - 1) ? (d_u(y + 1, x) - u) : 0 ));
    
    // project onto the g-unit ball
    float denom;
    if (use_weighting) {
      // Optimization: remove 1 / weight and turn division by weight below into multiplication.
      const float weight = /*1.f /*/ (1.f + tex2D<uchar>(d_tvWeight, x, y) * kSqrt2 / 255.f);
      // const float weight = 1.f / (__expf(tex2D<uchar>(d_tvWeight, x, y) * kSqrt2 / 255.f)*5);
      denom = max(1.0f, hypotf(resultX, resultY) * weight);
    } else {
      denom = max(1.0f, hypotf(resultX, resultY));
    }
    resultX /= denom;
    resultY /= denom;

    // dual proximal point extrapolation
    constexpr float kGamma = 0.1f;
    resultX += kGamma*(resultX-dualTVX);
    resultY += kGamma*(resultY-dualTVY);

    // write result back into global memory
    d_dualTVX(y, x) = resultX * 1.f / kDualIntToFloat;
    d_dualTVY(y, x) = resultY * 1.f / kDualIntToFloat;
  }

  if (block_adaptive) {
    extern __shared__ float sdata[];
    UpdateBlockActivations(hypotf(resultX, resultY), 
      hypotf(dualTVX, dualTVY), cell_change_threshold, 2, 
      sdata, d_block_activities);
  }
}

// This kernel does not produce output for the first kIterationsPerKernelCall
// rows and columns and for the last kIterationsPerKernelCall rows and columns.
template<int block_size_x, int block_size_y, bool use_weighting, bool check_convergence>
__global__ void TVInpaintingDualAndPrimalStepsKernel(
    const float huber_epsilon,
    CUDABuffer_<uint16_t> block_coordinates,
    CUDABuffer_<bool> d_tv_flag,
    CUDABuffer_<bool> d_tv_dual_flag,
    cudaTextureObject_t d_tvWeight,
    CUDABuffer_<int16_t> d_dualTVX,
    CUDABuffer_<int16_t> d_dualTVY,
    CUDABuffer_<float> d_u,
    CUDABuffer_<float> d_u_bar,
    CUDABuffer_<float> d_m) {
  const int x = max(0, min(d_u.width() - 1, block_coordinates(0, 2 * blockIdx.x + 0) + threadIdx.x - kIterationsPerKernelCall));
  const int y = max(0, min(d_u.height() - 1, block_coordinates(0, 2 * blockIdx.x + 1) + threadIdx.y - kIterationsPerKernelCall));
  
  const bool kDualFlag = d_tv_dual_flag(y, x);
  const bool kPrimalFlag = d_tv_flag(y, x);
  const bool kOutput =
      threadIdx.x >= kIterationsPerKernelCall &&
      threadIdx.y >= kIterationsPerKernelCall &&
      threadIdx.x < block_size_x - kIterationsPerKernelCall &&
      threadIdx.y < block_size_y - kIterationsPerKernelCall &&
      block_coordinates(0, 2 * blockIdx.x + 0) + threadIdx.x - kIterationsPerKernelCall < d_u.width() &&
      block_coordinates(0, 2 * blockIdx.x + 1) + threadIdx.y - kIterationsPerKernelCall < d_u.height();
  
  typedef cub::BlockReduce<
      float, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 32> BlockReduceFloat;
  __shared__ typename BlockReduceFloat::TempStorage float_storage;
  
  // Load inputs into private or shared memory.
  __shared__ float u_bar_shared[block_size_x * block_size_y];
  __shared__ float dual_x_shared[block_size_x * block_size_y];
  __shared__ float dual_y_shared[block_size_x * block_size_y];

  int shared_mem_index = threadIdx.x + block_size_x * threadIdx.y;
  float u_bar = d_u_bar(y, x);
  float dualTVX = kDualIntToFloat * d_dualTVX(y, x);
  float dualTVY = kDualIntToFloat * d_dualTVY(y, x);
  float u = d_u(y, x);
  
  const float weight = /*1.f /*/ (1.f + tex2D<uchar>(d_tvWeight, x, y) * kSqrt2 / 255.f);
  // const float weight = 1.f / (__expf(tex2D<uchar>(d_tvWeight, x, y) * kSqrt2 / 255.f)*5);
  
  u_bar_shared[shared_mem_index] = u_bar;
  dual_x_shared[shared_mem_index] = dualTVX;
  dual_y_shared[shared_mem_index] = dualTVY;
  
  // Wait for shared memory to be loaded.
  __syncthreads();
  
#pragma unroll
  for (int i = 0; i < kIterationsPerKernelCall; ++ i) {
    // Dual step.
    if (kDualFlag) {
      // update using the gradient of u
      constexpr float kDualStepWidth = 1.f;
      const float huberFactor = 1.0f / (1.0f + kDualStepWidth * 0.5f * huber_epsilon);
      float resultX =
          huberFactor * (dualTVX + kDualStepWidth * 0.5f *
          ( (x < d_u_bar.width() - 1 && threadIdx.x < block_size_x - 1) ? (u_bar_shared[shared_mem_index + 1]  - u_bar) : 0));
      float resultY =
          huberFactor * (dualTVY + kDualStepWidth * 0.5f *
          ( (y < d_u_bar.height() - 1 && threadIdx.y < block_size_y - 1) ? (u_bar_shared[shared_mem_index + block_size_x] - u_bar) : 0));
      
      // project onto the g-unit ball
      float denom;
      if (use_weighting) {
        denom = max(1.0f, hypotf(resultX, resultY) * weight);
      } else {
        denom = max(1.0f, hypotf(resultX, resultY));
      }
      resultX /= denom;
      resultY /= denom;

      // dual proximal point extrapolation
      constexpr float kGamma = 0.1f;
      resultX += kGamma * (resultX - dualTVX);
      resultY += kGamma * (resultY - dualTVY);

      // write result back
      dualTVX = resultX;
      dualTVY = resultY;
      dual_x_shared[shared_mem_index] = dualTVX;
      dual_y_shared[shared_mem_index] = dualTVY;
    }
    __syncthreads();
    
    // Primal step.
    float max_change = 0;
    if (kPrimalFlag) {
      // compute divergence update of dualTV - Neumann boundary conditions,
      // keep track of row sum for preconditioning
      float update = dualTVX + dualTVY;
      float rowSum = 2;
      if (x > 0 && threadIdx.x > 0) {
        update -= dual_x_shared[shared_mem_index - 1];
        rowSum++;
      }
      if (y > 0 && threadIdx.y > 0) {
        update -= dual_y_shared[shared_mem_index - block_size_x];
        rowSum++;
      }

      constexpr float kPrimalStepWidth = 1.f;
      const float tau = kPrimalStepWidth / rowSum;

      update = u + tau * update;

      // primal proximal point extrapolation
      constexpr float kGamma = 0.1f;
      update += kGamma * (update - u);

      // write result back
      u_bar = 2 * update - u;
      if (check_convergence && i == kIterationsPerKernelCall - 1 && kOutput) {
        max_change = fabs((update - u) / u);
      }
      u = update;
      u_bar_shared[shared_mem_index] = u_bar;
    }
    if (check_convergence) {
      float max_change_reduced = BlockReduceFloat(float_storage).Reduce(max_change, cub::Max());
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_m(0, blockIdx.x) = max_change_reduced;
      }
    }
    __syncthreads();
  }
  
  // write outputs back into global memory
  if (kOutput) {
    if (kPrimalFlag) {
      d_u(y, x) = u;
      d_u_bar(y, x) = u_bar;
    }
    if (kDualFlag) {
      d_dualTVX(y, x) = dualTVX * 1.f / kDualIntToFloat;
      d_dualTVY(y, x) = dualTVY * 1.f / kDualIntToFloat;
    }
  }
}

int InpaintAdaptiveDepthMapCUDA(
    cudaStream_t stream,
    int max_num_iterations,
    float max_change_rate_threshold,
    float depth_input_scaling_factor,
    bool block_adaptive,
    bool use_tv_weights,
    cudaTextureObject_t gradient_magnitude_div_sqrt2,
    cudaTextureObject_t depth_map_input,
    CUDABuffer<bool>* tv_flag,
    CUDABuffer<bool>* tv_dual_flag,
    CUDABuffer<int16_t>* tv_dual_x,
    CUDABuffer<int16_t>* tv_dual_y,
    CUDABuffer<float>* tv_u_bar,
    CUDABuffer<float>* tv_max_change,
    CUDABuffer<float>* depth_map_output,
    CUDABuffer<uint16_t>* block_coordinates,
    CUDABuffer<unsigned char>* block_activities) {
  const int width = depth_map_output->width();
  const int height = depth_map_output->height();
  const int kBlockWidth = block_adaptive ? 16 : 32;
  const int kBlockHeight = block_adaptive ? 16 : 32;
  dim3 grid_dim(cuda_util::GetBlockCount(width, kBlockWidth),
                cuda_util::GetBlockCount(height, kBlockHeight));
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  const int sm_size = block_adaptive ? kBlockWidth*kBlockHeight*sizeof(float) : 0;
  
  CUDABuffer<float>* tv_u = depth_map_output;
  
  constexpr bool kUseSingleKernel = true;
  const int kBlockOutputSizeX = kBlockWidth - 2 * kIterationsPerKernelCall;
  const int kBlockOutputSizeY = kBlockHeight - 2 * kIterationsPerKernelCall;
  grid_dim = dim3(cuda_util::GetBlockCount(width, kBlockOutputSizeX),
                  cuda_util::GetBlockCount(height, kBlockOutputSizeY));
  
  // Initialize variables.
  TVInpaintingInitializeVariablesKernel<<<grid_dim, block_dim, 0, stream>>>(
      grid_dim.x, kUseSingleKernel, depth_input_scaling_factor, depth_map_input, tv_flag->ToCUDA(), tv_dual_flag->ToCUDA(), tv_dual_x->ToCUDA(),
      tv_dual_y->ToCUDA(), tv_u->ToCUDA(), tv_u_bar->ToCUDA(), block_coordinates->ToCUDA());
  CHECK_CUDA_NO_ERROR();

  if (block_adaptive) {
    InitBlockActivationsFromInpaintRegionKernel<<<grid_dim, block_dim, 0, stream>>>(
        depth_map_input,
        width, height,
        block_activities->ToCUDA());
    CHECK_CUDA_NO_ERROR();
  }
  
  uint8_t* block_activity = new uint8_t[grid_dim.x * grid_dim.y];
  block_coordinates->DownloadPartAsync(0, grid_dim.x * grid_dim.y * sizeof(uint8_t), stream, reinterpret_cast<uint16_t*>(block_activity));
  cudaStreamSynchronize(stream);
  int active_block_count = 0;
  uint16_t* block_coordinates_cpu = new uint16_t[2 * grid_dim.x * grid_dim.y];
  for (size_t y = 0; y < grid_dim.y; ++ y) {
    for (size_t x = 0; x < grid_dim.x; ++ x) {
      if (block_activity[x + y * grid_dim.x] > 0) {
        block_coordinates_cpu[2 * active_block_count + 0] = x * (kUseSingleKernel ? kBlockOutputSizeX : kBlockWidth);
        block_coordinates_cpu[2 * active_block_count + 1] = y * (kUseSingleKernel ? kBlockOutputSizeY : kBlockHeight);
        ++ active_block_count;
      }
    }
  }
  delete[] block_activity;
  if (active_block_count == 0) {
    delete[] block_coordinates_cpu;
    return 0;
  }
  block_coordinates->UploadPartAsync(0, 2 * active_block_count * sizeof(uint16_t), stream, block_coordinates_cpu);
  float* max_change = new float[grid_dim.x * grid_dim.y];
  
  // Run optimization iterations.
  int i = 0;
  int last_convergence_check_iteration = -180;
  for (i = 0; i < max_num_iterations; i += (kUseSingleKernel ? kIterationsPerKernelCall : 1)) {
    const bool check_convergence = (i - last_convergence_check_iteration >= 200);
    
    if (kUseSingleKernel) {
      dim3 grid_dim_single_kernel(active_block_count);
      CHECK_EQ(kBlockWidth, 32);
      CHECK_EQ(kBlockHeight, 32);
      if (check_convergence) {
        TVInpaintingDualAndPrimalStepsKernel<32, 32, true, true><<<grid_dim_single_kernel, block_dim, sm_size, stream>>>(
            kHuberEpsilon,
            block_coordinates->ToCUDA(),
            tv_flag->ToCUDA(),
            tv_dual_flag->ToCUDA(),
            gradient_magnitude_div_sqrt2,
            tv_dual_x->ToCUDA(),
            tv_dual_y->ToCUDA(),
            tv_u->ToCUDA(),
            tv_u_bar->ToCUDA(),
            tv_max_change->ToCUDA());
      } else {
        TVInpaintingDualAndPrimalStepsKernel<32, 32, true, false><<<grid_dim_single_kernel, block_dim, sm_size, stream>>>(
            kHuberEpsilon,
            block_coordinates->ToCUDA(),
            tv_flag->ToCUDA(),
            tv_dual_flag->ToCUDA(),
            gradient_magnitude_div_sqrt2,
            tv_dual_x->ToCUDA(),
            tv_dual_y->ToCUDA(),
            tv_u->ToCUDA(),
            tv_u_bar->ToCUDA(),
            tv_max_change->ToCUDA());
      }
    } else {
      if (block_adaptive) {
        if (use_tv_weights) {
          TVInpaintingDualStepKernel<true,true><<<grid_dim, block_dim, sm_size, stream>>>(
              kHuberEpsilon, kCellChangeThreshold,
              tv_dual_flag->ToCUDA(),
              tv_u_bar->ToCUDA(),
              gradient_magnitude_div_sqrt2,
              tv_dual_x->ToCUDA(), tv_dual_y->ToCUDA(),
              block_activities->ToCUDA());
        } else {
          TVInpaintingDualStepKernel<false,true><<<grid_dim, block_dim, sm_size, stream>>>(
              kHuberEpsilon, kCellChangeThreshold,
              tv_dual_flag->ToCUDA(),
              tv_u_bar->ToCUDA(),
              gradient_magnitude_div_sqrt2,
              tv_dual_x->ToCUDA(), tv_dual_y->ToCUDA(),
              block_activities->ToCUDA());
        }
      } else {
        if (use_tv_weights) {
          TVInpaintingDualStepKernel<true,false><<<grid_dim, block_dim, sm_size, stream>>>(
              kHuberEpsilon, kCellChangeThreshold,
              tv_dual_flag->ToCUDA(),
              tv_u_bar->ToCUDA(),
              gradient_magnitude_div_sqrt2,
              tv_dual_x->ToCUDA(), tv_dual_y->ToCUDA(),
              block_activities->ToCUDA());
        } else {
          TVInpaintingDualStepKernel<false,false><<<grid_dim, block_dim, sm_size, stream>>>(
              kHuberEpsilon, kCellChangeThreshold,
              tv_dual_flag->ToCUDA(),
              tv_u_bar->ToCUDA(),
              gradient_magnitude_div_sqrt2,
              tv_dual_x->ToCUDA(), tv_dual_y->ToCUDA(),
              block_activities->ToCUDA());
        }
      } // if (block_adaptive)

      if (check_convergence) {
        if (block_adaptive) {
          TVInpaintingPrimalStepKernel<true,true><<<grid_dim, block_dim, sm_size, stream>>>(
              kCellChangeThreshold,
              tv_flag->ToCUDA(), tv_dual_x->ToCUDA(), tv_dual_y->ToCUDA(),
              tv_u->ToCUDA(), tv_u_bar->ToCUDA(), tv_max_change->ToCUDA(),
              block_activities->ToCUDA());
        } else {
          TVInpaintingPrimalStepKernel<true,false><<<grid_dim, block_dim, sm_size, stream>>>(
              kCellChangeThreshold,
              tv_flag->ToCUDA(), tv_dual_x->ToCUDA(), tv_dual_y->ToCUDA(),
              tv_u->ToCUDA(), tv_u_bar->ToCUDA(), tv_max_change->ToCUDA(),
              block_activities->ToCUDA());
        }
      } else {
        if (block_adaptive) {
          TVInpaintingPrimalStepKernel<false,true><<<grid_dim, block_dim, sm_size, stream>>>(
              kCellChangeThreshold,
              tv_flag->ToCUDA(), tv_dual_x->ToCUDA(), tv_dual_y->ToCUDA(),
              tv_u->ToCUDA(), tv_u_bar->ToCUDA(), CUDABuffer_<float>(),
              block_activities->ToCUDA());
        } else {
          TVInpaintingPrimalStepKernel<false,false><<<grid_dim, block_dim, sm_size, stream>>>(
              kCellChangeThreshold,
              tv_flag->ToCUDA(), tv_dual_x->ToCUDA(), tv_dual_y->ToCUDA(),
              tv_u->ToCUDA(), tv_u_bar->ToCUDA(), CUDABuffer_<float>(),
              block_activities->ToCUDA());
        }
      } // if (check_convergence)
    }  // if (kUseSingleKernel)

    if (check_convergence) {
      tv_max_change->DownloadPartAsync(0, active_block_count * sizeof(float), stream, max_change);
      cudaStreamSynchronize(stream);
      int new_active_block_count = 0;
      for (int j = 0, end = active_block_count; j < end; j ++) {
        if (max_change[j] > max_change_rate_threshold) {
          // block_coordinates_cpu[2 * new_active_block_count + 0] = block_coordinates_cpu[2 * j + 0];
          // block_coordinates_cpu[2 * new_active_block_count + 1] = block_coordinates_cpu[2 * j + 1];
          ++ new_active_block_count;
        }
      }
      //LOG(INFO) << "[" << i << "] Active blocks: " << active_block_count << " -> " << new_active_block_count;
      //LOG(INFO) << "max_change_rate: " << max_change_rate << " / " << max_change_rate_threshold;
      if (new_active_block_count == 0) {
        break;
      }
      //active_block_count = new_active_block_count;
      //block_coordinates->UploadPartAsync(0, 2 * active_block_count * sizeof(uint16_t), stream, block_coordinates_cpu);
      last_convergence_check_iteration = i;
    } // if (check_convergence)
  } // for (i = 0; i < max_num_iterations; ++i)
  delete[] max_change;
  delete[] block_coordinates_cpu;
  CHECK_CUDA_NO_ERROR();
  
  if (i < max_num_iterations) {
    LOG(INFO) << "TV converged after iteration: " << i;
  } else {
    LOG(WARNING) << "TV used maximum iteration count: " << i;
  }
  return i;
}

int InpaintDepthMapCUDA(
    cudaStream_t stream,
    InpaintingMode inpainting_mode,
    bool use_tv_weights,
    int max_num_iterations,
    float max_change_rate_threshold,
    float depth_input_scaling_factor,
    cudaTextureObject_t gradient_magnitude_div_sqrt2,
    cudaTextureObject_t depth_map_input,
    CUDABuffer<bool>* tv_flag,
    CUDABuffer<bool>* tv_dual_flag,
    CUDABuffer<int16_t>* tv_dual_x,
    CUDABuffer<int16_t>* tv_dual_y,
    CUDABuffer<float>* tv_u_bar,
    CUDABuffer<float>* tv_max_change,
    CUDABuffer<float>* depth_map_output,
    CUDABuffer<uint16_t>* block_coordinates,
    CUDABuffer<unsigned char>* block_activities) {
  switch(inpainting_mode) {
    case kIMClassic:
      return InpaintAdaptiveDepthMapCUDA(
          stream, 
          max_num_iterations, max_change_rate_threshold, depth_input_scaling_factor,
          false, use_tv_weights,
          gradient_magnitude_div_sqrt2, depth_map_input,
          tv_flag, tv_dual_flag, tv_dual_x, tv_dual_y, tv_u_bar, tv_max_change,
          depth_map_output, block_coordinates, block_activities);
    case kIMAdaptive:
      return InpaintAdaptiveDepthMapCUDA(
          stream, 
          max_num_iterations, max_change_rate_threshold, depth_input_scaling_factor,
          true, use_tv_weights,
          gradient_magnitude_div_sqrt2, depth_map_input,
          tv_flag, tv_dual_flag, tv_dual_x, tv_dual_y, tv_u_bar, tv_max_change,
          depth_map_output, block_coordinates, block_activities);
    default:
      return InpaintAdaptiveDepthMapCUDA(
          stream, 
          max_num_iterations, max_change_rate_threshold, depth_input_scaling_factor,
          false, use_tv_weights,
          gradient_magnitude_div_sqrt2, depth_map_input,
          tv_flag, tv_dual_flag, tv_dual_x, tv_dual_y, tv_u_bar, tv_max_change,
          depth_map_output, block_coordinates, block_activities);
  } // switch(inpainting_mode)
}

__global__ void TVInpaintingInitializeVariablesKernel(
    int grid_dim_x,
    CUDABuffer_<uchar4> input,
    CUDABuffer_<bool> tv_flag,
    CUDABuffer_<bool> tv_dual_flag,
    CUDABuffer_<float> tv_dual_x_r,
    CUDABuffer_<float> tv_dual_x_g,
    CUDABuffer_<float> tv_dual_x_b,
    CUDABuffer_<float> tv_dual_y_r,
    CUDABuffer_<float> tv_dual_y_g,
    CUDABuffer_<float> tv_dual_y_b,
    CUDABuffer_<float> tv_u_r,
    CUDABuffer_<float> tv_u_g,
    CUDABuffer_<float> tv_u_b,
    CUDABuffer_<float> tv_u_bar_r,
    CUDABuffer_<float> tv_u_bar_g,
    CUDABuffer_<float> tv_u_bar_b,
    CUDABuffer_<uint16_t> block_coordinates) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int width = tv_u_r.width();
  const int height = tv_u_r.height();
  bool thread_is_active = false;
  if (x < width && y < height) {
    tv_dual_x_r(y, x) = 0.f;
    tv_dual_x_g(y, x) = 0.f;
    tv_dual_x_b(y, x) = 0.f;
    tv_dual_y_r(y, x) = 0.f;
    tv_dual_y_g(y, x) = 0.f;
    tv_dual_y_b(y, x) = 0.f;
    const uchar4 f_input = input(y, x);
    tv_flag(y, x) = (f_input.w == 0);
    thread_is_active =
        (f_input.w == 0 ||
         (x > 0 && input(y, x - 1).w == 0) ||
         (y > 0 && input(y - 1, x).w == 0) ||
         (x < input.width() - 1 && input(y, x + 1).w == 0) ||
         (y < input.height() - 1 && input(y + 1, x).w == 0));
    tv_dual_flag(y, x) = thread_is_active;
    const float3 f_input_float = make_float3(
        (1.f / 255.f) * f_input.x,
        (1.f / 255.f) * f_input.y,
        (1.f / 255.f) * f_input.z);
    tv_u_r(y, x) = f_input_float.x;
    tv_u_g(y, x) = f_input_float.y;
    tv_u_b(y, x) = f_input_float.z;
    tv_u_bar_r(y, x) = f_input_float.x;
    tv_u_bar_g(y, x) = f_input_float.y;
    tv_u_bar_b(y, x) = f_input_float.z;
  }
  
  typedef cub::BlockReduce<
      int, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 32> BlockReduceInt;
  __shared__ typename BlockReduceInt::TempStorage int_storage;
  int num_active_threads = BlockReduceInt(int_storage).Reduce(thread_is_active ? 1 : 0, cub::Sum());
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    reinterpret_cast<uint8_t*>(block_coordinates.address())[blockIdx.x + blockIdx.y * grid_dim_x] = (num_active_threads > 0) ? 1 : 0;
  }
}

// performs primal update and extrapolation step:
// u^{k+1} = u^k + tau* div(p^{k+1})
// \bar{u}^{k+1} = 2*u^{k+1} - u^k
template<bool check_convergence>
__global__ void TVInpaintingPrimalStepKernel(
    CUDABuffer_<uint16_t> block_coordinates,
    CUDABuffer_<bool> tv_flag,
    CUDABuffer_<float> d_dualTVX_r,
    CUDABuffer_<float> d_dualTVX_g,
    CUDABuffer_<float> d_dualTVX_b,
    CUDABuffer_<float> d_dualTVY_r,
    CUDABuffer_<float> d_dualTVY_g,
    CUDABuffer_<float> d_dualTVY_b,
    CUDABuffer_<float> d_u_r,
    CUDABuffer_<float> d_u_g,
    CUDABuffer_<float> d_u_b,
    CUDABuffer_<float> d_u_bar_r,
    CUDABuffer_<float> d_u_bar_g,
    CUDABuffer_<float> d_u_bar_b,
    CUDABuffer_<float> d_m) {
  const int x = block_coordinates(0, 2 * blockIdx.x + 0) + threadIdx.x;
  const int y = block_coordinates(0, 2 * blockIdx.x + 1) + threadIdx.y;
  
  typedef cub::BlockReduce<
      float, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 32> BlockReduceFloat;
  __shared__ typename BlockReduceFloat::TempStorage float_storage;
  
  // only update within the inpainting region (f == 0)
  float max_change = 0;
  if (x < d_u_r.width() && y < d_u_r.height() && tv_flag(y, x)) {
    // this will accumulate the update step for the primal variable
    float3 update = make_float3(0, 0, 0);
    // this will accumulate all row entries of the linear operator for the preconditioned step width
    float rowSum = 0;

    // compute divergence update of dualTV - Neumann boundary conditions,
    // keep track of row sum for preconditioning
    update.x += d_dualTVX_r(y, x) + d_dualTVY_r(y, x);
    update.y += d_dualTVX_g(y, x) + d_dualTVY_g(y, x);
    update.z += d_dualTVX_b(y, x) + d_dualTVY_b(y, x);
    rowSum += 2;
    if (x > 0) {
      update.x -= d_dualTVX_r(y, x - 1);
      update.y -= d_dualTVX_g(y, x - 1);
      update.z -= d_dualTVX_b(y, x - 1);
      rowSum++;
    }
    if (y > 0) {
      update.x -= d_dualTVY_r(y - 1, x);
      update.y -= d_dualTVY_g(y - 1, x);
      update.z -= d_dualTVY_b(y - 1, x);
      rowSum++;
    }

    constexpr float kPrimalStepWidth = 1.f;
    const float tau = kPrimalStepWidth / rowSum;
    const float3 u = make_float3(d_u_r(y, x), d_u_g(y, x), d_u_b(y, x));

    update = u + tau * update;

    d_u_r(y, x) = update.x;
    d_u_g(y, x) = update.y;
    d_u_b(y, x) = update.z;
    float3 u_bar = 2 * update - u;
    d_u_bar_r(y, x) = u_bar.x;
    d_u_bar_g(y, x) = u_bar.y;
    d_u_bar_b(y, x) = u_bar.z;
    
    if (check_convergence) {
      max_change = max(max(fabs((update.x - u.x) / u.x),
                           fabs((update.y - u.y) / u.y)),
                       fabs((update.z - u.z) / u.z));
    }
  }
  
  if (check_convergence) {
    float max_change_reduced = BlockReduceFloat(float_storage).Reduce(max_change, cub::Max());
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      d_m(0, blockIdx.x) = max_change_reduced;
    }
  }
}

// performs dual update step
// p^{k=1} = \Pi_{|p|<=g} [ p^k + \sigma * \nabla \bar{u}^k ]
template<bool use_weighting>
__global__ void TVInpaintingDualStepKernel(
    CUDABuffer_<uint16_t> block_coordinates,
    CUDABuffer_<bool> tv_dual_flag,
    CUDABuffer_<float> d_u_r,
    CUDABuffer_<float> d_u_g,
    CUDABuffer_<float> d_u_b,
    cudaTextureObject_t d_tvWeight,
    CUDABuffer_<float> d_dualTVX_r,
    CUDABuffer_<float> d_dualTVX_g,
    CUDABuffer_<float> d_dualTVX_b,
    CUDABuffer_<float> d_dualTVY_r,
    CUDABuffer_<float> d_dualTVY_g,
    CUDABuffer_<float> d_dualTVY_b) {
  const int x = block_coordinates(0, 2 * blockIdx.x + 0) + threadIdx.x;
  const int y = block_coordinates(0, 2 * blockIdx.x + 1) + threadIdx.y;
  
  if (x < d_u_r.width() && y < d_u_r.height() && tv_dual_flag(y, x)) {
    const float dualStepWidth = 1.0f;
    const float HUBER_EPS = 0.01f;
    const float huberFactor = 1.0f / (1.0f + dualStepWidth * 0.5f * HUBER_EPS);
    
    // update using the gradient of u
    const float3 u = make_float3(d_u_r(y, x), d_u_g(y, x), d_u_b(y, x));
    constexpr float kDualStepWidth = 1.f;
    
    float3 u_plusx_minus_u = make_float3(0, 0, 0);
    if (x < d_u_r.width() - 1) {
      u_plusx_minus_u = make_float3(d_u_r(y, x + 1), d_u_g(y, x + 1), d_u_b(y, x + 1)) - u;
    }
    const float3 dualTVX = make_float3(d_dualTVX_r(y, x), d_dualTVX_g(y, x), d_dualTVX_b(y, x));
    
    float3 u_plusy_minus_u = make_float3(0, 0, 0);
    if (y < d_u_r.height() - 1) {
      u_plusy_minus_u = make_float3(d_u_r(y + 1, x), d_u_g(y + 1, x), d_u_b(y + 1, x)) - u;
    }
    const float3 dualTVY = make_float3(d_dualTVY_r(y, x), d_dualTVY_g(y, x), d_dualTVY_b(y, x));
    
    float3 resultX =
        huberFactor * (dualTVX + kDualStepWidth * 0.5f * u_plusx_minus_u);
    float3 resultY =
        huberFactor * (dualTVY + kDualStepWidth * 0.5f * u_plusy_minus_u);
    
    // project onto the g-unit ball
    float3 denom;
    if (use_weighting) {
      // Optimization: remove 1 / weight and turn division by weight below into multiplication.
      const float weight = /*1.f /*/ (1.f + tex2D<uchar>(d_tvWeight, x, y) * kSqrt2 / 255.f);
      denom.x = max(1.0f, hypotf(resultX.x, resultY.x) * weight);
      denom.y = max(1.0f, hypotf(resultX.y, resultY.y) * weight);
      denom.z = max(1.0f, hypotf(resultX.z, resultY.z) * weight);
    } else {
      denom.x = max(1.0f, hypotf(resultX.x, resultY.x));
      denom.y = max(1.0f, hypotf(resultX.y, resultY.y));
      denom.z = max(1.0f, hypotf(resultX.z, resultY.z));
    }
    resultX /= denom;
    resultY /= denom;

    // write result back into global memory
    d_dualTVX_r(y, x) = resultX.x;
    d_dualTVX_g(y, x) = resultX.y;
    d_dualTVX_b(y, x) = resultX.z;
    d_dualTVY_r(y, x) = resultY.x;
    d_dualTVY_g(y, x) = resultY.y;
    d_dualTVY_b(y, x) = resultY.z;
  }
}

int InpaintImageCUDA(
    cudaStream_t stream,
    int max_num_iterations,
    float max_change_rate_threshold,
    cudaTextureObject_t gradient_magnitude_div_sqrt2,
    const CUDABuffer<uchar4>& input,
    CUDABuffer<bool>* tv_flag,
    CUDABuffer<bool>* tv_dual_flag,
    CUDABuffer<float>* tv_dual_x_r,
    CUDABuffer<float>* tv_dual_x_g,
    CUDABuffer<float>* tv_dual_x_b,
    CUDABuffer<float>* tv_dual_y_r,
    CUDABuffer<float>* tv_dual_y_g,
    CUDABuffer<float>* tv_dual_y_b,
    CUDABuffer<float>* tv_u_bar_r,
    CUDABuffer<float>* tv_u_bar_g,
    CUDABuffer<float>* tv_u_bar_b,
    CUDABuffer<float>* tv_max_change,
    CUDABuffer<float>* output_r,
    CUDABuffer<float>* output_g,
    CUDABuffer<float>* output_b,
    CUDABuffer<uint16_t>* block_coordinates) {
  const int width = output_r->width();
  const int height = output_r->height();
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 grid_dim(cuda_util::GetBlockCount(width, kBlockWidth),
                cuda_util::GetBlockCount(height, kBlockHeight));
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  
  CUDABuffer<float>* tv_u_r = output_r;
  CUDABuffer<float>* tv_u_g = output_g;
  CUDABuffer<float>* tv_u_b = output_b;
  
  // Initialize variables.
  TVInpaintingInitializeVariablesKernel<<<grid_dim, block_dim, 0, stream>>>(
      grid_dim.x,
      input.ToCUDA(),
      tv_flag->ToCUDA(),
      tv_dual_flag->ToCUDA(),
      tv_dual_x_r->ToCUDA(),
      tv_dual_x_g->ToCUDA(),
      tv_dual_x_b->ToCUDA(),
      tv_dual_y_r->ToCUDA(),
      tv_dual_y_g->ToCUDA(),
      tv_dual_y_b->ToCUDA(),
      tv_u_r->ToCUDA(),
      tv_u_g->ToCUDA(),
      tv_u_b->ToCUDA(),
      tv_u_bar_r->ToCUDA(),
      tv_u_bar_g->ToCUDA(),
      tv_u_bar_b->ToCUDA(),
      block_coordinates->ToCUDA());
  CHECK_CUDA_NO_ERROR();
  
  uint8_t* block_activity = new uint8_t[grid_dim.x * grid_dim.y];
  block_coordinates->DownloadPartAsync(0, grid_dim.x * grid_dim.y * sizeof(uint8_t), stream, reinterpret_cast<uint16_t*>(block_activity));
  cudaStreamSynchronize(stream);
  int active_block_count = 0;
  uint16_t* block_coordinates_cpu = new uint16_t[2 * grid_dim.x * grid_dim.y];
  for (size_t y = 0; y < grid_dim.y; ++ y) {
    for (size_t x = 0; x < grid_dim.x; ++ x) {
      if (block_activity[x + y * grid_dim.x] > 0) {
        block_coordinates_cpu[2 * active_block_count + 0] = x * kBlockWidth;
        block_coordinates_cpu[2 * active_block_count + 1] = y * kBlockHeight;
        ++ active_block_count;
      }
    }
  }
  delete[] block_activity;
  if (active_block_count == 0) {
    return 0;
  }
  block_coordinates->UploadPartAsync(0, 2 * active_block_count * sizeof(uint16_t), stream, block_coordinates_cpu);
  float* max_change = new float[grid_dim.x * grid_dim.y];
  
  // Run optimization iterations.
  int i = 0;
  int last_convergence_check_iteration = -180;
  for (i = 0; i < max_num_iterations; i += 1) {
    // TODO: HACK: Minimum iteration count is necessary since it exits too early in some cases
    const bool check_convergence = (i - last_convergence_check_iteration >= 200) /*&& (i >= 500)*/;
    dim3 grid_dim_active(active_block_count);
    
    TVInpaintingDualStepKernel<true><<<grid_dim_active, block_dim, 0, stream>>>(
        block_coordinates->ToCUDA(),
        tv_dual_flag->ToCUDA(),
        tv_u_bar_r->ToCUDA(),
        tv_u_bar_g->ToCUDA(),
        tv_u_bar_b->ToCUDA(),
        gradient_magnitude_div_sqrt2,
        tv_dual_x_r->ToCUDA(),
        tv_dual_x_g->ToCUDA(),
        tv_dual_x_b->ToCUDA(),
        tv_dual_y_r->ToCUDA(),
        tv_dual_y_g->ToCUDA(),
        tv_dual_y_b->ToCUDA());

    if (check_convergence) {
      TVInpaintingPrimalStepKernel<true><<<grid_dim_active, block_dim, 0, stream>>>(
          block_coordinates->ToCUDA(),
          tv_flag->ToCUDA(),
          tv_dual_x_r->ToCUDA(),
          tv_dual_x_g->ToCUDA(),
          tv_dual_x_b->ToCUDA(),
          tv_dual_y_r->ToCUDA(),
          tv_dual_y_g->ToCUDA(),
          tv_dual_y_b->ToCUDA(),
          tv_u_r->ToCUDA(),
          tv_u_g->ToCUDA(),
          tv_u_b->ToCUDA(),
          tv_u_bar_r->ToCUDA(),
          tv_u_bar_g->ToCUDA(),
          tv_u_bar_b->ToCUDA(),
          tv_max_change->ToCUDA());
    } else {
      TVInpaintingPrimalStepKernel<false><<<grid_dim_active, block_dim, 0, stream>>>(
          block_coordinates->ToCUDA(),
          tv_flag->ToCUDA(),
          tv_dual_x_r->ToCUDA(),
          tv_dual_x_g->ToCUDA(),
          tv_dual_x_b->ToCUDA(),
          tv_dual_y_r->ToCUDA(),
          tv_dual_y_g->ToCUDA(),
          tv_dual_y_b->ToCUDA(),
          tv_u_r->ToCUDA(),
          tv_u_g->ToCUDA(),
          tv_u_b->ToCUDA(),
          tv_u_bar_r->ToCUDA(),
          tv_u_bar_g->ToCUDA(),
          tv_u_bar_b->ToCUDA(),
          CUDABuffer_<float>());
    }
    
    if (check_convergence) {
      tv_max_change->DownloadPartAsync(0, active_block_count * sizeof(float), stream, max_change);
      cudaStreamSynchronize(stream);
      int new_active_block_count = 0;
      for (int j = 0, end = active_block_count; j < end; j ++) {
        if (max_change[j] > max_change_rate_threshold) {
          // block_coordinates_cpu[2 * new_active_block_count + 0] = block_coordinates_cpu[2 * j + 0];
          // block_coordinates_cpu[2 * new_active_block_count + 1] = block_coordinates_cpu[2 * j + 1];
          ++ new_active_block_count;
        }
      }
      //LOG(INFO) << "[" << i << "] Active blocks: " << active_block_count << " -> " << new_active_block_count;
      //LOG(INFO) << "max_change_rate: " << max_change_rate << " / " << max_change_rate_threshold;
      if (new_active_block_count == 0) {
        break;
      }
      //active_block_count = new_active_block_count;
      //block_coordinates->UploadPartAsync(0, 2 * active_block_count * sizeof(uint16_t), stream, block_coordinates_cpu);
      last_convergence_check_iteration = i;
    } // if (check_convergence)
  }
  delete[] max_change;
  delete[] block_coordinates_cpu;
  CHECK_CUDA_NO_ERROR();
  
  if (i < max_num_iterations) {
    LOG(INFO) << "Color TV converged after iteration: " << i;
  } else {
    LOG(WARNING) << "Color TV used maximum iteration count: " << i;
  }
  return i;
}

} // namespace view_correction
