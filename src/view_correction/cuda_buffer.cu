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

#include "view_correction/cuda_buffer.cuh"

#include <curand_kernel.h>

#include "view_correction/cuda_util.h"

namespace view_correction {

template<typename T>
__global__ void CUDABufferClearKernel(CUDABuffer_<T> buffer, T value) {
  // NOTE: It may be faster to ensure writing to at least
  //       2 to 4 words in every thread if sizeof(T) is small.
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < buffer.width() && y < buffer.height()) {
    buffer(y, x) = value;
  }
}

template<typename T>
void CUDABuffer_<T>::Clear(T value, cudaStream_t stream) {
  constexpr int kTileWidth = 32;
  constexpr int kTileHeight = 8;
  dim3 grid_dim(cuda_util::GetBlockCount(width(), kTileWidth),
               cuda_util::GetBlockCount(height(), kTileHeight));
  dim3 block_dim(kTileWidth, kTileHeight);
  CUDABufferClearKernel<<<grid_dim, block_dim, 0, stream>>>(*this, value);
}

template<typename T>
__global__ void CUDABufferSetToKernel(CUDABuffer_<T> buffer,
                                      cudaTextureObject_t texture) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < buffer.width() && y < buffer.height()) {
    buffer(y, x) = tex2D<T>(texture, x, y);
  }
}

template<>
__global__ void CUDABufferSetToKernel(CUDABuffer_<float3> /*buffer*/,
                                      cudaTextureObject_t /*texture*/) {
  // Not supported, but needed to make this CUDABuffer_ variant compile at all.
}

template<typename T>
void CUDABuffer_<T>::SetTo(cudaTextureObject_t texture, cudaStream_t stream) {
  constexpr int kTileWidth = 32;
  constexpr int kTileHeight = 8;
  dim3 grid_dim(cuda_util::GetBlockCount(width(), kTileWidth),
               cuda_util::GetBlockCount(height(), kTileHeight));
  dim3 block_dim(kTileWidth, kTileHeight);
  CUDABufferSetToKernel<<<grid_dim, block_dim, 0, stream>>>(*this, texture);
}

// Avoid compilation of SetTo() for type curandState (which does not work as
// there is no suited tex2D() overload) by declaring but not defining a
// specialization for it.
template<>
void CUDABuffer_<curandState>::SetTo(cudaTextureObject_t texture, cudaStream_t stream);

// Precompile all variants of CUDABuffer_ that are used.
// The alternative would be to move above functions to an inl header,
// but then all files including cuda-buffer.h (and thus this inl header)
// would need to be compiled by nvcc.
// The float3 / int3 variants do not compile as corresponding tex2D<>()
// variants are missing for the SetTo() kernel.
template class CUDABuffer_<float>;
template class CUDABuffer_<float2>;
template class CUDABuffer_<float3>;
template class CUDABuffer_<float4>;
template class CUDABuffer_<int>;
template class CUDABuffer_<uint8_t>;
template class CUDABuffer_<uint16_t>;
template class CUDABuffer_<char2>;
template class CUDABuffer_<char4>;
template class CUDABuffer_<uchar4>;
template class CUDABuffer_<curandState>;

}  // namespace view_correction
