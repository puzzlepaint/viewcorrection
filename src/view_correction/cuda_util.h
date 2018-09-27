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

#ifndef VIEW_CORRECTION_CUDA_UTIL_H_
#define VIEW_CORRECTION_CUDA_UTIL_H_

#include <cuda_runtime.h>
#ifndef __CUDACC__
#include <Eigen/Core>
#endif
// #include <glog/logging.h>

#define CUDA_CHECKED_CALL(cuda_call)                                \
  do {                                                              \
    cudaError error = (cuda_call);                                  \
    if (cudaSuccess != error) {                                     \
      LOG(FATAL) << "Cuda Error: " << cudaGetErrorString(error);    \
    }                                                               \
  } while(false)

#define CHECK_CUDA_NO_ERROR()                                       \
  do {                                                              \
    cudaError error = cudaGetLastError();                           \
    if (cudaSuccess != error) {                                     \
      LOG(FATAL) << "Cuda Error: " << cudaGetErrorString(error);    \
    }                                                               \
  } while(false)

namespace view_correction {

struct CUDAMatrix3x4 {
__host__ __device__ CUDAMatrix3x4() {}
  
#ifndef __CUDACC__
  template <typename Derived> __host__ __device__  explicit
  CUDAMatrix3x4(const Eigen::MatrixBase<Derived>& matrix) {
    row0.x = matrix(0, 0);
    row0.y = matrix(0, 1);
    row0.z = matrix(0, 2);
    row0.w = matrix(0, 3);
    row1.x = matrix(1, 0);
    row1.y = matrix(1, 1);
    row1.z = matrix(1, 2);
    row1.w = matrix(1, 3);
    row2.x = matrix(2, 0);
    row2.y = matrix(2, 1);
    row2.z = matrix(2, 2);
    row2.w = matrix(2, 3);
  }
#endif
  
  float4 row0;
  float4 row1;
  float4 row2;
};

namespace cuda_util {

// Returns the required number of CUDA blocks to cover a given domain size,
// given a specific block size.
inline int GetBlockCount(int domain_size, int block_size) {
  //const int div = domain_size / block_size;
  //return (domain_size % block_size == 0) ? div : (div + 1);
  return (domain_size+block_size-1) / block_size; // avoid costly mod-operator
}

}  // namespace cuda_util

}  // namespace view_correction

#endif  // VIEW_CORRECTION_CUDA_UTIL_H_
