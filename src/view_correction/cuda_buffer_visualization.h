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

#ifndef VIEW_CORRECTION_CUDA_BUFFER_VISUALIZATION_H_
#define VIEW_CORRECTION_CUDA_BUFFER_VISUALIZATION_H_

#include "view_correction/cuda_buffer.h"

namespace view_correction {

// Utility class which can be used for debug visualizations of
// CUDABuffer objects.
template <typename T>
class CUDABufferVisualization_ {
 public:
  explicit CUDABufferVisualization_(const CUDABuffer<T>& buffer);

  // Downloads the buffer and displays it in an OpenCV window.
  void Display(float min_value, float max_value, const char* window_title,
               bool wait_key, const char* write_path) const;

  // Downloads the buffer and displays it as an inverse depth map with JET
  // colors in an OpenCV window.
  void DisplayInvDepthMap(float min_depth, float max_depth,
                          const char* window_title, bool wait_key, const char* write_path) const;
  void DisplayDepthMap(float min_depth, float max_depth,
                       const char* window_title, bool wait_key, const char* write_path) const;

 private:
  const CUDABuffer<T>& buffer_;
};

// Automatic template deduction applies to functions but not classes,
// so use a function to allocate the class more comfortably.
// Use for example as CUDABufferVisualization(buffer).Display(...);
template <typename T>
CUDABufferVisualization_<T> CUDABufferVisualization(
    const CUDABuffer<T>& buffer) {
  return CUDABufferVisualization_<T>(buffer);
}

}  // namespace view_correction

#include "view_correction/cuda_buffer_visualization_inl.h"

#endif  // VIEW_CORRECTION_CUDA_BUFFER_VISUALIZATION_H_
