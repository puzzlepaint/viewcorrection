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

#ifndef VIEW_CORRECTION_CUDA_BUFFER_ADAPTER_H_
#define VIEW_CORRECTION_CUDA_BUFFER_ADAPTER_H_

#include "view_correction/cuda_buffer.h"

namespace view_correction {

// Interfaces CUDABuffer with classes using C++11.
// This could not be done directly because CUDA version 6.0
// does not support C++11.
// TODO: Since CUDA 6.0 is outdated now, could merge this into the main CUDABuffer class.
template <typename T>
class CUDABufferAdapter_ {
 public:
  explicit CUDABufferAdapter_(CUDABuffer<T>* buffer);

  // Uploads the data from the cv::Mat_ to the device buffer.
  // For use in debug code only, otherwise use an async version.
  void DebugUpload(const cv::Mat_<T>& image);
  
  // Uploads the data from the cv::Mat_ asynchronously to the
  // device buffer.
  void UploadAsync(cudaStream_t stream,
                   const cv::Mat_<T>& image);
  
  // Uploads a rectangular area of the image to the device buffer.
  void UploadRectAsync(int src_y, int src_x, int height, int width,
                       int dest_y, int dest_x, cudaStream_t stream,
                       const cv::Mat_<T>* image);

  // Downloads the device buffer data to the cv::Mat_.
  // For use in debug code only, otherwise use an async version.
  void DebugDownload(cv::Mat_<T>* image) const;
  
  // Downloads the device buffer data asynchronously to the
  // cv::Mat_.
  void DownloadAsync(cudaStream_t stream,
                     cv::Mat_<T>* image) const;
  
  // Downloads a rectangular area of the device buffer to the
  // cv::Mat_.
  void DownloadRectAsync(int src_y, int src_x, int height, int width,
                         int dest_y, int dest_x, cudaStream_t stream,
                         cv::Mat_<T>* image) const;

 private:
  CUDABuffer<T>* buffer_;
};

// Automatic template deduction applies to functions but not classes,
// so use a function to allocate the class more comfortably.
// Use for example as CUDABufferAdapter(buffer).Function(...);
template <typename T>
CUDABufferAdapter_<T> CUDABufferAdapter(CUDABuffer<T>* buffer) {
  return CUDABufferAdapter_<T>(buffer);
}

}  // namespace view_correction

#include "view_correction/cuda_buffer_adapter_inl.h"

#endif  // VIEW_CORRECTION_CUDA_BUFFER_ADAPTER_H_
