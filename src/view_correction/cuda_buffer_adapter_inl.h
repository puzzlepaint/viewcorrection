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

#ifndef VIEW_CORRECTION_CUDA_BUFFER_ADAPTER_INL_H_
#define VIEW_CORRECTION_CUDA_BUFFER_ADAPTER_INL_H_

#include <glog/logging.h>

#include "view_correction/cuda_util.h"

namespace view_correction {

template <typename T>
CUDABufferAdapter_<T>::CUDABufferAdapter_(CUDABuffer<T>* buffer)
    : buffer_(buffer) {}

template <typename T>
void CUDABufferAdapter_<T>::DebugUpload(
    const cv::Mat_<T>& image) {
  buffer_->DebugUploadPitched(image.step,
                              reinterpret_cast<const T*>(image.data));
}

template <typename T>
void CUDABufferAdapter_<T>::UploadAsync(
    cudaStream_t stream, const cv::Mat_<T>& image) {
  buffer_->UploadPitchedAsync(stream, image.step,
                              reinterpret_cast<const T*>(image.data));
}

template <typename T>
void CUDABufferAdapter_<T>::UploadRectAsync(int src_y, int src_x, int height, int width,
                      int dest_y, int dest_x, cudaStream_t stream,
                      const cv::Mat_<T>* image) {
  CHECK_NOTNULL(image);
  CUDA_CHECKED_CALL(cudaMemcpy2DAsync(
      reinterpret_cast<T*>(reinterpret_cast<int8_t*>(buffer_->ToCUDA().address()) +
                           dest_y * buffer_->ToCUDA().pitch()) + dest_x,
      buffer_->ToCUDA().pitch(),
      image->data + src_y * image->step + src_x * sizeof(T),
      image->step,
      width * sizeof(T), height, cudaMemcpyHostToDevice, stream));
}

template <typename T>
void CUDABufferAdapter_<T>::DebugDownload(cv::Mat_<T>* image)
    const {
  buffer_->DebugDownloadPitched(
      image->step,
      reinterpret_cast<T*>(image->data));
}

template <typename T>
void CUDABufferAdapter_<T>::DownloadAsync(
    cudaStream_t stream, cv::Mat_<T>* image) const {
  buffer_->DownloadPitchedAsync(stream, image->step,
                                reinterpret_cast<T*>(image->data));
}

template <typename T>
void CUDABufferAdapter_<T>::DownloadRectAsync(
    int src_y, int src_x, int height, int width,
    int dest_y, int dest_x, cudaStream_t stream,
    cv::Mat_<T>* image) const {
  CHECK_NOTNULL(image);
  CUDA_CHECKED_CALL(cudaMemcpy2DAsync(
      image->data + dest_y * image->step + dest_x * sizeof(T), image->step,
      reinterpret_cast<T*>(reinterpret_cast<int8_t*>(buffer_->ToCUDA().address()) +
                           src_y * buffer_->ToCUDA().pitch()) +
          src_x,
      buffer_->ToCUDA().pitch(), width * sizeof(T), height, cudaMemcpyDeviceToHost, stream));
}

}  // namespace view_correction

#endif  // VIEW_CORRECTION_CUDA_BUFFER_ADAPTER_INL_H_
