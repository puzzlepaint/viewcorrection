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

#ifndef VIEW_CORRECTION_CUDA_BUFFER_VISUALIZATION_INL_H_
#define VIEW_CORRECTION_CUDA_BUFFER_VISUALIZATION_INL_H_

#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>

#include "view_correction/cuda_buffer_adapter.h"
#include "view_correction/util.h"

namespace view_correction {

template <typename T>
CUDABufferVisualization_<T>::CUDABufferVisualization_(
    const CUDABuffer<T>& buffer)
    : buffer_(buffer) {}

template <typename T>
void CUDABufferVisualization_<T>::Display(float min_value, float max_value,
                                          const char* window_title,
                                          bool wait_key, const char* write_path) const {
#if defined(ANDROID)
  (void)wait_key;
  (void)min_value;
  (void)max_value;
  (void)window_title;
  (void)write_path;
  LOG(FATAL) << "Not supported on Android.";
#else
  cv::Mat_<T> mat(buffer_.height(), buffer_.width());
  buffer_.Download(&mat);
  cv::Mat_<float> float_mat;
  mat.convertTo(float_mat, CV_32F);
  float_mat -= min_value;
  float_mat /= (max_value - min_value);

  if (write_path) {
    cv::Mat_<float> scaled_float_mat = std::numeric_limits<uint16_t>::max() * float_mat;
    cv::Mat_<uint16_t> uint16_t_mat;
    scaled_float_mat.convertTo(uint16_t_mat, CV_16UC1);
    cv::imwrite(write_path, uint16_t_mat);
  } else {
    cv::imshow(window_title, float_mat);
  }
  if (wait_key) {
    cv::waitKey(0);
  }
#endif
}

template <typename T>
void CUDABufferVisualization_<T>::DisplayInvDepthMap(float min_depth,
                                                     float max_depth,
                                                     const char* window_title,
                                                     bool wait_key, const char* write_path) const {
#if defined(ANDROID)
  (void)wait_key;
  (void)min_depth;
  (void)max_depth;
  (void)window_title;
  (void)write_path;
  LOG(FATAL) << "Not supported on Android.";
#else
  cv::Mat_<float> buffer_cpu(buffer_.height(), buffer_.width());
  CUDABufferAdapter(const_cast<CUDABuffer<T>*>(&buffer_))
      .DebugDownload(&buffer_cpu);
  for (int y = 0; y < buffer_cpu.rows; ++y) {
    for (int x = 0; x < buffer_cpu.cols; ++x) {
      if (buffer_cpu(y, x) != 0) {
        buffer_cpu(y, x) = 1.f / buffer_cpu(y, x);
      }
    }
  }
  cv::Mat mat =
      util::GetInvDepthColoredDepthmapMat(buffer_cpu, min_depth, max_depth);
  if (write_path) {
    cv::imwrite(write_path, mat);
  } else {
    cv::imshow(window_title, mat);
  }
  if (wait_key) {
    cv::waitKey(0);
  }
#endif
}

template <typename T>
void CUDABufferVisualization_<T>::DisplayDepthMap(float min_depth,
                                                  float max_depth,
                                                  const char* window_title,
                                                  bool wait_key, const char* write_path) const {
#if defined(ANDROID)
  (void)wait_key;
  (void)min_depth;
  (void)max_depth;
  (void)window_title;
  (void)write_path;
  LOG(FATAL) << "Not supported on Android.";
#else
  cv::Mat_<float> buffer_cpu(buffer_.height(), buffer_.width());
  CUDABufferAdapter(const_cast<CUDABuffer<T>*>(&buffer_))
      .DebugDownload(&buffer_cpu);
  cv::Mat mat =
      util::GetInvDepthColoredDepthmapMat(buffer_cpu, min_depth, max_depth);
  if (write_path) {
    cv::imwrite(write_path, mat);
  } else {
    cv::imshow(window_title, mat);
  }
  if (wait_key) {
    cv::waitKey(0);
  }
#endif
}

}  // namespace view_correction

#endif  // VIEW_CORRECTION_CUDA_BUFFER_VISUALIZATION_INL_H_
