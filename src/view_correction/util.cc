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

#include "view_correction/util.h"

#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace view_correction {

namespace util {

void DisplayDepthmapInvDepthColored(
    const cv::Mat_<float>& depthmap, float min_depth,
    float max_depth, const char* window_title, bool wait_key) {
  cv::Mat colInvDepth =
      GetInvDepthColoredDepthmapMat(depthmap, min_depth, max_depth);
  cv::namedWindow(window_title, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
  cv::imshow(window_title, colInvDepth);
  if (wait_key) {
    cv::waitKey(0);
  }
}

cv::Mat GetInvDepthColoredDepthmapMat(
    const cv::Mat_<float>& depthmap, float min_depth,
    float max_depth) {
  cv::Mat result = cv::Mat::zeros(depthmap.rows, depthmap.cols, CV_8UC3);
  for (int y = 0; y < result.rows; y++) {
    uint8_t* pixel = result.ptr<uint8_t>(y);
    for (int x = 0; x < result.cols; ++x) {
      const float depth = depthmap(y, x);
      if (depth > 0.f) {
        float factor = std::max(0.0f, std::min(1 / depth - 1 / max_depth,
                                               1 / min_depth - 1 / max_depth) /
                                           (1 / min_depth - 1 / max_depth));

        pixel[0] = 255 * factor + 0.5f;
        pixel[1] = 255 * factor + 0.5f;
        pixel[2] = 255 * factor + 0.5f;
      } else {
        pixel[0] = 255;
        pixel[0] = 0;
        pixel[0] = 0;
      }
      pixel += 3;
    }
  }
  return result;
}

}  // namespace util

}  // namespace view_correction
