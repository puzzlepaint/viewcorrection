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

#ifndef VIEW_CORRECTION_FLAGS_H_
#define VIEW_CORRECTION_FLAGS_H_

#include <gflags/gflags.h>

DECLARE_bool(vc_debug);
DECLARE_bool(vc_write_images);
DECLARE_string(vc_depth_source);

DECLARE_string(vc_inpainting_method);

DECLARE_bool(vc_evaluate_rgb_frame_inpainting);
DECLARE_bool(vc_evaluate_vs_previous_frame);

DECLARE_bool(vc_render_tsdf_in_target);
DECLARE_bool(vc_ensure_target_frame_temporal_consistency);

DECLARE_bool(vc_do_timings);
DECLARE_bool(vc_save_timings);

DECLARE_bool(vc_evaluate_stereo);
DECLARE_bool(vc_write_stereo_result);
DECLARE_bool(vc_use_weights_for_inpainting);

namespace view_correction {

// Define possible values for enum-like string flags to avoid having them in
// multiple places.
struct vc_depth_source {
  static constexpr const char* depth_camera = "depth_camera";
  static constexpr const char* mesh = "mesh";
};

struct vc_inpainting_method {
  static constexpr const char* convolution = "convolution";
  static constexpr const char* TV = "TV";
};

inline bool UsingDepthCameraInput() {
  return FLAGS_vc_depth_source == view_correction::vc_depth_source::depth_camera;
}

inline bool UsingMeshInput() {
  return FLAGS_vc_depth_source == view_correction::vc_depth_source::mesh;
}

}

#endif // VIEW_CORRECTION_FLAGS_H_
