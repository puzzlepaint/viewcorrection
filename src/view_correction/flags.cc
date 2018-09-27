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

#include <gflags/gflags.h>

DEFINE_bool(vc_debug, false,
            "Enable debug image output on Linux");
DEFINE_bool(vc_write_images, false,
            "Write out debug images. vc_debug must be set as well to use this.");
DEFINE_string(vc_depth_source, "mesh",
              "Source of depth maps for view correction. Possible "
              "values are \"depth_camera\" and \"mesh\".");

DEFINE_string(vc_inpainting_method, "convolution",
              "One of \"convolution\", \"TV\"");

DEFINE_bool(vc_evaluate_rgb_frame_inpainting, false, "");
DEFINE_bool(vc_evaluate_vs_previous_frame, false, "");

DEFINE_bool(vc_render_tsdf_in_target, false, "");
DEFINE_bool(vc_ensure_target_frame_temporal_consistency, true, "");

DEFINE_bool(vc_do_timings, false, "");
DEFINE_bool(vc_save_timings, false, "");

DEFINE_bool(vc_evaluate_stereo, false, "");
DEFINE_bool(vc_write_stereo_result, false, "");

DEFINE_bool(vc_use_weights_for_inpainting, true, "");
