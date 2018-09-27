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

#include "view_correction/view_correction_display.h"
#include "view_correction/view_correction_display.cuh"

#include <fstream>
#include <iomanip>

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "view_correction/cuda_buffer.h"
#include "view_correction/cuda_buffer_adapter.h"
#include "view_correction/forward_declarations.h"
#include "view_correction/cuda_buffer_visualization.h"
#include "view_correction/cuda_convolution_inpainting.cuh"
#include "view_correction/cuda_convolution_inpainting_rgb.cuh"
#include "view_correction/cuda_tv_inpainting_functions.cuh"
#include "view_correction/flags.h"
#include "view_correction/mesh_renderer.h"
#include "view_correction/opengl_util.h"
#include "view_correction/position_receiver.h"
#include "view_correction/util.h"

namespace view_correction {

constexpr float kMinDepthForDisplay = 0.5f;
constexpr float kMaxDepthForDisplay = 4.5f;

constexpr float kMinDepthForRendering = 0.1f;
constexpr float kMaxDepthForRendering = 50.f;

static constexpr double kNanosecondsToSeconds = 1e-9;
// static constexpr double kSecondsToNanoseconds = 1e9;

struct ViewCorrectionDisplayImpl {
  // ### Observer position receiver ###
  
  PositionReceiver position_receiver;
  
  // ### Inputs ###
  
  // UV part of source color image.
  CUDABufferPtr<uint16_t> uv_image_gpu;
  // Image pyramid of Y part of source color image ordered by resolution,
  // highest resolution image at index 0.
  std::vector<CUDABufferPtr<uint8_t>> y_image_gpu;
  // Source depth image (in millimeters).
  CUDABufferPtr<uint16_t> depth_image_gpu;
  // Compactly stored gradient magnitude (with pixels in [0, 255]) of
  // y_image_gpu.back() divided by sqrt(2). To convert to actual gradient
  // magnitude, use:
  // gradient_magnitude_div_sqrt2 * sqrt(2)
  CUDABufferPtr<uint8_t> gradient_magnitude_div_sqrt2;
  
  cudaTextureObject_t gradient_magnitude_div_sqrt2_texture;
  cudaTextureObject_t depth_image_gpu_texture;
  
  // ### Source frame TV inpainting ###
  
  CUDABufferPtr<bool> src_tv_flag;
  CUDABufferPtr<bool> src_tv_dual_flag;
  CUDABufferPtr<int16_t> src_tv_dual_x;
  CUDABufferPtr<int16_t> src_tv_dual_y;
  CUDABufferPtr<float> src_tv_u_bar;
  CUDABufferPtr<uint8_t> src_tv_max_change;
  CUDABufferPtr<float> src_tv_max_change_float;
  // Inpainted source frame depth image (in meters).
  CUDABufferPtr<float> src_inpainted_depth_map;
  CUDABufferPtr<float> src_inpainted_depth_map_alternate;
  CUDABufferPtr<uint16_t> src_block_coordinates;
  CUDABufferPtr<unsigned char> src_block_activities;

  // ### Meshed inpainted depth map ###
  
  bool have_meshed_inpainted_depth_map = false;
  Sophus::SE3f G_T_src_C_;
  uint64_t G_T_src_C_timestamp_;
  cudaGraphicsResource_t vertex_buffer_resource;
  cudaGraphicsResource_t color_buffer_resource;
  cudaGraphicsResource_t index_buffer_resource;
  GLuint vertex_buffer;
  GLuint color_buffer;
  GLuint index_buffer;
  int num_mesh_indices;
  
  cudaGraphicsResource_t raw_vertex_buffer_resource;
  cudaGraphicsResource_t raw_color_buffer_resource;
  cudaGraphicsResource_t raw_index_buffer_resource;
  GLuint raw_vertex_buffer;
  GLuint raw_color_buffer;
  GLuint raw_index_buffer;
  int raw_num_mesh_indices;
  
  // ### Mesh renderer ###
  
  std::unique_ptr<MeshRenderer> src_mesh_renderer_;
  std::unique_ptr<MeshRenderer> mesh_renderer_;
  std::unique_ptr<MeshRenderer> tsdf_mesh_renderer_;
  
  // ### Target frame buffers ###
  
  CUDABufferPtr<float> target_rendered_depth;
  cudaTextureObject_t target_rendered_depth_texture;
  CUDABufferPtr<uchar4> target_rendered_color;
  
  CUDABufferPtr<bool> target_tv_flag;
  CUDABufferPtr<bool> target_tv_dual_flag;
  CUDABufferPtr<int16_t> target_tv_dual_x;
  CUDABufferPtr<int16_t> target_tv_dual_y;
  CUDABufferPtr<float> target_tv_u_bar;
  CUDABufferPtr<uint8_t> target_tv_max_change;
  CUDABufferPtr<float> target_tv_max_change_float;
  CUDABufferPtr<float> target_inpainted_depth_map;
  CUDABufferPtr<uint16_t> target_block_coordinates;
  CUDABufferPtr<unsigned char> target_block_activities;

  CUDABufferPtr<bool> target_color_tv_flag;
  CUDABufferPtr<bool> target_color_tv_dual_flag;
  CUDABufferPtr<float> target_color_tv_dual_x_r;
  CUDABufferPtr<float> target_color_tv_dual_x_g;
  CUDABufferPtr<float> target_color_tv_dual_x_b;
  CUDABufferPtr<float> target_color_tv_dual_y_r;
  CUDABufferPtr<float> target_color_tv_dual_y_g;
  CUDABufferPtr<float> target_color_tv_dual_y_b;
  CUDABufferPtr<float> target_color_tv_u_bar_r;
  CUDABufferPtr<float> target_color_tv_u_bar_g;
  CUDABufferPtr<float> target_color_tv_u_bar_b;
  CUDABufferPtr<uint8_t> target_color_tv_max_change;
  CUDABufferPtr<float> target_color_tv_max_change_float;
  CUDABufferPtr<uchar4> target_inpainted_color_rgb;
  CUDABufferPtr<float> target_inpainted_color_r;
  CUDABufferPtr<float> target_inpainted_color_g;
  CUDABufferPtr<float> target_inpainted_color_b;
  
  // ### Rendering to screen ###
  
  GLuint display_color_texture;
  cudaGraphicsResource_t display_color_texture_resource;
  GLuint display_depth_texture;
  cudaGraphicsResource_t display_depth_texture_resource;
  
  GLuint display_shader_program;
  GLuint display_vertex_shader;
  GLuint display_fragment_shader;
  GLint display_a_position_location;
  GLint display_a_tex_coord_location;
  GLint display_u_color_texture_location;
  GLint display_u_depth_texture_location;
  GLint display_u_proj22_location;
  GLint display_u_proj23_location;
  
  // ### AR Rendering ###
  
//   ARRenderer ar_renderer;
  
  // ### Other ###
  
  // Pose of the last rendering (for propagating it to the next one).
  Sophus::SE3f last_target_T_G_;
  float last_target_fx_inv;
  float last_target_fy_inv;
  float last_target_cx_inv;
  float last_target_cy_inv;
  bool have_previous_rendering_;
  
  int output_frame_index;
  
  // Events for timing.
  cudaEvent_t render_or_upload_depth_start_event;
  cudaEvent_t meshing_start_event;
  cudaEvent_t meshing_upload_end_event;
  cudaEvent_t meshing_downsampling_end_event;
  cudaEvent_t meshing_gradient_mags_end_event;
  cudaEvent_t meshing_inpainting_end_event;
  cudaEvent_t meshing_end_event;
  
  cudaEvent_t rendering_start_event;
  cudaEvent_t rendering_end_event;
  cudaEvent_t color_reprojection_end_event;
  cudaEvent_t last_frame_reprojection_end_event;
  cudaEvent_t target_depth_inpainting_end_event;
  cudaEvent_t target_color_inpainting_end_event;
  
  // CUDA stream.
  cudaStream_t stream;
};

ViewCorrectionDisplay::ViewCorrectionDisplay(
    int width, int height, int offset_x, int offset_y,
    TargetViewMode target_view_mode,
    const Intrinsics& depth_intrinsics,
    const Intrinsics& yuv_intrinsics)
    : target_view_mode_(target_view_mode),
      input_mesh_(new MeshStub()),
      latest_depth_image_timestamp_(0),
      depth_intrinsics_(depth_intrinsics),
      depth_fx_(depth_intrinsics_.fx),
      depth_fy_(depth_intrinsics_.fy),
      depth_cx_(depth_intrinsics_.cx),
      depth_cy_(depth_intrinsics_.cy),
      depth_fx_inv_(1.f / depth_intrinsics_.fx),
      depth_fy_inv_(1.f / depth_intrinsics_.fy),
      depth_cx_inv_(-depth_intrinsics_.cx / depth_intrinsics_.fx),
      depth_cy_inv_(-depth_intrinsics_.cy / depth_intrinsics_.fy),
      yuv_intrinsics_(yuv_intrinsics),
      offset_x_(offset_x),
      offset_y_(offset_y),
      width_(width),
      height_(height),
      d_(new ViewCorrectionDisplayImpl()) {
  LOG(INFO) << "Initializing ViewCorrectionDisplay of size " << width << " x " << height;
  
  d_->have_previous_rendering_ = false;
  
  // The Tango tablet's screen is 1920 x 1200 at 323 ppi.
  // The pixel size should remain square. Some suggested resolutions are in the
  // comments:
  target_render_width_  = 480;  // 384;  480;  640;  960;  1920;
  target_render_height_ = 300;  // 240;  300;  400;  600;  1200;
  if (FLAGS_vc_evaluate_stereo) {
    target_render_width_ /= 2;
  }
  
  // Initialize UDP receiver if required.
  if (target_view_mode_ == TargetViewMode::kReceiveFromUDP) {
    constexpr uint16_t udp_pose_port = 9999;
    if (!d_->position_receiver.Initialize(udp_pose_port)) {
      LOG(FATAL) << "Failed to initalize PositionReceiver.";
    }
    d_->position_receiver.StartReceiveThread();
  }
  
  d_->output_frame_index = 0;
}

ViewCorrectionDisplay::~ViewCorrectionDisplay() {
  cudaEventDestroy(d_->render_or_upload_depth_start_event);
  cudaEventDestroy(d_->meshing_start_event);
  cudaEventDestroy(d_->meshing_upload_end_event);
  cudaEventDestroy(d_->meshing_downsampling_end_event);
  cudaEventDestroy(d_->meshing_gradient_mags_end_event);
  cudaEventDestroy(d_->meshing_inpainting_end_event);
  cudaEventDestroy(d_->meshing_end_event);
  
  cudaEventDestroy(d_->rendering_start_event);
  cudaEventDestroy(d_->rendering_end_event);
  cudaEventDestroy(d_->color_reprojection_end_event);
  cudaEventDestroy(d_->last_frame_reprojection_end_event);
  cudaEventDestroy(d_->target_depth_inpainting_end_event);
  cudaEventDestroy(d_->target_color_inpainting_end_event);
  
  if (UsingDepthCameraInput()) {
    cudaDestroyTextureObject(d_->depth_image_gpu_texture);
  }
  cudaDestroyTextureObject(d_->gradient_magnitude_div_sqrt2_texture);
  
  cudaDestroyTextureObject(d_->target_rendered_depth_texture);
  
  cudaGraphicsUnregisterResource(d_->display_color_texture_resource);
  glDeleteTextures(1, &d_->display_color_texture);
  cudaGraphicsUnregisterResource(d_->display_depth_texture_resource);
  glDeleteTextures(1, &d_->display_depth_texture);
  
  cudaGraphicsUnregisterResource(d_->vertex_buffer_resource);
  cudaGraphicsUnregisterResource(d_->color_buffer_resource);
  cudaGraphicsUnregisterResource(d_->index_buffer_resource);
  glDeleteBuffers(1, &d_->vertex_buffer);
  glDeleteBuffers(1, &d_->color_buffer);
  glDeleteBuffers(1, &d_->index_buffer);
  
  if (FLAGS_vc_evaluate_rgb_frame_inpainting) {
    cudaGraphicsUnregisterResource(d_->raw_vertex_buffer_resource);
    cudaGraphicsUnregisterResource(d_->raw_color_buffer_resource);
    cudaGraphicsUnregisterResource(d_->raw_index_buffer_resource);
    glDeleteBuffers(1, &d_->raw_vertex_buffer);
    glDeleteBuffers(1, &d_->raw_color_buffer);
    glDeleteBuffers(1, &d_->raw_index_buffer);
  }
}

void ViewCorrectionDisplay::Init() {
  GLenum error_code;
  while ((error_code = glGetError()) != GL_NO_ERROR) {}
  CHECK_OPENGL_NO_ERROR();
  
  d_->G_T_src_C_timestamp_ = -std::numeric_limits<float>::infinity();
  
  const int yuv_width = yuv_intrinsics_.width;
  const int yuv_height = yuv_intrinsics_.height;
  const int depth_width = depth_intrinsics_.width;
  const int depth_height = depth_intrinsics_.height;
  
  // Create CUDA stream.
  cudaStreamCreate(&d_->stream);
  
  // Create CUDA events.
  cudaEventCreate(&d_->render_or_upload_depth_start_event);
  cudaEventCreate(&d_->meshing_start_event);
  cudaEventCreate(&d_->meshing_upload_end_event);
  cudaEventCreate(&d_->meshing_downsampling_end_event);
  cudaEventCreate(&d_->meshing_gradient_mags_end_event);
  cudaEventCreate(&d_->meshing_inpainting_end_event);
  cudaEventCreate(&d_->meshing_end_event);
  
  cudaEventCreate(&d_->rendering_start_event);
  cudaEventCreate(&d_->rendering_end_event);
  cudaEventCreate(&d_->color_reprojection_end_event);
  cudaEventCreate(&d_->last_frame_reprojection_end_event);
  cudaEventCreate(&d_->target_depth_inpainting_end_event);
  cudaEventCreateWithFlags(&d_->target_color_inpainting_end_event, cudaEventBlockingSync);
  
  // Create input data buffers.
  d_->depth_image_gpu.reset(new CUDABuffer<uint16_t>(depth_height, depth_width));
  d_->gradient_magnitude_div_sqrt2.reset(new CUDABuffer<uint8_t>(depth_height, depth_width));
  d_->uv_image_gpu.reset(new CUDABuffer<uint16_t>(yuv_height / 2, yuv_width / 2));
  int num_y_pyramid_levels = log2(yuv_height / depth_height) + 1.5;
  d_->y_image_gpu.resize(num_y_pyramid_levels);
  for (int i = 0; i < num_y_pyramid_levels; ++ i) {
    d_->y_image_gpu[i].reset(new CUDABuffer<uint8_t>(
        yuv_height / exp2(i), yuv_width / exp2(i)));
  }
  
  d_->gradient_magnitude_div_sqrt2->CreateTextureObject(
      cudaAddressModeClamp, cudaAddressModeClamp, cudaFilterModePoint,
      cudaReadModeElementType, false, &d_->gradient_magnitude_div_sqrt2_texture);
  // If using depth camera input, map the depth_image_gpu buffer to the
  // depth_image_gpu_texture. Otherwise, the depth_image_gpu_texture will be
  // mapped to the rendering output.
  if (UsingDepthCameraInput()) {
    d_->depth_image_gpu->CreateTextureObject(
        cudaAddressModeClamp, cudaAddressModeClamp, cudaFilterModePoint,
        cudaReadModeNormalizedFloat, false, &d_->depth_image_gpu_texture);
  }
  
  // Create source frame inpainting buffers.
  d_->src_tv_flag.reset(new CUDABuffer<bool>(depth_height, depth_width));
  d_->src_tv_dual_flag.reset(new CUDABuffer<bool>(depth_height, depth_width));
  d_->src_tv_dual_x.reset(new CUDABuffer<int16_t>(depth_height, depth_width));
  d_->src_tv_dual_y.reset(new CUDABuffer<int16_t>(depth_height, depth_width));
  d_->src_tv_u_bar.reset(new CUDABuffer<float>(depth_height, depth_width));
  d_->src_tv_max_change.reset(new CUDABuffer<uint8_t>(1, depth_height * depth_width));
  d_->src_tv_max_change_float.reset(new CUDABuffer<float>(1, depth_height * depth_width));
  d_->src_inpainted_depth_map.reset(new CUDABuffer<float>(depth_height, depth_width));
  d_->src_inpainted_depth_map_alternate.reset(new CUDABuffer<float>(depth_height, depth_width));
  d_->src_block_coordinates.reset(new CUDABuffer<uint16_t>(
      1, depth_height * depth_width));
  d_->src_block_activities.reset(new CUDABuffer<unsigned char>(
      depth_height, depth_width));

  // Create vertex and index buffer for meshed inpainted depth map.
  const int num_vertices = depth_width * depth_height;
  d_->num_mesh_indices = MeshRenderer::GetIndexCount(depth_width, depth_height);

  CHECK_OPENGL_NO_ERROR();
  glGenBuffers(1, &d_->vertex_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, d_->vertex_buffer);
  glBufferData(GL_ARRAY_BUFFER, num_vertices * 3 * sizeof(float),  // NOLINT
               nullptr, GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  CHECK_OPENGL_NO_ERROR();
  
  glGenBuffers(1, &d_->color_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, d_->color_buffer);
  glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(uint8_t),  // NOLINT
               nullptr, GL_STREAM_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  CHECK_OPENGL_NO_ERROR();

  glGenBuffers(1, &d_->index_buffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, d_->index_buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               d_->num_mesh_indices * sizeof(uint32_t),  // NOLINT
               nullptr, GL_STREAM_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  CHECK_OPENGL_NO_ERROR();
  
  CUDA_CHECKED_CALL(cudaGraphicsGLRegisterBuffer(
      &d_->vertex_buffer_resource, d_->vertex_buffer,
      cudaGraphicsRegisterFlagsWriteDiscard));
  CUDA_CHECKED_CALL(cudaGraphicsGLRegisterBuffer(
      &d_->color_buffer_resource, d_->color_buffer,
      cudaGraphicsRegisterFlagsWriteDiscard));
  CUDA_CHECKED_CALL(cudaGraphicsGLRegisterBuffer(
      &d_->index_buffer_resource, d_->index_buffer,
      cudaGraphicsRegisterFlagsWriteDiscard));
  
  if (FLAGS_vc_evaluate_rgb_frame_inpainting) {
    CHECK_OPENGL_NO_ERROR();
    glGenBuffers(1, &d_->raw_vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, d_->raw_vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, num_vertices * 3 * sizeof(float),  // NOLINT
                nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CHECK_OPENGL_NO_ERROR();
    
    glGenBuffers(1, &d_->raw_color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, d_->raw_color_buffer);
    glBufferData(GL_ARRAY_BUFFER, num_vertices * sizeof(uint8_t),  // NOLINT
                nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CHECK_OPENGL_NO_ERROR();

    glGenBuffers(1, &d_->raw_index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, d_->raw_index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                d_->num_mesh_indices * sizeof(uint32_t),  // NOLINT
                nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    CHECK_OPENGL_NO_ERROR();
    
    CUDA_CHECKED_CALL(cudaGraphicsGLRegisterBuffer(
        &d_->raw_vertex_buffer_resource, d_->raw_vertex_buffer,
        cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECKED_CALL(cudaGraphicsGLRegisterBuffer(
        &d_->raw_color_buffer_resource, d_->raw_color_buffer,
        cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECKED_CALL(cudaGraphicsGLRegisterBuffer(
        &d_->raw_index_buffer_resource, d_->raw_index_buffer,
        cudaGraphicsRegisterFlagsWriteDiscard));
  }
  
  // Initialize mesh renderer.
  // TODO: could use one renderer with the maximum of the image sizes.
  d_->src_mesh_renderer_.reset(new MeshRenderer(depth_width, depth_height, MeshRenderer::kRenderDepthOnly));
  d_->mesh_renderer_.reset(new MeshRenderer(target_render_width_, target_render_height_, MeshRenderer::kRenderDepthAndIntensity));
  if (UsingMeshInput() && FLAGS_vc_render_tsdf_in_target) {
    d_->tsdf_mesh_renderer_.reset(new MeshRenderer(target_render_width_, target_render_height_, MeshRenderer::kRenderDepthAndColor));
  }
  
  // Initialize target frame buffers.
  d_->target_rendered_depth.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_rendered_depth->CreateTextureObject(
      cudaAddressModeClamp, cudaAddressModeClamp, cudaFilterModePoint,
      cudaReadModeElementType, false, &d_->target_rendered_depth_texture);
  d_->target_rendered_color.reset(new CUDABuffer<uchar4>(target_render_height_, target_render_width_));
  d_->target_tv_flag.reset(new CUDABuffer<bool>(target_render_height_, target_render_width_));
  d_->target_tv_dual_flag.reset(new CUDABuffer<bool>(target_render_height_, target_render_width_));
  d_->target_tv_dual_x.reset(new CUDABuffer<int16_t>(target_render_height_, target_render_width_));
  d_->target_tv_dual_y.reset(new CUDABuffer<int16_t>(target_render_height_, target_render_width_));
  d_->target_tv_u_bar.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_tv_max_change.reset(new CUDABuffer<uint8_t>(
      1, target_render_height_ * target_render_width_));
  d_->target_tv_max_change_float.reset(new CUDABuffer<float>(
      1, target_render_height_ * target_render_width_));
  d_->target_inpainted_depth_map.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_block_coordinates.reset(new CUDABuffer<uint16_t>(
      1, target_render_height_ * target_render_width_));
  d_->target_block_activities.reset(new CUDABuffer<unsigned char>(
      target_render_height_, target_render_width_));
  d_->target_color_tv_flag.reset(new CUDABuffer<bool>(target_render_height_, target_render_width_));
  d_->target_color_tv_dual_flag.reset(new CUDABuffer<bool>(target_render_height_, target_render_width_));
  d_->target_color_tv_dual_x_r.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_color_tv_dual_x_g.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_color_tv_dual_x_b.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_color_tv_dual_y_r.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_color_tv_dual_y_g.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_color_tv_dual_y_b.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_color_tv_u_bar_r.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_color_tv_u_bar_g.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_color_tv_u_bar_b.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_color_tv_max_change.reset(new CUDABuffer<uint8_t>(1, target_render_height_ * target_render_width_));
  d_->target_color_tv_max_change_float.reset(new CUDABuffer<float>(1, target_render_height_ * target_render_width_));
  d_->target_inpainted_color_rgb.reset(new CUDABuffer<uchar4>(target_render_height_, target_render_width_));
  d_->target_inpainted_color_r.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_inpainted_color_g.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  d_->target_inpainted_color_b.reset(new CUDABuffer<float>(target_render_height_, target_render_width_));
  
  // Initialize shader program for display.
  InitDisplay();
  
//   // Initialize shader for AR rendering.
//   if (FLAGS_vc_ar_demo) {
//     InitAR();
//   }
}

void ViewCorrectionDisplay::InitDisplay() {
  // Create vertex shader.
  const GLchar* vertex_shader_src[] = {
      "#version 300 es\n"
      "in vec4 a_position;\n"
      "in vec2 a_tex_coord;\n"
      "out vec2 v_tex_coord;\n"
      "void main() {\n"
      "  v_tex_coord = a_tex_coord;\n"
      "  gl_Position = a_position;\n"
      "}\n"};

  d_->display_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(d_->display_vertex_shader, 1, vertex_shader_src, NULL);
  glCompileShader(d_->display_vertex_shader);

  GLint compiled;
  glGetShaderiv(d_->display_vertex_shader, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    GLint length;
    glGetShaderiv(d_->display_vertex_shader, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetShaderInfoLog(d_->display_vertex_shader, length, &length, log.get());
    LOG(FATAL) << "GL Shader Compilation Error: " << log.get();
  }

  // Create fragment shader.
  const GLchar* fragment_shader_src[] = {
      "#version 300 es\n"
      "uniform sampler2D color_texture;\n"
      "uniform sampler2D depth_texture;\n"
      "uniform highp float proj22;\n"
      "uniform highp float proj23;\n"
      "in mediump vec2 v_tex_coord;\n"
      "layout(location = 0) out lowp vec3 out_color;\n"
      "void main() {\n"
      "  out_color = texture(color_texture, v_tex_coord).rgb;\n"
      "  highp float depth = texture(depth_texture, v_tex_coord).r;\n"
      "  highp float ndc_depth = (proj22 * depth + proj23) / depth;\n"  // (projection * pos).z / (projection * pos).w
      "  gl_FragDepth = 0.5 * ndc_depth + 0.5;\n"
      "}\n"};

  d_->display_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(d_->display_fragment_shader, 1, fragment_shader_src, NULL);
  glCompileShader(d_->display_fragment_shader);

  glGetShaderiv(d_->display_fragment_shader, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    GLint length;
    glGetShaderiv(d_->display_fragment_shader, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetShaderInfoLog(d_->display_fragment_shader, length, &length, log.get());
    LOG(FATAL) << "GL Shader Compilation Error: " << log.get();
  }
  
  // Create shader program.
  d_->display_shader_program = glCreateProgram();
  glAttachShader(d_->display_shader_program, d_->display_fragment_shader);
  glAttachShader(d_->display_shader_program, d_->display_vertex_shader);
  glLinkProgram(d_->display_shader_program);

  GLint linked;
  glGetProgramiv(d_->display_shader_program, GL_LINK_STATUS, &linked);
  if (!linked) {
    GLint length;
    glGetProgramiv(d_->display_shader_program, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetProgramInfoLog(d_->display_shader_program, length, &length, log.get());
    LOG(FATAL) << "GL Program Linker Error: " << log.get();
  }

  glUseProgram(d_->display_shader_program);

  // Get attributes.
  d_->display_a_position_location = glGetAttribLocation(d_->display_shader_program, "a_position");
  d_->display_a_tex_coord_location = glGetAttribLocation(d_->display_shader_program, "a_tex_coord");
  d_->display_u_color_texture_location = glGetUniformLocation(d_->display_shader_program, "color_texture");
  d_->display_u_depth_texture_location = glGetUniformLocation(d_->display_shader_program, "depth_texture");
  d_->display_u_proj22_location = glGetUniformLocation(d_->display_shader_program, "proj22");
  d_->display_u_proj23_location = glGetUniformLocation(d_->display_shader_program, "proj23");
  
  glUseProgram(0);
  
  // Create color display texture.
  glGenTextures(1, &d_->display_color_texture);
  glBindTexture(GL_TEXTURE_2D, d_->display_color_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, target_render_width_,
               target_render_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  CUDA_CHECKED_CALL(cudaGraphicsGLRegisterImage(
      &d_->display_color_texture_resource, d_->display_color_texture,
      GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
  
  // Create depth display texture.
  glGenTextures(1, &d_->display_depth_texture);
  glBindTexture(GL_TEXTURE_2D, d_->display_depth_texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, target_render_width_,
               target_render_height_, 0, GL_RED, GL_FLOAT, NULL);
  CUDA_CHECKED_CALL(cudaGraphicsGLRegisterImage(
      &d_->display_depth_texture_resource, d_->display_depth_texture,
      GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

bool ViewCorrectionDisplay::Render() {
  // Clear OpenGL errors which happened before.
  while (glGetError() != GL_NO_ERROR);
  
  // Run pipeline normally.
  RunPipeline(true, true, false);
  
  if (FLAGS_vc_evaluate_stereo) {
    // Render the right image (the first one was the left one).
    RunPipeline(false, true, true);
  }
  
  // If enabled, also render the same image with different settings for
  // evaluation.
  if (FLAGS_vc_evaluate_rgb_frame_inpainting) {
    RunPipeline(false, false, false);
  }
  
  return true;
}

bool ViewCorrectionDisplay::RunPipeline(bool update_data, bool inpaint_in_rgb_frame, bool render_stereo_image) {
  if (!FLAGS_vc_evaluate_vs_previous_frame) {
    ++ d_->output_frame_index;
  }
  
  // If using depth camera input and a new depth map & color image pair is
  // available, or using mesh input, update the meshed inpainted depth map.
  // NOTE(puzzlepaint): This could be done asynchronously to the render loop,
  // possibly even with a lower frequency.
  DepthImage new_depth_image;
  ColorImage new_yuv_image;
  bool have_new_input = false;
  if (update_data) {
    std::unique_lock<std::mutex> input_lock(input_mutex_);
    // Check for new input in the case of using the depth camera images directly.
    if (!input_depth_image_.empty()) {
      new_depth_image = input_depth_image_;
      
      // Search for the yuv image with the closest timestamp to the input depth image.
      if (!input_yuv_images_.empty() &&
          input_yuv_images_.front().timestamp_ns() > new_depth_image.timestamp_ns()) {
        LOG(ERROR) << "No suitable color image cached anymore for depth"
                   << " image. Increase the caching time.";
        new_yuv_image = input_yuv_images_.front();
      } else {
        for (int i = 0; i < static_cast<int>(input_yuv_images_.size()) - 1; ++ i) {
          uint64_t prev_timestamp = input_yuv_images_[i].timestamp_ns();
          uint64_t next_timestamp = input_yuv_images_[i + 1].timestamp_ns();
          if (prev_timestamp <= new_depth_image.timestamp_ns() &&
              next_timestamp >= new_depth_image.timestamp_ns()) {
            // Choose YUV image which is closest to the depth image.
            if (new_depth_image.timestamp_ns() - prev_timestamp >
                next_timestamp - new_depth_image.timestamp_ns()) {
              new_yuv_image = input_yuv_images_[i + 1];
              break;
            } else {
              new_yuv_image = input_yuv_images_[i];
              break;
            }
          }
        }
        
        if (new_yuv_image.empty()) {
          if (input_yuv_images_.empty()) {
            LOG(WARNING) << "No color image available yet for depth image.";
          } else {
            // No color image available which is newer than the last depth image.
            // Wait longer. This may happen for datasets with sparse color image
            // recording.
            LOG(WARNING) << "No color image available which is newer than the"
                        << " last depth image. Using last one.";
            new_yuv_image = input_yuv_images_.back();
          }
        }
      }
    }
    if (!new_depth_image.empty() && !new_yuv_image.empty()) {
      have_new_input = true;
      input_depth_image_.release();
      if (FLAGS_vc_evaluate_vs_previous_frame) {
        if (!input_mesh_->empty()) {
          mesh_to_render = input_mesh_;
        }
      }
    }
    
    // Check for new input in the case of using meshes.
    if (!FLAGS_vc_evaluate_vs_previous_frame && !input_mesh_->empty() && !input_yuv_images_.empty()) {
      have_new_input = true;
      mesh_to_render = input_mesh_;
      
      // Choose the latest YUV image for which the pose is available without waiting.
      uint64_t latest_timestamp = GetCurrentTimestamp();
      
      int num_images_to_delete = 0;
      new_yuv_image = input_yuv_images_.front();
      for (size_t i = 1; i < input_yuv_images_.size(); ++ i) {
        if (input_yuv_images_[i].timestamp_ns() <= latest_timestamp) {
          new_yuv_image = input_yuv_images_[i];
          num_images_to_delete = i - 1;
        }
      }
      
      for (int i = 0; i < num_images_to_delete; ++ i) {
        input_yuv_images_.erase(input_yuv_images_.begin());
      }
      
      // Set have_new_input = false if the chosen image is not new.
      // NOTE: Comment this out for performance measurements (to simulate live mode where new color images are available in every frame)!
      if (new_yuv_image.timestamp_ns() <= d_->G_T_src_C_timestamp_) {
        have_new_input = false;
      }
    }
    input_lock.unlock();
  }
  if (FLAGS_vc_evaluate_vs_previous_frame &&
      (!have_new_input ||
          (!last_yuv_image_.empty() && new_yuv_image.timestamp_ns() == last_yuv_image_.timestamp_ns()))) {
    return true;
  }
  
  uint32_t num_src_depth_pixels_to_inpaint = 0;
  if (have_new_input) {
    if (FLAGS_vc_evaluate_vs_previous_frame) {
      d_->output_frame_index = 1000 * kNanosecondsToSeconds * new_yuv_image.timestamp_ns();
    }
    
    // Get pose of yuv image.
    bool success = GetYUVImagePose(new_yuv_image, &d_->G_T_src_C_);
    d_->G_T_src_C_timestamp_ = new_yuv_image.timestamp_ns();
    if (!success) {
      // Show red screen to signal that something went wrong.
      glViewport(offset_x_, offset_y_, width_, height_);
      glClearColor(0.9, 0.1, 0.1, 1.0);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      return true;
    }
    
//     // Use latest "device" pose for the image (should be very close) for
//     // consistency (the other way returned different poses).
//     std::unique_lock<std::mutex> input_lock(input_mutex_);
//     d_->G_T_src_C_ = input_G_T_C_;
//     d_->G_T_src_C_timestamp_ = input_G_T_C_timestamp_;
//     input_lock.unlock();
    
    // Render or upload depth map.
    cudaEventRecord(d_->render_or_upload_depth_start_event, d_->stream);
    if (mesh_to_render) {
      // Render depth image from mesh.
      RenderDepthImageFromMesh(*mesh_to_render, d_->G_T_src_C_);
      d_->depth_image_gpu_texture =
          d_->src_mesh_renderer_->MapDepthResultAsTexture(
              cudaAddressModeClamp, cudaAddressModeClamp, cudaFilterModePoint,
              false, d_->stream);
    } else {
      // Upload depth map.
      d_->depth_image_gpu->UploadPitchedAsync(d_->stream, new_depth_image.step, reinterpret_cast<const uint16_t*>(new_depth_image.data));
    }
    
    // Debug: show initial depth map.
    if (FLAGS_vc_debug || FLAGS_vc_write_images) {
      if (UsingMeshInput()) {
        CUDABuffer<float> temp_depth_buffer(d_->depth_image_gpu->height(),
                                            d_->depth_image_gpu->width());
        temp_depth_buffer.SetTo(d_->depth_image_gpu_texture, d_->stream);
        std::ostringstream filename;
        filename << "debug_images/" << d_->output_frame_index << "_0_source_depth.png";
        CUDABufferVisualization(temp_depth_buffer).DisplayDepthMap(kMinDepthForDisplay, kMaxDepthForDisplay, "0 - Source frame depth rendering", false, FLAGS_vc_write_images ? filename.str().c_str() : nullptr);
      } else {
        cv::Mat_<float> buffer_cpu(d_->depth_image_gpu->height(),
                                   d_->depth_image_gpu->width());
        for (int y = 0; y < buffer_cpu.rows; ++y) {
          for (int x = 0; x < buffer_cpu.cols; ++x) {
            buffer_cpu(y, x) = (1 / 1000.f) * *(reinterpret_cast<const uint16_t*>(new_depth_image.data + y * new_depth_image.step) + x);
          }
        }
        cv::Mat mat =
            util::GetInvDepthColoredDepthmapMat(buffer_cpu, kMinDepthForDisplay, kMaxDepthForDisplay);
        if (FLAGS_vc_debug) {
          cv::imshow("0 - Source frame depth input", mat);
        }
        if (FLAGS_vc_write_images) {
          std::ostringstream filename;
          filename << "debug_images/" << d_->output_frame_index << "_0_source_depth.png";
          cv::imwrite(filename.str(), mat);
        }
      }
    }
    
    CreateMeshedInpaintedDepthMap(new_yuv_image, &num_src_depth_pixels_to_inpaint);
    
    if (mesh_to_render) {
      d_->src_mesh_renderer_->UnmapDepthResult(d_->depth_image_gpu_texture,
                                               d_->stream);
    }
  }
  
  if (!d_->have_meshed_inpainted_depth_map) {
    // No depth input yet.
    glViewport(offset_x_, offset_y_, width_, height_);
    glClearColor(0.1, 0.1, 0.1, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    return true;
  }
  
  // Get latest available color camera pose.
  // NOTE: The correct pose to use would be the expected pose at the time at
  // which the result of this rendering iteration is displayed.
  bool pose_retrieval_result = GetCurrentColorCameraPose(&G_T_latest_C, &G_T_latest_C_timestamp);
  if (!pose_retrieval_result) {
    LOG(ERROR) << "Cannot get latest color camera pose";
    // Show red screen to signal that something went wrong.
    glViewport(offset_x_, offset_y_, width_, height_);
    glClearColor(0.9, 0.1, 0.1, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    return true;
  }
  
//   if (update_data) {
//     std::unique_lock<std::mutex> input_lock(input_mutex_);
//     G_T_latest_C = input_G_T_C_;
//     G_T_latest_C_timestamp = input_G_T_C_timestamp_;
//     input_lock.unlock();
//   }
  
  // Define target view. This needs to set target_T_src such that
  // it holds the transformation bringing points in the source frame into the
  // virtual view. Ideally, this should be the transformation which is valid at
  // the point in time at which the result of this iteration will be displayed.
  // Furthermore, the virtual camera intrinsics target_<fx, fy, cx, cy> must be set.
  Sophus::SE3f target_T_src;
  float target_fx;
  float target_fy;
  float target_cx;
  float target_cy;
  if (!SetupTargetView(G_T_latest_C, render_stereo_image, &target_T_src, &target_fx,
                       &target_fy, &target_cx, &target_cy)) {
    if (FLAGS_vc_evaluate_vs_previous_frame && have_new_input) {
      last_yuv_image_ = new_yuv_image;
      last_yuv_image_G_T_C_ = d_->G_T_src_C_;
    }
    
    glViewport(offset_x_, offset_y_, width_, height_);
    glClearColor(0.1, 0.1, 0.1, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    return true;
  }
  
  // Invert target frame intrinsics.
  const float target_fx_inv = 1.f / target_fx;
  const float target_fy_inv = 1.f / target_fy;
  const float target_cx_inv = -target_cx / target_fx;
  const float target_cy_inv = -target_cy / target_fy;
  
  cudaEventRecord(d_->rendering_start_event, d_->stream);
  
  // Render meshed depth map into the target frame to get the initial target
  // depth map and the inpainting weights.
  constexpr float kRenderMinDepth = 0.1f;
  constexpr float kRenderMaxDepth = 50.f;
  d_->mesh_renderer_->RenderMesh(
      inpaint_in_rgb_frame ? d_->vertex_buffer : d_->raw_vertex_buffer,
      inpaint_in_rgb_frame ? d_->color_buffer : d_->raw_color_buffer,
      inpaint_in_rgb_frame ? d_->index_buffer : d_->raw_index_buffer,
      d_->num_mesh_indices,
      target_T_src,
      target_fx, target_fy, target_cx, target_cy,
      kRenderMinDepth, kRenderMaxDepth);
  cudaTextureObject_t rendered_depth_texture =
      d_->mesh_renderer_->MapDepthResultAsTexture(
          cudaAddressModeClamp, cudaAddressModeClamp, cudaFilterModePoint,
          false, d_->stream);
  cudaTextureObject_t rendered_intensity_texture =
      d_->mesh_renderer_->MapIntensityResultAsTexture(
          cudaAddressModeClamp, cudaAddressModeClamp, cudaFilterModePoint,
          cudaReadModeElementType, false, d_->stream);
  
  // Copy the depth to a CUDA buffer, which allows to modify it. It will be
  // accessed from there later, so unmap the original. The intensity texture
  // is kept for longer.
  d_->target_rendered_depth->SetTo(rendered_depth_texture, d_->stream);
  d_->mesh_renderer_->UnmapDepthResult(rendered_depth_texture, d_->stream);
  
  cudaEventRecord(d_->rendering_end_event, d_->stream);
  
  // Debug.
  if (FLAGS_vc_debug || FLAGS_vc_write_images) {
    // Display rendered depth map.
    std::ostringstream filename;
    filename << "debug_images/" << d_->output_frame_index << "_4_target_rendered_depth.png";
    CUDABufferVisualization(*d_->target_rendered_depth).DisplayDepthMap(kMinDepthForDisplay, kMaxDepthForDisplay, "4 - Target frame rendered depth", false, FLAGS_vc_write_images ? filename.str().c_str() : nullptr);
    
    // Display rendered gradient magnitudes for weights.
    CUDABuffer<uint8_t> temp_intensity_buffer(target_render_height_, target_render_width_);
    temp_intensity_buffer.SetTo(rendered_intensity_texture, d_->stream);
    std::ostringstream filename2;
    filename2 << "debug_images/" << d_->output_frame_index << "_5_target_rendered_intensities_for_weights.png";
    CUDABufferVisualization(temp_intensity_buffer).Display(0, 255, "5 - Target frame rendered intensities for weights", false, FLAGS_vc_write_images ? filename2.str().c_str() : nullptr);
  }
  
  // Get partial color image for target frame by projecting the yuv image onto
  // the target depth map.
  ProjectImageOntoDepthMapCUDA(
      d_->stream,
      d_->target_rendered_depth_texture,
      *d_->y_image_gpu.front(),
      *d_->uv_image_gpu,
      yuv_intrinsics_.fx,
      yuv_intrinsics_.fy,
      yuv_intrinsics_.cx,
      yuv_intrinsics_.cy,
      0 /*yuv_intrinsics_.distortion_coefficients()(0, 0)*/,  // TODO: Assuming pinhole camera
      0 /*yuv_intrinsics_.distortion_coefficients()(1, 0)*/,  // TODO: Assuming pinhole camera
      0 /*yuv_intrinsics_.distortion_coefficients()(2, 0)*/,  // TODO: Assuming pinhole camera
      target_fx_inv,
      target_fy_inv,
      target_cx_inv,
      target_cy_inv,
      CUDAMatrix3x4(target_T_src.inverse().matrix3x4()),
      d_->target_rendered_color.get());
  
  cudaEventRecord(d_->color_reprojection_end_event, d_->stream);
  
  // Debug.
  if (FLAGS_vc_debug || FLAGS_vc_write_images) {
    cv::Mat_<cv::Vec4b> mat(d_->target_rendered_color->height(), d_->target_rendered_color->width());
    uchar4* buffer = new uchar4[d_->target_rendered_color->height() * d_->target_rendered_color->width()];
    d_->target_rendered_color->DebugDownload(buffer);
    for (int y = 0; y < d_->target_rendered_color->height(); ++ y) {
      for (int x = 0; x < d_->target_rendered_color->width(); ++ x) {
        const uchar4& value = buffer[x + y * d_->target_rendered_color->width()];
        // Flip R and B.
        mat(y, x) = cv::Vec4b(value.z, value.y, value.x, value.w);
      }
    }
    delete[] buffer;
    if (FLAGS_vc_debug) {
      cv::imshow("6 - Target frame rendered colors", mat);
    }
    if (FLAGS_vc_write_images) {
      std::ostringstream filename;
      filename << "debug_images/" << d_->output_frame_index << "_6_target_rendered_colors.png";
      cv::imwrite(filename.str(), mat);
    }
  }
  
  // In case of using the TSDF reconstruction, also render the reconstruction
  // into the target frame and use it to fill all regions that are not covered
  // by the meshed_inpainted_depth_map rendering.
  if (UsingMeshInput() && FLAGS_vc_render_tsdf_in_target) {
    Sophus::SE3f target_T_G = target_T_src * d_->G_T_src_C_.inverse();
    
    d_->tsdf_mesh_renderer_->BeginRenderingMeshesDepthAndColor(
        target_T_G, target_fx, target_fy, target_cx, target_cy, kMinDepthForRendering,
        kMaxDepthForRendering);
    d_->tsdf_mesh_renderer_->RenderMeshDepthAndColor(mesh_to_render->vertex_position_data, mesh_to_render->vertex_color_data, mesh_to_render->face_data, mesh_to_render->face_count);
    d_->tsdf_mesh_renderer_->EndRenderingMeshesDepthAndColor();
    
    cudaTextureObject_t tsdf_rendering_depth_texture =      
        d_->tsdf_mesh_renderer_->MapDepthResultAsTexture(
            cudaAddressModeClamp, cudaAddressModeClamp, cudaFilterModePoint,
            false, d_->stream);
    cudaTextureObject_t tsdf_rendering_color_texture =
        d_->tsdf_mesh_renderer_->MapColorResultAsTexture(
            cudaAddressModeClamp, cudaAddressModeClamp, cudaFilterModePoint,
            cudaReadModeElementType, false, d_->stream);
    
    // Debug.
    if (FLAGS_vc_debug || FLAGS_vc_write_images) {
      // Display rendered depth map.
      CUDABuffer<float> temp_depth_buffer(target_render_height_, target_render_width_);
      temp_depth_buffer.SetTo(tsdf_rendering_depth_texture, d_->stream);
      cudaDeviceSynchronize();
      std::ostringstream filename;
      filename << "debug_images/" << d_->output_frame_index << "_6b_target_tsdf_rendered_depth.png";
      CUDABufferVisualization(temp_depth_buffer).DisplayDepthMap(kMinDepthForDisplay, kMaxDepthForDisplay, "6 b - TSDF rendered depth", false, FLAGS_vc_write_images ? filename.str().c_str() : nullptr);
      
      // Display rendered gradient magnitudes for weights.
      CUDABuffer<uchar4> temp_color_buffer(target_render_height_, target_render_width_);
      temp_color_buffer.SetTo(tsdf_rendering_color_texture, d_->stream);
      cv::Mat_<cv::Vec4b> temp_color_buffer_mat(target_render_height_, target_render_width_);
      temp_color_buffer.Download(reinterpret_cast<cv::Mat_<uchar4>*>(&temp_color_buffer_mat));
      cudaDeviceSynchronize();
      cv::Mat_<cv::Vec3b> temp_color_buffer_mat2(target_render_height_, target_render_width_);
      for (int y = 0; y < target_render_height_; ++ y) {
        for (int x = 0; x < target_render_width_; ++ x) {
          const cv::Vec4b& color = temp_color_buffer_mat(y, x);
          temp_color_buffer_mat2(y, x) = cv::Vec3b(color(2), color(1), color(0));
        }
      }
      if (FLAGS_vc_debug) {
        cv::imshow("6 c - TSDF rendered colors", temp_color_buffer_mat2);
      }
      if (FLAGS_vc_write_images) {
        std::ostringstream filename;
        filename << "debug_images/" << d_->output_frame_index << "_6c_target_tsdf_rendered_colors.png";
        cv::imwrite(filename.str(), temp_color_buffer_mat2);
      }
    }
    
    // Insert result into output (d_->target_rendered_depth,
    // d_->target_rendered_color) for pixels which are not set yet.
    CopyValidToInvalidPixelsCUDA(
        d_->stream,
        tsdf_rendering_depth_texture,
        tsdf_rendering_color_texture,
        d_->target_rendered_depth.get(),
        d_->target_rendered_color.get());
    
    d_->tsdf_mesh_renderer_->UnmapColorResult(tsdf_rendering_color_texture, d_->stream);
    d_->tsdf_mesh_renderer_->UnmapDepthResult(tsdf_rendering_depth_texture, d_->stream);
  }
  
  // Render (some of) the last frame into pixels that are still invalid to get
  // temporal consistency. It is a good idea to fix the exposure time if using
  // this (or know the differences and adapt the colors accordingly).
  if (FLAGS_vc_ensure_target_frame_temporal_consistency && d_->have_previous_rendering_) {
    if (FLAGS_vc_inpainting_method == vc_inpainting_method::convolution) {
      ForwardReprojectToInvalidPixelsCUDA(
          d_->stream,
          CUDAMatrix3x4((target_T_src * d_->G_T_src_C_.cast<float>().inverse() * d_->last_target_T_G_.inverse()).matrix3x4()),
          d_->last_target_fx_inv, d_->last_target_fy_inv, d_->last_target_cx_inv, d_->last_target_cy_inv,
          *d_->target_inpainted_depth_map,
          *d_->target_inpainted_color_rgb,
          target_fx, target_fy, target_cx, target_cy,
          d_->target_rendered_depth.get(),
          d_->target_rendered_color.get());
    } else if (FLAGS_vc_inpainting_method == vc_inpainting_method::TV) {
      ForwardReprojectToInvalidPixelsCUDA(
          d_->stream,
          CUDAMatrix3x4((target_T_src * d_->G_T_src_C_.cast<float>().inverse() * d_->last_target_T_G_.inverse()).matrix3x4()),
          d_->last_target_fx_inv, d_->last_target_fy_inv, d_->last_target_cx_inv, d_->last_target_cy_inv,
          *d_->target_inpainted_depth_map,
          *d_->target_inpainted_color_r,
          *d_->target_inpainted_color_g,
          *d_->target_inpainted_color_b,
          target_fx, target_fy, target_cx, target_cy,
          d_->target_rendered_depth.get(),
          d_->target_rendered_color.get());
    }
    
    if (FLAGS_vc_debug || FLAGS_vc_write_images) {
      std::ostringstream filename;
      filename << "debug_images/" << d_->output_frame_index << "_6d_target_propagated_depth.png";
      CUDABufferVisualization(*d_->target_rendered_depth).DisplayDepthMap(kMinDepthForDisplay, kMaxDepthForDisplay, "6 d - Old frame propagated depth", false, FLAGS_vc_write_images ? filename.str().c_str() : nullptr);
      
      cv::Mat_<cv::Vec4b> temp_color_buffer_mat(target_render_height_, target_render_width_);
      d_->target_rendered_color->Download(reinterpret_cast<cv::Mat_<uchar4>*>(&temp_color_buffer_mat));
      cudaDeviceSynchronize();
      cv::Mat_<cv::Vec3b> temp_color_buffer_mat2(target_render_height_, target_render_width_);
      for (int y = 0; y < target_render_height_; ++ y) {
        for (int x = 0; x < target_render_width_; ++ x) {
          const cv::Vec4b& color = temp_color_buffer_mat(y, x);
          temp_color_buffer_mat2(y, x) = cv::Vec3b(color(2), color(1), color(0));
        }
      }
      if (FLAGS_vc_debug) {
        cv::imshow("6 e - Old frame propagated colors", temp_color_buffer_mat2);
      }
      if (FLAGS_vc_write_images) {
        std::ostringstream filename;
        filename << "debug_images/" << d_->output_frame_index << "_6e_target_propagated_colors.png";
        cv::imwrite(filename.str(), temp_color_buffer_mat2);
      }
    }
  }
  
  cudaEventRecord(d_->last_frame_reprojection_end_event, d_->stream);
  
  // Inpaint partial target frame depth map.
  uint32_t num_target_depth_pixels_to_inpaint = 0;
  if (FLAGS_vc_inpainting_method == vc_inpainting_method::convolution) {
  InpaintDepthMapWithConvolutionCUDA(
      d_->stream,
      FLAGS_vc_use_weights_for_inpainting,
      std::max(d_->target_inpainted_depth_map->width(), d_->target_inpainted_depth_map->height()),
      1e-3f,
      1.0f,
      rendered_intensity_texture,
      d_->target_rendered_depth_texture,
      d_->target_tv_max_change.get() /*used for max_change*/,
      d_->target_inpainted_depth_map.get(),
      d_->target_block_coordinates.get(),
      &num_target_depth_pixels_to_inpaint);
  } else if (FLAGS_vc_inpainting_method == vc_inpainting_method::TV) {
    InpaintDepthMapCUDA(d_->stream, kIMClassic,  // kIMAdaptive,
                        true, 800, 1e-3f, 1.0f, rendered_intensity_texture,
                        d_->target_rendered_depth_texture,
                        d_->target_tv_flag.get(),
                        d_->target_tv_dual_flag.get(),
                        d_->target_tv_dual_x.get(), d_->target_tv_dual_y.get(),
                        d_->target_tv_u_bar.get(), d_->target_tv_max_change_float.get(),
                        d_->target_inpainted_depth_map.get(),
                        d_->target_block_coordinates.get(), d_->target_block_activities.get());
  }

  cudaEventRecord(d_->target_depth_inpainting_end_event, d_->stream);
  
  // Debug.
  if (FLAGS_vc_debug || FLAGS_vc_write_images) {
    std::ostringstream filename;
    filename << "debug_images/" << d_->output_frame_index << "_7_target_inpainted_depth.png";
    CUDABufferVisualization(*d_->target_inpainted_depth_map).DisplayDepthMap(kMinDepthForDisplay, kMaxDepthForDisplay, "7 - Target frame inpainted depth", false, FLAGS_vc_write_images ? filename.str().c_str() : nullptr);
  }
  
  // Inpaint partial target frame color image.
  uint32_t num_target_color_pixels_to_inpaint = 0;
  if (FLAGS_vc_inpainting_method == vc_inpainting_method::convolution) {
    InpaintImageWithConvolutionCUDA(
        d_->stream,
        FLAGS_vc_use_weights_for_inpainting,
        std::max(d_->target_rendered_color->width(), d_->target_rendered_color->height()),
        1e-2f,
        rendered_intensity_texture,
        *d_->target_rendered_color,
        d_->target_color_tv_max_change.get()  /*used for max_change*/,
        d_->target_inpainted_color_rgb.get(),
        d_->target_block_coordinates.get(),
        &num_target_color_pixels_to_inpaint);
  } else if (FLAGS_vc_inpainting_method == vc_inpainting_method::TV) {
    InpaintImageCUDA(
        d_->stream,
        800,
        1e-2f,
        rendered_intensity_texture,
        *d_->target_rendered_color,
        d_->target_color_tv_flag.get(),
        d_->target_color_tv_dual_flag.get(),
        d_->target_color_tv_dual_x_r.get(),
        d_->target_color_tv_dual_x_g.get(),
        d_->target_color_tv_dual_x_b.get(),
        d_->target_color_tv_dual_y_r.get(),
        d_->target_color_tv_dual_y_g.get(),
        d_->target_color_tv_dual_y_b.get(),
        d_->target_color_tv_u_bar_r.get(),
        d_->target_color_tv_u_bar_g.get(),
        d_->target_color_tv_u_bar_b.get(),
        d_->target_color_tv_max_change_float.get(),
        d_->target_inpainted_color_r.get(),
        d_->target_inpainted_color_g.get(),
        d_->target_inpainted_color_b.get(),
        d_->target_block_coordinates.get());
  }
  
  cudaEventRecord(d_->target_color_inpainting_end_event, d_->stream);
  
  d_->mesh_renderer_->UnmapIntensityResult(rendered_intensity_texture, d_->stream);
  
  // Debug.
  if (FLAGS_vc_debug || FLAGS_vc_write_images) {
    cv::Mat_<cv::Vec3b> mat(d_->target_inpainted_color_rgb->height(), d_->target_inpainted_color_rgb->width());
    if (FLAGS_vc_inpainting_method == vc_inpainting_method::TV) {
      float* buffer_r = new float[d_->target_inpainted_color_rgb->height() * d_->target_inpainted_color_rgb->width()];
      float* buffer_g = new float[d_->target_inpainted_color_rgb->height() * d_->target_inpainted_color_rgb->width()];
      float* buffer_b = new float[d_->target_inpainted_color_rgb->height() * d_->target_inpainted_color_rgb->width()];
      d_->target_inpainted_color_r->DebugDownload(buffer_r);
      d_->target_inpainted_color_g->DebugDownload(buffer_g);
      d_->target_inpainted_color_b->DebugDownload(buffer_b);
      for (int y = 0; y < d_->target_inpainted_color_r->height(); ++ y) {
        for (int x = 0; x < d_->target_inpainted_color_r->width(); ++ x) {
          // Flip R and B.
          mat(y, x) = cv::Vec3b(
              255.99f * buffer_b[x + y * d_->target_inpainted_color_rgb->width()],
              255.99f * buffer_g[x + y * d_->target_inpainted_color_rgb->width()],
              255.99f * buffer_r[x + y * d_->target_inpainted_color_rgb->width()]);
        }
      }
      delete[] buffer_r;
      delete[] buffer_g;
      delete[] buffer_b;
    } else if (FLAGS_vc_inpainting_method == vc_inpainting_method::convolution) {
      uchar4* buffer_rgbx = new uchar4[d_->target_inpainted_color_rgb->height() * d_->target_inpainted_color_rgb->width()];
      d_->target_inpainted_color_rgb->DebugDownload(buffer_rgbx);
      for (int y = 0; y < d_->target_inpainted_color_rgb->height(); ++ y) {
        for (int x = 0; x < d_->target_inpainted_color_rgb->width(); ++ x) {
          const uchar4& rgbx = buffer_rgbx[x + y * d_->target_inpainted_color_rgb->width()];
          // Flip R and B.
          mat(y, x) = cv::Vec3b(rgbx.z, rgbx.y, rgbx.x);
        }
      }
      delete[] buffer_rgbx;
    }
    if (FLAGS_vc_debug) {
      cv::imshow("8 - Target frame inpainted colors", mat);
    }
    if (FLAGS_vc_write_images) {
      std::ostringstream filename;
      filename << "debug_images/" << d_->output_frame_index << "_8_target_inpainted_colors.png";
      cv::imwrite(filename.str(), mat);
    }
    
    // Evaluate against last frame.
    if (FLAGS_vc_evaluate_vs_previous_frame && !last_yuv_image_.empty()) {
      // Undistort previous frame.
      ColorImage undistorted_yuv_image = last_yuv_image_;  // TODO: Could perform image undistortion here
      
      // Scale undistorted frame to target render resolution.
      cv::resize(undistorted_yuv_image, undistorted_yuv_image, cv::Size(target_render_width_, target_render_height_), 0, 0, CV_INTER_AREA);
      
      // Show / write undistorted previous frame.
      if (FLAGS_vc_debug) {
        cv::imshow("Last frame (undistorted)", undistorted_yuv_image);
      }
      if (FLAGS_vc_write_images) {
        std::ostringstream filename;
        filename << "debug_images/" << d_->output_frame_index << "_last_frame_undistorted.png";
        cv::imwrite(filename.str(), undistorted_yuv_image);
      }
      
      // Difference image of target_inpainted_color (mat) to undistorted_yuv_image.
      cv::Mat_<cv::Vec3b> difference_image(target_render_height_, target_render_width_);
      for (int y = 0; y < target_render_height_; ++ y) {
        for (int x = 0; x < target_render_width_; ++ x) {
          bool masked_out = false;  //validity_map(y, x) != 0;
          if (masked_out) {
            difference_image(y, x) = cv::Vec3b(0, 0, 0);
          } else {
            difference_image(y, x) = cv::Vec3b(
                std::abs(mat(y, x)[0] - undistorted_yuv_image(y, x)[0]),
                std::abs(mat(y, x)[1] - undistorted_yuv_image(y, x)[1]),
                std::abs(mat(y, x)[2] - undistorted_yuv_image(y, x)[2]));
          }
        }
      }
      if (FLAGS_vc_debug) {
        cv::imshow("Difference: last frame (undistorted) - predicted last frame", difference_image);
      }
      if (FLAGS_vc_write_images) {
        std::ostringstream filename;
        filename << "debug_images/" << d_->output_frame_index << "_difference_image.png";
        cv::imwrite(filename.str(), difference_image);
      }
      
      if (FLAGS_vc_write_images) {
        // Compute total L1 differences.
        constexpr int kBorderSize = 40;
        float difference_sum = 0;
        int difference_pixel_count = 0;
        for (int y = kBorderSize; y < target_render_height_ - kBorderSize; ++ y) {
          for (int x = kBorderSize; x < target_render_width_ - kBorderSize; ++ x) {
            bool masked_out = false;  // validity_map(y, x) != 0;
            if (!masked_out) {
              difference_sum +=
                  difference_image(y, x)[0] + difference_image(y, x)[1] + difference_image(y, x)[2];
              ++ difference_pixel_count;
            }
          }
        }
        
        // Write to file.
        std::ostringstream filename;
        filename << "debug_images/" << d_->output_frame_index << "_difference.txt";
        std::ofstream file_stream(filename.str(), std::ios::out);
        file_stream << (difference_sum / difference_pixel_count) << std::endl;
        file_stream.close();
      }
    }
  }
  
  CHECK_OPENGL_NO_ERROR();
  if (FLAGS_vc_evaluate_stereo) {
    if (render_stereo_image) {
      glViewport(offset_x_ + width_ / 2, offset_y_, width_ / 2, height_);
    } else {
      glViewport(offset_x_, offset_y_, width_ / 2, height_);
    }
  } else {
    glViewport(offset_x_, offset_y_, width_, height_);
  }
  glClear(GL_DEPTH_BUFFER_BIT);
  CHECK_OPENGL_NO_ERROR();
  
  d_->have_previous_rendering_ = true;
  d_->last_target_T_G_ = target_T_src * d_->G_T_src_C_.cast<float>().inverse();
  d_->last_target_fx_inv = target_fx_inv;
  d_->last_target_fy_inv = target_fy_inv;
  d_->last_target_cx_inv = target_cx_inv;
  d_->last_target_cy_inv = target_cy_inv;
  
  // Render color image (target_inpainted_color) and depth image
  // (target_inpainted_depth_map) with a shader to the screen which sets the
  // fragment depth according to the depth map.
  DisplayOnScreen();
  
//   // Render augmented reality content.
//   if (FLAGS_vc_ar_demo) {
//     RenderARContent(
//         G_T_latest_C_timestamp,
//         target_T_src * d_->G_T_src_C_.inverse(),
//         target_fx, target_fy, target_cx, target_cy, kMinDepthForRendering,
//         kMaxDepthForRendering);
//   }
  
  if (FLAGS_vc_write_images || (render_stereo_image && FLAGS_vc_write_stereo_result)) {
    std::ostringstream filename;
    filename << "debug_images/" << d_->output_frame_index << "_9_final_image_with_AR_demo.png";
    
    cv::Mat_<cv::Vec3b> final_image_mat(height_, width_, cv::Vec3b(0, 0, 0));
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(offset_x_, offset_y_, width_, height_, GL_RGB, GL_UNSIGNED_BYTE,
                  final_image_mat.data);
    for (int y = 0; y < height_; ++ y) {
      for (int x = 0; x < width_; ++ x) {
        std::swap(final_image_mat(y, x)[0], final_image_mat(y, x)[2]);
      }
    }
    for (int y = 0; y < height_ / 2; ++ y) {
      for (int x = 0; x < width_; ++ x) {
        std::swap(final_image_mat(y, x), final_image_mat(height_ - 1 - y, x));
      }
    }
    cv::imwrite(filename.str(), final_image_mat);
  }
  
  if (FLAGS_vc_evaluate_vs_previous_frame && have_new_input) {
    last_yuv_image_ = new_yuv_image;
    last_yuv_image_G_T_C_ = d_->G_T_src_C_;
  }
  
  // Timing.
  if (FLAGS_vc_do_timings || FLAGS_vc_save_timings) {
    cudaEventSynchronize(d_->target_color_inpainting_end_event);
//     sm::timing::Timing* timing = &sm::timing::Timing::Instance();
    std::ofstream timing_file_stream;
    if (FLAGS_vc_save_timings) {
#ifdef ANDROID
      timing_file_stream.open("/sdcard/view_correction_timings.txt", std::ios::out | std::ios::app);
#else
      timing_file_stream.open("view_correction_timings.txt", std::ios::out | std::ios::app);
#endif
      timing_file_stream << d_->output_frame_index << std::endl;
    }
    float elapsed_time;
    
    if (have_new_input) {
      cudaEventElapsedTime(&elapsed_time, d_->render_or_upload_depth_start_event, d_->meshing_start_event);
//       timing->AddTime(timing->GetHandle("VC Meshing 1 - Uploading / rendering depth image"), 0.001 * elapsed_time);
      if (FLAGS_vc_save_timings) { timing_file_stream << "M1 " << elapsed_time << std::endl; }
      
      cudaEventElapsedTime(&elapsed_time, d_->meshing_start_event, d_->meshing_upload_end_event);
//       timing->AddTime(timing->GetHandle("VC Meshing 2 - Uploading color image"), 0.001 * elapsed_time);
      if (FLAGS_vc_save_timings) { timing_file_stream << "M2 " << elapsed_time << std::endl; }
      
      cudaEventElapsedTime(&elapsed_time, d_->meshing_upload_end_event, d_->meshing_downsampling_end_event);
//       timing->AddTime(timing->GetHandle("VC Meshing 3 - Downsampling color image"), 0.001 * elapsed_time);
      if (FLAGS_vc_save_timings) { timing_file_stream << "M3 " << elapsed_time << std::endl; }
      
      cudaEventElapsedTime(&elapsed_time, d_->meshing_downsampling_end_event, d_->meshing_gradient_mags_end_event);
//       timing->AddTime(timing->GetHandle("VC Meshing 4 - Computing gradient magnitudes"), 0.001 * elapsed_time);
      if (FLAGS_vc_save_timings) { timing_file_stream << "M4 " << elapsed_time << std::endl; }
      
      cudaEventElapsedTime(&elapsed_time, d_->meshing_gradient_mags_end_event, d_->meshing_inpainting_end_event);
//       timing->AddTime(timing->GetHandle("VC Meshing 5 - Inpainting depth"), 0.001 * elapsed_time);
      if (FLAGS_vc_save_timings) { timing_file_stream << "M5 " << elapsed_time << std::endl; }
      if (FLAGS_vc_save_timings) { timing_file_stream << "M5_pixel_count " << num_src_depth_pixels_to_inpaint << std::endl; }
      
      cudaEventElapsedTime(&elapsed_time, d_->meshing_inpainting_end_event, d_->meshing_end_event);
//       timing->AddTime(timing->GetHandle("VC Meshing 6 - Meshing inpainted depth"), 0.001 * elapsed_time);
      if (FLAGS_vc_save_timings) { timing_file_stream << "M6 " << elapsed_time << std::endl; }
    }
    
    cudaEventElapsedTime(&elapsed_time, d_->rendering_start_event, d_->rendering_end_event);
//     timing->AddTime(timing->GetHandle("VC Rendering 1 - Rendering meshed depth map"), 0.001 * elapsed_time);
    if (FLAGS_vc_save_timings) { timing_file_stream << "R1 " << elapsed_time << std::endl; }
    
    cudaEventElapsedTime(&elapsed_time, d_->rendering_end_event, d_->color_reprojection_end_event);
//     timing->AddTime(timing->GetHandle("VC Rendering 2 - Reprojecting color"), 0.001 * elapsed_time);
    if (FLAGS_vc_save_timings) { timing_file_stream << "R2 " << elapsed_time << std::endl; }
    
    cudaEventElapsedTime(&elapsed_time, d_->color_reprojection_end_event, d_->last_frame_reprojection_end_event);
//     timing->AddTime(timing->GetHandle("VC Rendering 2b - Reprojecting last frame"), 0.001 * elapsed_time);
    if (FLAGS_vc_save_timings) { timing_file_stream << "R2b " << elapsed_time << std::endl; }
    
    cudaEventElapsedTime(&elapsed_time, d_->last_frame_reprojection_end_event, d_->target_depth_inpainting_end_event);
//     timing->AddTime(timing->GetHandle("VC Rendering 3 - Inpainting depth"), 0.001 * elapsed_time);
    if (FLAGS_vc_save_timings) { timing_file_stream << "R3 " << elapsed_time << std::endl; }
    if (FLAGS_vc_save_timings) { timing_file_stream << "R3_pixel_count " << num_target_depth_pixels_to_inpaint << std::endl; }
    
    cudaEventElapsedTime(&elapsed_time, d_->target_depth_inpainting_end_event, d_->target_color_inpainting_end_event);
//     timing->AddTime(timing->GetHandle("VC Rendering 4 - Inpainting color"), 0.001 * elapsed_time);
    if (FLAGS_vc_save_timings) { timing_file_stream << "R4 " << elapsed_time << std::endl; }
    if (FLAGS_vc_save_timings) { timing_file_stream << "R4_pixel_count " << num_target_color_pixels_to_inpaint << std::endl; }
    
    if (FLAGS_vc_save_timings) {
      timing_file_stream.close();
    }
    
    constexpr int kLogTimingsInterval = 20;
    static int log_counter = 0;
    ++log_counter;
    if (log_counter % kLogTimingsInterval == 0) {
      // TODO: Could print timings to the screen here
    }
  }

  // Wait for keypress if debugging.
  if (FLAGS_vc_debug && !FLAGS_vc_write_images) {
    cv::waitKey(0);
  }

  return true;
}

void ViewCorrectionDisplay::DisplayOnScreen() {
  // Copy and convert the color image result to an RGB8 texture.
  cudaArray_t display_color_texture_array;
  CUDA_CHECKED_CALL(cudaGraphicsMapResources(1, &d_->display_color_texture_resource, d_->stream));
  CUDA_CHECKED_CALL(cudaGraphicsSubResourceGetMappedArray(
      &display_color_texture_array, d_->display_color_texture_resource, 0, 0));
  
  cudaResourceDesc display_color_texture_resource_desc;
  display_color_texture_resource_desc.resType = cudaResourceTypeArray;
  display_color_texture_resource_desc.res.array.array = display_color_texture_array;
  cudaSurfaceObject_t display_color_texture_surface;
  cudaCreateSurfaceObject(&display_color_texture_surface, &display_color_texture_resource_desc);
  if (FLAGS_vc_inpainting_method == vc_inpainting_method::TV) {
    CopyFloatColorImageToRgb8SurfaceCUDA(
        d_->stream,
        *d_->target_inpainted_color_r,
        *d_->target_inpainted_color_g,
        *d_->target_inpainted_color_b,
        display_color_texture_surface);
  } else if (FLAGS_vc_inpainting_method == vc_inpainting_method::convolution) {
    CopyColorImageToRgb8SurfaceCUDA(
        d_->stream,
        *d_->target_inpainted_color_rgb,
        display_color_texture_surface);
  }
  cudaDestroySurfaceObject(display_color_texture_surface);
  
  CUDA_CHECKED_CALL(cudaGraphicsUnmapResources(1, &d_->display_color_texture_resource, d_->stream));
  
  // Copy the depth map result to a float texture.
  cudaArray_t display_depth_texture_array;
  CUDA_CHECKED_CALL(cudaGraphicsMapResources(1, &d_->display_depth_texture_resource, d_->stream));
  CUDA_CHECKED_CALL(cudaGraphicsSubResourceGetMappedArray(
      &display_depth_texture_array, d_->display_depth_texture_resource, 0, 0));
  
  cudaMemcpy2DToArrayAsync(
      display_depth_texture_array,
      0, 0, /* offset */
      d_->target_inpainted_depth_map->ToCUDA().address(),
      d_->target_inpainted_depth_map->ToCUDA().pitch(),
      d_->target_inpainted_depth_map->width() * sizeof(float),
      d_->target_inpainted_depth_map->height(),
      cudaMemcpyDeviceToDevice,
      d_->stream);
  
//   cudaResourceDesc display_depth_texture_resource_desc;
//   display_depth_texture_resource_desc.resType = cudaResourceTypeArray;
//   display_depth_texture_resource_desc.res.array.array = display_depth_texture_array;
//   cudaSurfaceObject_t display_depth_texture_surface;
//   cudaCreateSurfaceObject(&display_depth_texture_surface, &display_depth_texture_resource_desc); 
//   CopyFloatColorImageToRgb8SurfaceCUDA(
//       d_->stream,
//       *d_->target_inpainted_color,
//       display_depth_texture_surface);
//   cudaDestroySurfaceObject(display_depth_texture_surface);
  
  CUDA_CHECKED_CALL(cudaGraphicsUnmapResources(1, &d_->display_depth_texture_resource, d_->stream));
  
  // Render textured quad.
  static GLfloat box[] = {
      -1.f, -1.f,
       1.f, -1.f,
      -1.f,  1.f,
       1.f,  1.f};
  static GLfloat texture_coord[] = {
      0.f, 1.f,
      1.f, 1.f,
      0.f, 0.f,
      1.f, 0.f};
  
  CHECK_OPENGL_NO_ERROR();
  glDisable(GL_CULL_FACE);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, d_->display_color_texture);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, d_->display_depth_texture);
  CHECK_OPENGL_NO_ERROR();

  glUseProgram(d_->display_shader_program);
  glEnableVertexAttribArray(d_->display_a_position_location);
  glVertexAttribPointer(d_->display_a_position_location, 2, GL_FLOAT, GL_FALSE,
                        2 * sizeof(float),  // NOLINT
                        box);
  CHECK_OPENGL_NO_ERROR();
  glEnableVertexAttribArray(d_->display_a_tex_coord_location);
  glVertexAttribPointer(d_->display_a_tex_coord_location, 2, GL_FLOAT, GL_FALSE,
                        2 * sizeof(float),  // NOLINT
                        texture_coord);
  CHECK_OPENGL_NO_ERROR();
  glUniform1i(d_->display_u_color_texture_location, 0);  // Use texture unit 0.
  glUniform1i(d_->display_u_depth_texture_location, 1);  // Use texture unit 1.
  CHECK_OPENGL_NO_ERROR();
  // Compute relevant entries of projection matrix for depth writing.
  const float proj22 = (kMaxDepthForRendering + kMinDepthForRendering) / (kMaxDepthForRendering - kMinDepthForRendering);
  const float proj23 = -(2 * kMaxDepthForRendering * kMinDepthForRendering) / (kMaxDepthForRendering - kMinDepthForRendering);
  glUniform1f(d_->display_u_proj22_location, proj22);
  glUniform1f(d_->display_u_proj23_location, proj23);
  CHECK_OPENGL_NO_ERROR();
  
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  CHECK_OPENGL_NO_ERROR();
  
  glDisableVertexAttribArray(d_->display_a_position_location);
  glDisableVertexAttribArray(d_->display_a_tex_coord_location);
  glUseProgram(0);
  CHECK_OPENGL_NO_ERROR();
}

// void ViewCorrectionDisplay::InitAR() {
//   // GFX engine.
//   startGFX();
//   
//   // Load mesh.
//   if (!d_->ar_renderer.Initialize()) {
//     LOG(ERROR) << "Cannot initialize AR renderer.";
//   }
// }

// void ViewCorrectionDisplay::RenderARContent(
//     double timestamp,
//     const Sophus::SE3f& target_T_G,
//     float fx, float fy, float cx, float cy, float min_depth, float max_depth) {
//   Eigen::Matrix4f projection_matrix;
//   float* matrix = projection_matrix.data();
//   matrix[0] = (2 * fx) / target_render_width_;
//   matrix[4] = 0;
//   matrix[8] = 2 * (0.5f + cx) / target_render_width_ - 1.0f;
//   matrix[12] = 0;
//   
//   matrix[1] = 0;
//   matrix[5] = -1 * ((2 * fy) / target_render_height_);
//   matrix[9] = -1 * (2 * (0.5f + cy) / target_render_height_ - 1.0f);
//   matrix[13] = 0;
//   
//   matrix[2] = 0;
//   matrix[6] = 0;
//   matrix[10] = (max_depth + min_depth) / (max_depth - min_depth);
//   matrix[14] = -(2 * max_depth * min_depth) / (max_depth - min_depth);
//   
//   matrix[3] = 0;
//   matrix[7] = 0;
//   matrix[11] = 1;
//   matrix[15] = 0;
//   
//   d_->ar_renderer.Render(timestamp, projection_matrix, target_T_G.T().cast<float>());
// }

void ViewCorrectionDisplay::ColorCameraPoseCallback(
    const Sophus::SE3f& G_T_C, uint64_t timestamp) {
  std::lock_guard<std::mutex> input_lock(input_mutex_);
  
  input_G_T_C_ = G_T_C;
  input_G_T_C_timestamp_ = timestamp;
}

void ViewCorrectionDisplay::YUVImageCallback(const ColorImage& image) {
  constexpr float kYUVImageCacheDuration = 1.0f;
  
  if (image.empty()) {
    return;
  }
  
  std::lock_guard<std::mutex> input_lock(input_mutex_);
  
  // Cache new image.
  input_yuv_images_.push_back(image);
  
  // Delete cached images which are too old.
  if (!input_yuv_images_.empty()) {
    const double min_timestamp = UsingDepthCameraInput() ?
        (kNanosecondsToSeconds * latest_depth_image_timestamp_ - kYUVImageCacheDuration) :
        (kNanosecondsToSeconds * input_yuv_images_.back().timestamp_ns() - kYUVImageCacheDuration);
    while (!input_yuv_images_.empty() &&
           kNanosecondsToSeconds * input_yuv_images_.front().timestamp_ns() < min_timestamp) {
      input_yuv_images_.erase(input_yuv_images_.begin());
    }
  }
}

void ViewCorrectionDisplay::DepthImageCallback(const DepthImage& image) {
  // Save input depth map.
  std::lock_guard<std::mutex> input_lock(input_mutex_);
  latest_depth_image_timestamp_ = image.timestamp_ns();
  input_depth_image_ = image;
}

void ViewCorrectionDisplay::MeshCallback(const std::shared_ptr<MeshStub>& mesh) {
  std::lock_guard<std::mutex> input_lock(input_mutex_);
  
  // Update the local mesh map with the incoming mesh.
  input_mesh_ = mesh;
}

void ViewCorrectionDisplay::RenderDepthImageFromMesh(const MeshStub& mesh,
                                                     const Sophus::SE3f& G_T_C) {
  Sophus::SE3f C_T_G = G_T_C.inverse();
  
  d_->src_mesh_renderer_->BeginRenderingMeshesDepth(
      C_T_G, depth_fx_, depth_fy_, depth_cx_, depth_cy_, kMinDepthForRendering,
      kMaxDepthForRendering);
  d_->src_mesh_renderer_->RenderMeshDepth(mesh.vertex_position_data, mesh.face_data, mesh.face_count);
  d_->src_mesh_renderer_->EndRenderingMeshesDepth();
}

void ViewCorrectionDisplay::CreateMeshedInpaintedDepthMap(
    const ColorImage& rgb_image,
    uint32_t* num_src_depth_pixels_to_inpaint) {
  cudaEventRecord(d_->meshing_start_event, d_->stream);
  
  if (FLAGS_vc_debug) {
    cv::imshow("0 b - RGB input", rgb_image);
  }
  if (FLAGS_vc_write_images) {
    std::ostringstream filename;
    filename << "debug_images/" << d_->output_frame_index << "_0b_source_yuv_input.png";
    cv::imwrite(filename.str(), rgb_image);
  }
  
  // Upload color image.
  // (NOTE: if it seems to be worth it, could try to use color image from GPU to
  //  avoid upload. Probably need to render it to a standard texture to be able
  //  to access it from CUDA?)
  
  // TODO: Color images were provided as YUV on Tango.
  //       We convert RGB to YUV here to simulate that, but it would be better to just use RGB.
  cv::Mat_<uint8_t> y_mat(rgb_image.rows, rgb_image.cols);
  for (int y = 0; y < rgb_image.rows; ++ y) {
    for (int x = 0; x < rgb_image.cols; ++ x) {
      const cv::Vec3b& rgb = rgb_image(y, x);
      y_mat(y, x) = 0.299 * rgb(0) + 0.587 * rgb(1) + 0.114 * rgb(2);  // NOTE: Assuming RGB order in rgb_image (not BGR!)
    }
  }
  
  cv::Mat_<uint16_t> uv_mat(rgb_image.rows / 2, rgb_image.cols / 2);
  for (int y = 0; y < uv_mat.rows; ++ y) {
    for (int x = 0; x < uv_mat.cols; ++ x) {
      const cv::Vec3b& rgb = rgb_image(2 * y, 2 * x);  // NOTE: could also average the corresponding 2x2 pixels
      uint8_t u = -0.169 * rgb(0) - 0.331 * rgb(1) + 0.499 * rgb(2) + 128;  // NOTE: Assuming RGB order in rgb_image (not BGR!)
      uint8_t v = 0.499 * rgb(0) - 0.418 * rgb(1) - 0.813 * rgb(2) + 128;  // NOTE: Assuming RGB order in rgb_image (not BGR!)
      
      uv_mat(y, x) = (u << 8) | v;
    }
  }
  
  d_->y_image_gpu[0]->UploadPitchedAsync(
      d_->stream, y_mat.step, y_mat.data);
  d_->uv_image_gpu->UploadPitchedAsync(
      d_->stream, uv_mat.step, reinterpret_cast<uint16_t*>(uv_mat.data));
  
  cudaEventRecord(d_->meshing_upload_end_event, d_->stream);
  
  // Downsample color image intensities to depth image resolution.
  for (int i = 0; i < static_cast<int>(d_->y_image_gpu.size()) - 1; ++ i) {
    // Downsample d_->y_image_gpu[i] to d_->y_image_gpu[i + 1].
    DownsampleImageToHalfSizeCUDA(
        d_->stream, *d_->y_image_gpu[i], d_->y_image_gpu[i + 1].get());
  }
  
  cudaEventRecord(d_->meshing_downsampling_end_event, d_->stream);
  
  // Debug.
  if (FLAGS_vc_debug || FLAGS_vc_write_images) {
    std::ostringstream filename;
    filename << "debug_images/" << d_->output_frame_index << "_1_source_y_input_downsampled.png";
    CUDABufferVisualization(*d_->y_image_gpu.back()).Display(0, 255, "1 - Y input (downsampled)", false, FLAGS_vc_write_images ? filename.str().c_str() : nullptr);
  }
  
  // Compute gradient magnitudes for downsampled color image.
  ComputeGradientMagnitudeDiv2CUDA(d_->stream, *d_->y_image_gpu.back(),
                                   d_->gradient_magnitude_div_sqrt2.get());
  
  cudaEventRecord(d_->meshing_gradient_mags_end_event, d_->stream);
  
  // Debug.
  if (FLAGS_vc_debug || FLAGS_vc_write_images) {
    std::ostringstream filename;
    filename << "debug_images/" << d_->output_frame_index << "_2_source_y_input_gradient_magnitudes.png";
    CUDABufferVisualization(*d_->gradient_magnitude_div_sqrt2).Display(0, 255, "2 - Y input gradient magnitudes", false, FLAGS_vc_write_images ? filename.str().c_str() : nullptr);
  }
  
  // Inpaint depth map using color image gradients as weights.
  const float depth_scaling_factor =
      UsingDepthCameraInput() ? (1.f / 1000.f * std::numeric_limits<uint16_t>::max()) : 1.f;
  if (FLAGS_vc_inpainting_method == vc_inpainting_method::convolution) {
    InpaintDepthMapWithConvolutionCUDA(
        d_->stream,
        FLAGS_vc_use_weights_for_inpainting,
        std::max(d_->src_inpainted_depth_map->width(), d_->src_inpainted_depth_map->height()),
        1e-3f,
        depth_scaling_factor,
        d_->gradient_magnitude_div_sqrt2_texture,
        d_->depth_image_gpu_texture,
        d_->src_tv_max_change.get() /*used for max_change*/,
        d_->src_inpainted_depth_map.get(),
        d_->src_block_coordinates.get(),
        num_src_depth_pixels_to_inpaint);
  } else if (FLAGS_vc_inpainting_method == vc_inpainting_method::TV) {
    InpaintDepthMapCUDA(
        d_->stream, kIMClassic,  // kIMAdaptive,
        true, 1500, 1e-3f, depth_scaling_factor,
        d_->gradient_magnitude_div_sqrt2_texture, d_->depth_image_gpu_texture,
        d_->src_tv_flag.get(),
        d_->src_tv_dual_flag.get(),
        d_->src_tv_dual_x.get(), d_->src_tv_dual_y.get(), d_->src_tv_u_bar.get(),
        d_->src_tv_max_change_float.get(), d_->src_inpainted_depth_map.get(),
        d_->src_block_coordinates.get(), d_->src_block_activities.get());
  }

  cudaEventRecord(d_->meshing_inpainting_end_event, d_->stream);
  
  // Debug.
  if (FLAGS_vc_debug || FLAGS_vc_write_images) {
    std::ostringstream filename;
    filename << "debug_images/" << d_->output_frame_index << "_3_source_inpainted_depth.png";
    CUDABufferVisualization(*d_->src_inpainted_depth_map).DisplayDepthMap(kMinDepthForDisplay, kMaxDepthForDisplay, "3 - Source frame inpainted depth", false, FLAGS_vc_write_images ? filename.str().c_str() : nullptr);
  }
  
  constexpr bool kDeleteAlmostOccludedPixels = false;
  if (kDeleteAlmostOccludedPixels) {
    // Delete all background pixels which are close to foreground pixels. This
    // helps to prevent foreground objects from being projected onto the
    // background if occlusion boundaries are imprecise.
    constexpr int occlusion_safety_radius = 2;  // In pixels.
    constexpr float occlusion_threshold = occlusion_safety_radius * 0.07f;  // In meters.
    DeleteAlmostOccludedPixelsCUDA(
        d_->stream,
        occlusion_safety_radius,
        occlusion_threshold,
        *d_->src_inpainted_depth_map,
        d_->src_inpainted_depth_map_alternate.get());
    std::swap(d_->src_inpainted_depth_map, d_->src_inpainted_depth_map_alternate);
    
    // Debug.
    if (FLAGS_vc_debug || FLAGS_vc_write_images) {
      std::ostringstream filename;
      filename << "debug_images/" << d_->output_frame_index << "_3b_source_masked_inpainted_depth.png";
      CUDABufferVisualization(*d_->src_inpainted_depth_map).DisplayDepthMap(kMinDepthForDisplay, kMaxDepthForDisplay, "3 - Source frame masked inpainted depth", false, FLAGS_vc_write_images ? filename.str().c_str() : nullptr);
    }
  }
  
  // Mesh inpainted depth map. Set colors differently on discontinuities.
  MeshDepthmapCUDA(
      d_->src_inpainted_depth_map->ToCUDA(),
      depth_fx_inv_, depth_fy_inv_, depth_cx_inv_, depth_cy_inv_,
      d_->stream,
      d_->vertex_buffer_resource,
      d_->color_buffer_resource,
      d_->index_buffer_resource);
  
  cudaEventRecord(d_->meshing_end_event, d_->stream);
  
  if (FLAGS_vc_evaluate_rgb_frame_inpainting) {
    // Also create a mesh from the original depth map.
    if (UsingMeshInput()) {
      CUDABuffer<float> temp_depth_buffer(d_->depth_image_gpu->height(),
                                          d_->depth_image_gpu->width());
      temp_depth_buffer.SetTo(d_->depth_image_gpu_texture, d_->stream);
      MeshDepthmapCUDA(
          temp_depth_buffer.ToCUDA(),
          depth_fx_inv_, depth_fy_inv_, depth_cx_inv_, depth_cy_inv_,
          d_->stream,
          d_->raw_vertex_buffer_resource,
          d_->raw_color_buffer_resource,
          d_->raw_index_buffer_resource);
    } else {
      MeshDepthmapMMCUDA(
          d_->depth_image_gpu->ToCUDA(),
          depth_fx_inv_, depth_fy_inv_, depth_cx_inv_, depth_cy_inv_,
          d_->stream,
          d_->raw_vertex_buffer_resource,
          d_->raw_color_buffer_resource,
          d_->raw_index_buffer_resource);
    }
  }
  
  d_->have_meshed_inpainted_depth_map = true;
}

bool ViewCorrectionDisplay::SetupTargetView(
    const Sophus::SE3f& G_T_latest_C,
    bool render_stereo_image,
    Sophus::SE3f* target_T_src,
    float* target_fx, float* target_fy, float* target_cx, float* target_cy) {
  if (FLAGS_vc_evaluate_vs_previous_frame) {
    if (!last_yuv_image_.empty()) {
      *target_T_src = (last_yuv_image_G_T_C_.inverse() * d_->G_T_src_C_).cast<float>();
      float scaling_factor_x = target_render_width_ / static_cast<float>(yuv_intrinsics_.width);
      float scaling_factor_y = target_render_height_ / static_cast<float>(yuv_intrinsics_.height);
      *target_fx = scaling_factor_x * yuv_intrinsics_.fx;
      *target_fy = scaling_factor_y * yuv_intrinsics_.fy;
      *target_cx = scaling_factor_x * yuv_intrinsics_.cx;
      *target_cy = scaling_factor_y * yuv_intrinsics_.cy;
    }
    return !last_yuv_image_.empty();
  }
  
  // Determine the observer position in the camera frame of this device, at the
  // current point in time.
  Eigen::Vector3f latest_camera_observer_position = Eigen::Vector3f::Zero();
  if (target_view_mode_ == TargetViewMode::kReceiveFromUDP) {
//     // DEBUG: fix the pose to some global point.
//     static int counter = 0;
//     static Eigen::Vector3f fixed_global_observer;
//     ++counter;
//     if (counter < 100) {
//       return false;
//     } else if (counter == 100) {
//       fixed_global_observer = G_T_latest_C.p().cast<float>();
//     }
//     latest_camera_observer_position =
//         G_T_latest_C.Inverse().cast<float>() * fixed_global_observer;
    
    // Listen for new observer pose messages.
    // d_->position_receiver.ReceiveNonBlocking();
    
    // Set to true to assume that the pose is a fixed observer position
    // (as with target_view_mode_ == TargetViewMode::kFixedOffset). Set to false
    // to asssume that it is a global map pose.
    constexpr bool kPoseIsRelative = false;
    
    if (kPoseIsRelative) {
      const char* position_filename = "/sdcard/view_correction_last_observer_position.txt";
      
      if (!d_->position_receiver.received_any_observer_position()) {
        // Use default pose or try to load it from file.
        latest_camera_observer_position =
            Eigen::Vector3f(-0.05f, 0.05f, -0.50f);  // Default.
        
        static bool pose_loaded = false;
        static Eigen::Vector3f loaded_pose;
        if (!pose_loaded) {
          // Try to load the last pose from file.
          std::ifstream file_stream(position_filename, std::ios::in);
          if (file_stream) {
            file_stream >> loaded_pose[0] >> loaded_pose[1] >> loaded_pose[2];
            file_stream.close();
          } else {
            loaded_pose = latest_camera_observer_position;
          }
          
          pose_loaded = true;
        }
        if (pose_loaded) {
          latest_camera_observer_position = loaded_pose;
        }
      } else {
        // Use received pose.
        if (latest_camera_observer_position !=
            d_->position_receiver.last_received_observer_position()) {
          latest_camera_observer_position =
              d_->position_receiver.last_received_observer_position();
          
          // Save the new position as the new default.
          std::ofstream file_stream(position_filename, std::ios::out);
          file_stream << latest_camera_observer_position[0] << " " << latest_camera_observer_position[1] << " " << latest_camera_observer_position[2] << std::endl;
          file_stream.close();
        }
        
        latest_camera_observer_position =
            d_->position_receiver.last_received_observer_position();
      }
    } else {
      // Check that we received at least one observer pose.
      if (!d_->position_receiver.received_any_observer_position()) {
        // No pose input yet.
        return false;
      }
      
      // Here, I assume that the observer position we get is at the current
      // time and expressed in the same coordinate system as this device's pose.
      const Eigen::Vector3f global_observer_position =
          d_->position_receiver.last_received_observer_position();
      
      // Transform the observer position into the camera frame of this device,
      // at the current point in time.
      latest_camera_observer_position =
          G_T_latest_C.inverse().cast<float>() * global_observer_position;
    }
    
    // // DEBUG: Log positions.
    // static bool initialized = false;
    // static std::ofstream debug_camera_stream;
    // static std::ofstream debug_observer_stream;
    // if (!initialized) {
    //   debug_camera_stream.open("/sdcard/debug_camera.obj", std::ios::out);
    //   debug_observer_stream.open("/sdcard/debug_observer.obj", std::ios::out);
    //   initialized = true;
    // }
    // debug_camera_stream << "v " << G_T_latest_C.p().transpose() << std::endl;
    // debug_observer_stream << "v " << global_observer_position.transpose() << std::endl;
  } else if (target_view_mode_ == TargetViewMode::kFixedOffset) {
    // Set a fixed observer position.
    if (FLAGS_vc_evaluate_stereo) {
      constexpr float kEyeDistance = 0.06f;
      if (render_stereo_image) {
        latest_camera_observer_position =
            Eigen::Vector3f(-0.05f + kEyeDistance / 2, 0.05f, -0.30f);
      } else {
        latest_camera_observer_position =
            Eigen::Vector3f(-0.05f - kEyeDistance / 2, 0.05f, -0.30f);
      }
    } else {
      // TODO: Get this from some configuration input instead of hardcoding it
      // NOTE: This is extremely close to the screen for debugging purposes, in effect the outer parts shown in the screen are not observed then.
      latest_camera_observer_position =
          Eigen::Vector3f(-0.05f, 0.05f, -0.05f /*-0.25f*/);
    }
  }

  // Screen corner positions in the color camera frame
  // TODO: Get this from some configuration input instead of hardcoding it
  Eigen::Vector3f screen_top_left(-126.81f / 1000.f, -1.4035f / 1000.f, -12.004f / 1000.f);
  Eigen::Vector3f screen_top_right(24.766f / 1000.f, -3.6885f / 1000.f, -10.633f / 1000.f);
  Eigen::Vector3f screen_bottom_left(-125.23f / 1000.f, 90.944f / 1000.f, -33.147f / 1000.f);
  
  if (FLAGS_vc_evaluate_stereo) {
    Eigen::Vector3f screen_right = screen_top_right - screen_top_left;
    if (render_stereo_image) {
      screen_top_left += screen_right / 2;
      screen_bottom_left += screen_right / 2;
    } else {
      screen_top_right -= screen_right / 2;
    }
  }

  // The Tango tablet's screen is 1920 x 1200 at 323 ppi.
  // The following calculates the pose and intrinsics based on all the
  // previously retrieved information.
  
  // Get the direction vectors of the screen coordinate system.
  Eigen::Vector3f screen_right = screen_top_right - screen_top_left;
  Eigen::Vector3f screen_down = screen_bottom_left - screen_top_left;
  Eigen::Vector3f screen_forward = screen_right.cross(screen_down);
  
  // Validate the calibration against the expected size of the screen.
  constexpr float kMetersPerInch = 0.0254f;
  float x_pixels_per_meter = 323.f / kMetersPerInch * target_render_width_ / (FLAGS_vc_evaluate_stereo ? (1920.f / 2) : 1920.f);
  float y_pixels_per_meter = 323.f / kMetersPerInch * target_render_height_ / 1200.f;
#ifndef ANDROID
  float expected_screen_width_meters = target_render_width_ / x_pixels_per_meter;
  float expected_screen_height_meters = target_render_height_ / y_pixels_per_meter;
  CHECK_NEAR(expected_screen_width_meters, screen_right.norm(), 0.001f);
  CHECK_NEAR(expected_screen_height_meters, screen_down.norm(), 0.001f);
  
  // Validate that the screen is rectangular.
  CHECK_NEAR(screen_right.dot(screen_down), 0.f, 0.001f);
#endif
  
  // Calculate the vector from the top left corner of the screen to the
  // observer position.
  Eigen::Vector3f screen_top_left_to_observer =
      latest_camera_observer_position - screen_top_left;
  
  // Calculate the principal point.
  float cx_meters = screen_right.dot(screen_top_left_to_observer) / screen_right.norm();
  float cy_meters = screen_down.dot(screen_top_left_to_observer) / screen_down.norm();
  *target_cx = cx_meters * x_pixels_per_meter - 0.5f;
  *target_cy = cy_meters * y_pixels_per_meter - 0.5f;
  
  // Calculate the point-to-plane distance of the observer to the image plane
  // to get fx, fy.
  float f_meters = fabs(screen_forward.dot(screen_top_left_to_observer) / screen_forward.norm());
  *target_fx = f_meters * x_pixels_per_meter;
  *target_fy = f_meters * y_pixels_per_meter;
  
  // Calculate the transformation from the current camera frame to the target
  // view. First, the (inverse) rotation matrix.
  Eigen::Matrix3f latest_C_T_target_R;
  latest_C_T_target_R.col(0) = screen_right / screen_right.norm();
  latest_C_T_target_R.col(1) = screen_down / screen_down.norm();
  latest_C_T_target_R.col(2) = screen_forward / screen_forward.norm();
  
  Sophus::SE3f target_T_latest_C;
  target_T_latest_C.setQuaternion(Eigen::Quaternionf(latest_C_T_target_R.transpose()));
  target_T_latest_C.translation() = latest_C_T_target_R.transpose() * (-1 * latest_camera_observer_position);
  
  // Get the transformation from the source frame to the color frame at the
  // current time.
  Sophus::SE3f latest_C_T_src_C =
      G_T_latest_C.cast<float>().inverse() * d_->G_T_src_C_.cast<float>();
  
  // Calculate the transformation from the source frame to the target view.
  *target_T_src = target_T_latest_C * latest_C_T_src_C;
  return true;
}

}  // namespace view_correction
