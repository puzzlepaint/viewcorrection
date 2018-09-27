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

#ifndef VIEW_CORRECTION_MESH_RENDERER_H_
#define VIEW_CORRECTION_MESH_RENDERER_H_

#ifdef ANDROID
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <glues/glu.h>
#else
#include <GL/glew.h>
#include <GL/gl.h>
#endif

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <sophus/se3.hpp>

#include "view_correction/forward_declarations.h"

namespace view_correction {

// Uses OpenGL for warping depthmaps between different camera viewpoints.
// Radial camera distortion is applied on the vertex level, so the geometry
// should be represented by a dense mesh for accurate warping.
// NOTE: Color rendering is modified to produce binary output!
class MeshRenderer {
 public:
  enum Type {
    kRenderDepthOnly = 0,
    kRenderDepthAndIntensity,
    kRenderDepthAndColor
  };
  
  // Creates OpenGL objects
  // (i.e., must be called with the correct OpenGL context).
  MeshRenderer(int width, int height, Type type);

  // Destructor.
  ~MeshRenderer();

  // Starts warping the depth map represented by the vertex and index buffer,
  // transformed into the target camera frame with the given transformation.
  // min_depth and max_depth give the z range for the view frustum used for
  // rendering. Their units correspond to what is passed in for camera poses
  // and vertices.
  void RenderMesh(
      GLuint vertex_buffer, GLuint color_buffer, GLuint index_buffer,
      int num_indices,
      const Sophus::SE3f& transformation,
      const float fx, const float fy, const float cx, const float cy,
      float min_depth, float max_depth);
  
  void BeginRenderingMeshesDepth(
      const Sophus::SE3f& transformation,
      const float fx, const float fy, const float cx, const float cy,
      float min_depth, float max_depth);
  void RenderMeshDepth(
      const void* vertex_data, const void* face_data, int num_faces);
  void EndRenderingMeshesDepth();
  
  void BeginRenderingMeshesDepthAndColor(
      const Sophus::SE3f& transformation,
      const float fx, const float fy, const float cx, const float cy,
      float min_depth, float max_depth);
  void RenderMeshDepthAndColor(
      const void* vertex_data, const void* color_data, const void* face_data, int num_faces);
  void EndRenderingMeshesDepthAndColor();

  // Returns the result color resource as a cudaGraphicsResource_t.
  cudaGraphicsResource_t result_resource_depth() const;
  cudaGraphicsResource_t result_resource_intensity() const;

  // Waits for the result to be available and returns it as a CUDA texture.
  // Call UnmapResult() when finished with using the result.
  cudaTextureObject_t MapDepthResultAsTexture(
      cudaTextureAddressMode address_mode_x,
      cudaTextureAddressMode address_mode_y, cudaTextureFilterMode filter_mode,
      bool normalized_coordinate_access, cudaStream_t stream);
  cudaTextureObject_t MapIntensityResultAsTexture(
      cudaTextureAddressMode address_mode_x,
      cudaTextureAddressMode address_mode_y, cudaTextureFilterMode filter_mode,
      cudaTextureReadMode read_mode,
      bool normalized_coordinate_access, cudaStream_t stream);
  cudaTextureObject_t MapColorResultAsTexture(
      cudaTextureAddressMode address_mode_x,
      cudaTextureAddressMode address_mode_y, cudaTextureFilterMode filter_mode,
      cudaTextureReadMode read_mode,
      bool normalized_coordinate_access, cudaStream_t stream);

  void UnmapDepthResult(cudaTextureObject_t texture, cudaStream_t stream);
  void UnmapIntensityResult(cudaTextureObject_t texture, cudaStream_t stream);
  void UnmapColorResult(cudaTextureObject_t texture, cudaStream_t stream);
  
  static int GetIndexCount(int width, int height);

 private:
  void CreateFrameBufferObject(Type type);
  
  void CreateVertexShader();
  void CreateFragmentShader();
  void CreateProgram();
  
  void CreateDepthVertexShader();
  void CreateDepthFragmentShader();
  void CreateDepthProgram();
  
  void CreateDepthAndColorVertexShader();
  void CreateDepthAndColorFragmentShader();
  void CreateDepthAndColorProgram();

  void SetupProjection(const Sophus::SE3f& transformation,
                       const float fx, const float fy, const float cx,
                       const float cy, float min_depth, float max_depth,
                       GLint u_projection_matrix_location,
                       GLint u_model_view_matrix_location);

  // Rendering target.
  GLuint frame_buffer_object_;
  GLuint depth_buffer_;
  GLuint rendertarget_0_texture_;
  GLuint rendertarget_1_texture_;
  cudaGraphicsResource_t rendertarget_0_resource_cuda_;
  cudaGraphicsResource_t rendertarget_1_resource_cuda_;
  
  // Depth + color shader.
  GLuint depth_color_fragment_shader_;
  GLuint depth_color_vertex_shader_;
  GLuint depth_color_shader_program_;
  GLint depth_color_a_position_location_;
  GLint depth_color_a_color_location_;
  GLint depth_color_u_model_view_matrix_location_;
  GLint depth_color_u_projection_matrix_location_;

  // Depth + intensity shader.
  GLuint fragment_shader_;
  GLuint vertex_shader_;
  GLuint shader_program_;
  GLint a_position_location_;
  GLint a_intensity_location_;
  GLint u_model_view_matrix_location_;
  GLint u_projection_matrix_location_;
  
  // Depth shader.
  GLuint depth_fragment_shader_;
  GLuint depth_vertex_shader_;
  GLuint depth_shader_program_;
  GLint depth_a_position_location_;
  GLint depth_u_model_view_matrix_location_;
  GLint depth_u_projection_matrix_location_;

  // Settings.
  int width_;
  int height_;
  Type type_;
};

}  // namespace view_correction

#endif  // VIEW_CORRECTION_MESH_RENDERER_H_
