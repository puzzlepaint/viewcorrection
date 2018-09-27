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

#include "view_correction/mesh_renderer.h"

#include "view_correction/cuda_util.h"
#include "view_correction/opengl_util.h"
#include "view_correction/util.h"

namespace view_correction {

MeshRenderer::MeshRenderer(int width, int height, Type type) {
  CHECK_OPENGL_NO_ERROR();
  width_ = width;
  height_ = height;
  type_ = type;
  
  CreateFrameBufferObject(type);
  
  if (type == kRenderDepthOnly) {
    CreateDepthVertexShader();
    CreateDepthFragmentShader();
    CreateDepthProgram();
  } else if (type == kRenderDepthAndIntensity) {
    CreateVertexShader();
    CreateFragmentShader();
    CreateProgram();
  } else if (type == kRenderDepthAndColor) {
    CreateDepthAndColorVertexShader();
    CreateDepthAndColorFragmentShader();
    CreateDepthAndColorProgram();
  }
}

MeshRenderer::~MeshRenderer() {
  CUDA_CHECKED_CALL(cudaGraphicsUnregisterResource(rendertarget_0_resource_cuda_));
  if (type_ != kRenderDepthOnly) {
    CUDA_CHECKED_CALL(cudaGraphicsUnregisterResource(rendertarget_1_resource_cuda_));
  }

  if (type_ == kRenderDepthOnly) {
    glDetachShader(depth_shader_program_, depth_vertex_shader_);
    glDetachShader(depth_shader_program_, depth_fragment_shader_);
    glDeleteShader(depth_vertex_shader_);
    glDeleteShader(depth_fragment_shader_);
    glDeleteProgram(depth_shader_program_);
  } else if (type_ == kRenderDepthAndIntensity) {
    glDetachShader(shader_program_, vertex_shader_);
    glDetachShader(shader_program_, fragment_shader_);
    glDeleteShader(vertex_shader_);
    glDeleteShader(fragment_shader_);
    glDeleteProgram(shader_program_);
  } else if (type_ == kRenderDepthAndColor) {
    glDetachShader(depth_color_shader_program_, depth_color_vertex_shader_);
    glDetachShader(depth_color_shader_program_, depth_color_fragment_shader_);
    glDeleteShader(depth_color_vertex_shader_);
    glDeleteShader(depth_color_fragment_shader_);
    glDeleteProgram(depth_color_shader_program_);
  }

  glDeleteTextures(1, &rendertarget_0_texture_);
  if (type_ != kRenderDepthOnly) {
    glDeleteTextures(1, &rendertarget_1_texture_);
  }
  glDeleteRenderbuffers(1, &depth_buffer_);

  glDeleteFramebuffers(1, &frame_buffer_object_);

  CHECK_OPENGL_NO_ERROR();
}

void MeshRenderer::RenderMesh(
    GLuint vertex_buffer, GLuint color_buffer, GLuint index_buffer,
    int num_indices,
    const Sophus::SE3f& transformation,
    const float fx, const float fy, const float cx, const float cy,
    float min_depth, float max_depth) {
  CHECK_EQ(type_, kRenderDepthAndIntensity);
  
  CHECK_OPENGL_NO_ERROR();
  
  // Set states.
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  // TODO(puzzlepaint): enable culling?
  glDisable(GL_CULL_FACE);
  CHECK_OPENGL_NO_ERROR();

  // Setup framebuffer and shaders.
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object_);
  CHECK_OPENGL_NO_ERROR();

  GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  glDrawBuffers(2, buffers);
  CHECK_OPENGL_NO_ERROR();

  // Clear buffers.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Render geometry.
  glUseProgram(shader_program_);

  SetupProjection(transformation, fx, fy, cx, cy, min_depth, max_depth,
                  u_projection_matrix_location_,
                  u_model_view_matrix_location_);
  CHECK_OPENGL_NO_ERROR();

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  glEnableVertexAttribArray(a_position_location_);
  glVertexAttribPointer(a_position_location_, 3, GL_FLOAT, GL_FALSE,
                        3 * sizeof(float),  // NOLINT
                        reinterpret_cast<char*>(0) + 0);
  CHECK_OPENGL_NO_ERROR();

  glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
  CHECK_OPENGL_NO_ERROR();

  glEnableVertexAttribArray(a_intensity_location_);
  CHECK_OPENGL_NO_ERROR();
  glVertexAttribPointer(a_intensity_location_, 1, GL_UNSIGNED_BYTE, GL_TRUE,
                        1 * sizeof(uint8_t),  // NOLINT
                        reinterpret_cast<char*>(0) + 0);
  CHECK_OPENGL_NO_ERROR();

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
  CHECK_OPENGL_NO_ERROR();

  glDrawElements(GL_TRIANGLE_STRIP, num_indices, GL_UNSIGNED_INT,
                 reinterpret_cast<char*>(0) + 0);
  CHECK_OPENGL_NO_ERROR();

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glDisableVertexAttribArray(a_intensity_location_);
  glDisableVertexAttribArray(a_position_location_);
  CHECK_OPENGL_NO_ERROR();

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  CHECK_OPENGL_NO_ERROR();
}

void MeshRenderer::BeginRenderingMeshesDepth(
    const Sophus::SE3f& transformation, const float fx,
    const float fy, const float cx, const float cy, float min_depth,
    float max_depth) {
  CHECK_EQ(type_, kRenderDepthOnly);
  
  // Set states.
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glEnable(GL_CULL_FACE);
  glFrontFace(GL_CW);
  CHECK_OPENGL_NO_ERROR();

  // Setup framebuffer and shaders.
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object_);
  GLenum buffers[] = {GL_COLOR_ATTACHMENT0};
  glDrawBuffers(1, buffers);
  glUseProgram(depth_shader_program_);
  CHECK_OPENGL_NO_ERROR();

  // Clear buffers.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Setup projection.
  SetupProjection(transformation, fx, fy, cx, cy, min_depth, max_depth,
                  depth_u_projection_matrix_location_,
                  depth_u_model_view_matrix_location_);
  
  // Setup input source.
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glEnableVertexAttribArray(depth_a_position_location_);
}

void MeshRenderer::RenderMeshDepth(
    const void* vertex_data, const void* face_data, int num_faces) {
  CHECK_EQ(type_, kRenderDepthOnly);
  
  // Render from CPU memory.
  glVertexAttribPointer(depth_a_position_location_, 3, GL_FLOAT, GL_FALSE,
                        3 * sizeof(float),  // NOLINT
                        vertex_data);
  glDrawElements(GL_TRIANGLES, 3 * num_faces, GL_UNSIGNED_INT,
                 face_data);
}

void MeshRenderer::EndRenderingMeshesDepth() {
  CHECK_EQ(type_, kRenderDepthOnly);
  
  glDisableVertexAttribArray(depth_a_position_location_);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  CHECK_OPENGL_NO_ERROR();
  
  // // DEBUG: display result.
  // glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object_);
  // cv::Mat_<float> texture_buffer(height_, width_);
  // glReadPixels(0, 0, width_, height_, GL_RED, GL_FLOAT,
  //               texture_buffer.data);
  // CHECK_OPENGL_NO_ERROR();
  // glBindFramebuffer(GL_FRAMEBUFFER, 0);
  // util::DisplayDepthmapInvDepthColored(
  //   texture_buffer, 0.8f,
  //   50.f, "reprojected depth", false);
  // cv::waitKey(1);
}

void MeshRenderer::BeginRenderingMeshesDepthAndColor(
    const Sophus::SE3f& transformation, const float fx,
    const float fy, const float cx, const float cy, float min_depth,
    float max_depth) {
  CHECK_EQ(type_, kRenderDepthAndColor);
  
  // Set states.
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glEnable(GL_CULL_FACE);
  glFrontFace(GL_CW);
  CHECK_OPENGL_NO_ERROR();

  // Setup framebuffer and shaders.
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object_);
  GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
  glDrawBuffers(2, buffers);
  glUseProgram(depth_color_shader_program_);
  CHECK_OPENGL_NO_ERROR();

  // Clear buffers.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Setup projection.
  SetupProjection(transformation, fx, fy, cx, cy, min_depth, max_depth,
                  depth_color_u_projection_matrix_location_,
                  depth_color_u_model_view_matrix_location_);
  
  // Setup input source.
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glEnableVertexAttribArray(depth_color_a_position_location_);
  glEnableVertexAttribArray(depth_color_a_color_location_);
  CHECK_OPENGL_NO_ERROR();
}

void MeshRenderer::RenderMeshDepthAndColor(
    const void* vertex_data, const void* color_data, const void* face_data, int num_faces) {
  CHECK_EQ(type_, kRenderDepthAndColor);
  
  // Render from CPU memory.
  glVertexAttribPointer(depth_color_a_position_location_, 3, GL_FLOAT, GL_FALSE,
                        3 * sizeof(float),  // NOLINT
                        vertex_data);
  glVertexAttribPointer(depth_color_a_color_location_, 3, GL_UNSIGNED_BYTE, GL_TRUE,
                        4 * sizeof(uint8_t),  // NOLINT
                        color_data);
  glDrawElements(GL_TRIANGLES, 3 * num_faces, GL_UNSIGNED_INT,
                 face_data);
}

void MeshRenderer::EndRenderingMeshesDepthAndColor() {
  CHECK_EQ(type_, kRenderDepthAndColor);
  
  glDisableVertexAttribArray(depth_color_a_position_location_);
  glDisableVertexAttribArray(depth_color_a_color_location_);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  CHECK_OPENGL_NO_ERROR();
}

cudaGraphicsResource_t MeshRenderer::result_resource_depth() const {
  return rendertarget_0_resource_cuda_;
}

cudaGraphicsResource_t MeshRenderer::result_resource_intensity() const {
  return rendertarget_1_resource_cuda_;
}

cudaTextureObject_t MeshRenderer::MapDepthResultAsTexture(
    cudaTextureAddressMode address_mode_x,
    cudaTextureAddressMode address_mode_y, cudaTextureFilterMode filter_mode,
    bool normalized_coordinate_access, cudaStream_t stream) {
  // Map the resource.
  cudaArray_t warped_depth_array;
  CUDA_CHECKED_CALL(cudaGraphicsMapResources(1, &rendertarget_0_resource_cuda_, stream));
  CUDA_CHECKED_CALL(cudaGraphicsSubResourceGetMappedArray(
      &warped_depth_array, rendertarget_0_resource_cuda_, 0, 0));

  // Create resource description.
  struct cudaResourceDesc warped_depth_resource_description;
  memset(&warped_depth_resource_description, 0,
         sizeof(warped_depth_resource_description));
  warped_depth_resource_description.resType = cudaResourceTypeArray;
  warped_depth_resource_description.res.array.array = warped_depth_array;

  // Create texture object.
  struct cudaTextureDesc warped_depth_texture_description;
  memset(&warped_depth_texture_description, 0,
         sizeof(warped_depth_texture_description));
  warped_depth_texture_description.addressMode[0] = address_mode_x;
  warped_depth_texture_description.addressMode[1] = address_mode_y;
  warped_depth_texture_description.filterMode = filter_mode;
  warped_depth_texture_description.readMode = cudaReadModeElementType;
  warped_depth_texture_description.normalizedCoords =
      normalized_coordinate_access ? 1 : 0;
  cudaTextureObject_t warped_depth_texture;
  CUDA_CHECKED_CALL(cudaCreateTextureObject(
      &warped_depth_texture, &warped_depth_resource_description,
      &warped_depth_texture_description, NULL));

  return warped_depth_texture;
}

cudaTextureObject_t MeshRenderer::MapIntensityResultAsTexture(
    cudaTextureAddressMode address_mode_x,
    cudaTextureAddressMode address_mode_y, cudaTextureFilterMode filter_mode,
    cudaTextureReadMode read_mode,
    bool normalized_coordinate_access, cudaStream_t stream) {
  // Map the resource.
  cudaArray_t result_array;
  CUDA_CHECKED_CALL(cudaGraphicsMapResources(1, &rendertarget_1_resource_cuda_, stream));
  CUDA_CHECKED_CALL(cudaGraphicsSubResourceGetMappedArray(
      &result_array, rendertarget_1_resource_cuda_, 0, 0));

  // Create resource description.
  struct cudaResourceDesc result_resource_description;
  memset(&result_resource_description, 0,
         sizeof(result_resource_description));
  result_resource_description.resType = cudaResourceTypeArray;
  result_resource_description.res.array.array = result_array;

  // Create texture object.
  struct cudaTextureDesc result_texture_description;
  memset(&result_texture_description, 0,
         sizeof(result_texture_description));
  result_texture_description.addressMode[0] = address_mode_x;
  result_texture_description.addressMode[1] = address_mode_y;
  result_texture_description.filterMode = filter_mode;
  result_texture_description.readMode = read_mode;
  result_texture_description.normalizedCoords =
      normalized_coordinate_access ? 1 : 0;
  cudaTextureObject_t result_texture;
  CUDA_CHECKED_CALL(cudaCreateTextureObject(
      &result_texture, &result_resource_description,
      &result_texture_description, NULL));

  return result_texture;
}

cudaTextureObject_t MeshRenderer::MapColorResultAsTexture(
    cudaTextureAddressMode address_mode_x,
    cudaTextureAddressMode address_mode_y,
    cudaTextureFilterMode filter_mode,
    cudaTextureReadMode read_mode,
    bool normalized_coordinate_access,
    cudaStream_t stream) {
  // Map the resource.
  cudaArray_t result_array;
  CUDA_CHECKED_CALL(cudaGraphicsMapResources(1, &rendertarget_1_resource_cuda_, stream));
  CUDA_CHECKED_CALL(cudaGraphicsSubResourceGetMappedArray(
      &result_array, rendertarget_1_resource_cuda_, 0, 0));

  // Create resource description.
  struct cudaResourceDesc result_resource_description;
  memset(&result_resource_description, 0,
         sizeof(result_resource_description));
  result_resource_description.resType = cudaResourceTypeArray;
  result_resource_description.res.array.array = result_array;

  // Create texture object.
  struct cudaTextureDesc result_texture_description;
  memset(&result_texture_description, 0,
         sizeof(result_texture_description));
  result_texture_description.addressMode[0] = address_mode_x;
  result_texture_description.addressMode[1] = address_mode_y;
  result_texture_description.filterMode = filter_mode;
  result_texture_description.readMode = read_mode;
  result_texture_description.normalizedCoords =
      normalized_coordinate_access ? 1 : 0;
  cudaTextureObject_t result_texture;
  CUDA_CHECKED_CALL(cudaCreateTextureObject(
      &result_texture, &result_resource_description,
      &result_texture_description, NULL));

  return result_texture;
}

void MeshRenderer::UnmapDepthResult(
    cudaTextureObject_t texture, cudaStream_t stream) {
  CUDA_CHECKED_CALL(cudaDestroyTextureObject(texture));
  CUDA_CHECKED_CALL(
      cudaGraphicsUnmapResources(1, &rendertarget_0_resource_cuda_, stream));
}

void MeshRenderer::UnmapIntensityResult(
    cudaTextureObject_t texture, cudaStream_t stream) {
  CUDA_CHECKED_CALL(cudaDestroyTextureObject(texture));
  CUDA_CHECKED_CALL(
      cudaGraphicsUnmapResources(1, &rendertarget_1_resource_cuda_, stream));
}

void MeshRenderer::UnmapColorResult(cudaTextureObject_t texture, cudaStream_t stream) {
  CUDA_CHECKED_CALL(cudaDestroyTextureObject(texture));
  CUDA_CHECKED_CALL(
      cudaGraphicsUnmapResources(1, &rendertarget_1_resource_cuda_, stream));
}

void MeshRenderer::CreateFrameBufferObject(Type type) {
  glGenFramebuffers(1, &frame_buffer_object_);
  CHECK_OPENGL_NO_ERROR();
  glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object_);
  CHECK_OPENGL_NO_ERROR();

  // Add a depth buffer to the frame buffer object.
  glGenRenderbuffers(1, &depth_buffer_);
  glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width_, height_);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, depth_buffer_);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  CHECK_OPENGL_NO_ERROR();

  // Add a color texture to the frame buffer object.
  // This class renders the depth to this color texture in addition to the depth
  // buffer because reading out the depth buffer did not seem to be supported in
  // OpenGL ES 2.0. This might have changed in later versions and
  // efficiency might benefit from removing the additional color texture.
  glGenTextures(1, &rendertarget_0_texture_);
  glBindTexture(GL_TEXTURE_2D, rendertarget_0_texture_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width_, height_, 0, GL_RED, GL_FLOAT,
               0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_2D, 0);
  CHECK_OPENGL_NO_ERROR();
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         rendertarget_0_texture_, 0);
  
  if (type == kRenderDepthAndIntensity) {
    glGenTextures(1, &rendertarget_1_texture_);
    glBindTexture(GL_TEXTURE_2D, rendertarget_1_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width_, height_, 0, GL_RED,
                GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    CHECK_OPENGL_NO_ERROR();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D,
                          rendertarget_1_texture_, 0);
  } else if (type == kRenderDepthAndColor) {
    glGenTextures(1, &rendertarget_1_texture_);
    glBindTexture(GL_TEXTURE_2D, rendertarget_1_texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    CHECK_OPENGL_NO_ERROR();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D,
                          rendertarget_1_texture_, 0);
  }

  // Verify frame buffer object creation.
  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  CHECK_EQ(static_cast<int>(status), GL_FRAMEBUFFER_COMPLETE);
  CHECK_OPENGL_NO_ERROR();
  
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // Prepare for CUDA interop.
  CUDA_CHECKED_CALL(cudaGraphicsGLRegisterImage(
      &rendertarget_0_resource_cuda_, rendertarget_0_texture_, GL_TEXTURE_2D,
      cudaGraphicsRegisterFlagsReadOnly));
  if (type != kRenderDepthOnly) {
    CUDA_CHECKED_CALL(cudaGraphicsGLRegisterImage(
        &rendertarget_1_resource_cuda_, rendertarget_1_texture_, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsReadOnly));
  }
}

void MeshRenderer::CreateVertexShader() {
  const std::string vertex_shader_src =
      "#version 300 es\n"
      "uniform mat4 u_model_view_matrix;\n"
      "uniform mat4 u_projection_matrix;\n"
      "in vec4 in_position;\n"
      "in float in_intensity;\n"
      "out float var_depth;\n"
      "out float var_intensity;\n"
      "void main() {\n"
      "   var_intensity = in_intensity;\n"
      "   vec4 local_point = u_model_view_matrix * in_position;\n"
      "   local_point.xyz /= local_point.w;\n"
      "   var_depth = local_point.z;\n"
      "   local_point.w = 1.0;\n"
      "   gl_Position = u_projection_matrix * local_point;\n"
      "}\n";

  vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
  const GLchar* vertex_shader_src_ptr =
      static_cast<const GLchar*>(vertex_shader_src.c_str());
  glShaderSource(vertex_shader_, 1, &vertex_shader_src_ptr, NULL);
  glCompileShader(vertex_shader_);

  GLint compiled;
  glGetShaderiv(vertex_shader_, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    GLint length;
    glGetShaderiv(vertex_shader_, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetShaderInfoLog(vertex_shader_, length, &length, log.get());
    LOG(FATAL) << "GL Shader Compilation Error: " << log.get();
  }
}

void MeshRenderer::CreateFragmentShader() {
  const std::string fragment_shader_src =
      "#version 300 es\n"
      "in highp float var_depth;\n"
      "in lowp float var_intensity;\n"
      "layout(location = 0) out highp float out_depth;\n"
      "layout(location = 1) out lowp float out_intensity;\n"
      "void main()\n"
      "{\n"
      "   out_depth = var_depth;\n"
      "   out_intensity = (var_intensity > 0.0) ? 1.0 : 0.0;\n"
      "}\n";

  fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
  const GLchar* fragment_shader_src_ptr =
      static_cast<const GLchar*>(fragment_shader_src.c_str());
  glShaderSource(fragment_shader_, 1, &fragment_shader_src_ptr, NULL);
  glCompileShader(fragment_shader_);

  GLint compiled;
  glGetShaderiv(fragment_shader_, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    GLint length;
    glGetShaderiv(fragment_shader_, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetShaderInfoLog(fragment_shader_, length, &length, log.get());
    LOG(FATAL) << "GL Shader Compilation Error: " << log.get();
  }
}

void MeshRenderer::CreateProgram() {
  shader_program_ = glCreateProgram();
  glAttachShader(shader_program_, fragment_shader_);
  glAttachShader(shader_program_, vertex_shader_);
  glLinkProgram(shader_program_);

  GLint linked;
  glGetProgramiv(shader_program_, GL_LINK_STATUS, &linked);
  if (!linked) {
    GLint length;
    glGetProgramiv(shader_program_, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetProgramInfoLog(shader_program_, length, &length, log.get());
    LOG(FATAL) << "GL Program Linker Error: " << log.get();
  }

  glUseProgram(shader_program_);
  CHECK_OPENGL_NO_ERROR();

  // Get attributes.
  a_position_location_ = glGetAttribLocation(shader_program_, "in_position");
  CHECK_OPENGL_NO_ERROR();
  CHECK_GE(a_position_location_, 0) << "Attribute needs to be used";
  a_intensity_location_ = glGetAttribLocation(shader_program_, "in_intensity");
  CHECK_OPENGL_NO_ERROR();
  CHECK_GE(a_intensity_location_, 0) << "Attribute needs to be used";

  u_model_view_matrix_location_ =
      glGetUniformLocation(shader_program_, "u_model_view_matrix");
  CHECK_OPENGL_NO_ERROR();

  u_projection_matrix_location_ =
      glGetUniformLocation(shader_program_, "u_projection_matrix");
  CHECK_OPENGL_NO_ERROR();
}

void MeshRenderer::CreateDepthVertexShader() {
  const std::string depth_vertex_shader_src =
      "#version 300 es\n"
      "uniform mat4 u_model_view_matrix;\n"
      "uniform mat4 u_projection_matrix;\n"
      "in vec4 in_position;\n"
      "out float var_depth;\n"
      "void main() {\n"
      "   vec4 local_point = u_model_view_matrix * in_position;\n"
      "   local_point.xyz /= local_point.w;\n"
      "   var_depth = local_point.z;\n"
      "   local_point.w = 1.0;\n"
      "   gl_Position = u_projection_matrix * local_point;\n"
      "}\n";

  depth_vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
  const GLchar* depth_vertex_shader_src_ptr =
      static_cast<const GLchar*>(depth_vertex_shader_src.c_str());
  glShaderSource(depth_vertex_shader_, 1, &depth_vertex_shader_src_ptr, NULL);
  glCompileShader(depth_vertex_shader_);

  GLint compiled;
  glGetShaderiv(depth_vertex_shader_, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    GLint length;
    glGetShaderiv(depth_vertex_shader_, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetShaderInfoLog(depth_vertex_shader_, length, &length, log.get());
    LOG(FATAL) << "GL Shader Compilation Error: " << log.get();
  }
}

void MeshRenderer::CreateDepthFragmentShader() {
  const std::string depth_fragment_shader_src =
      "#version 300 es\n"
      "in highp float var_depth;\n"
      "layout(location = 0) out highp float out_depth;\n"
      "void main()\n"
      "{\n"
      "   out_depth = var_depth;\n"
      "}\n";

  depth_fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
  const GLchar* depth_fragment_shader_src_ptr =
      static_cast<const GLchar*>(depth_fragment_shader_src.c_str());
  glShaderSource(depth_fragment_shader_, 1, &depth_fragment_shader_src_ptr, NULL);
  glCompileShader(depth_fragment_shader_);

  GLint compiled;
  glGetShaderiv(depth_fragment_shader_, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    GLint length;
    glGetShaderiv(depth_fragment_shader_, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetShaderInfoLog(depth_fragment_shader_, length, &length, log.get());
    LOG(FATAL) << "GL Shader Compilation Error: " << log.get();
  }
}

void MeshRenderer::CreateDepthProgram() {
  depth_shader_program_ = glCreateProgram();
  glAttachShader(depth_shader_program_, depth_fragment_shader_);
  glAttachShader(depth_shader_program_, depth_vertex_shader_);
  glLinkProgram(depth_shader_program_);

  GLint linked;
  glGetProgramiv(depth_shader_program_, GL_LINK_STATUS, &linked);
  if (!linked) {
    GLint length;
    glGetProgramiv(depth_shader_program_, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetProgramInfoLog(depth_shader_program_, length, &length, log.get());
    LOG(FATAL) << "GL Program Linker Error: " << log.get();
  }

  glUseProgram(depth_shader_program_);
  CHECK_OPENGL_NO_ERROR();

  // Get attributes.
  depth_a_position_location_ = glGetAttribLocation(depth_shader_program_, "in_position");
  CHECK_OPENGL_NO_ERROR();
  CHECK_GE(depth_a_position_location_, 0) << "Attribute needs to be defined, used and of attribute type.";

  depth_u_model_view_matrix_location_ =
      glGetUniformLocation(depth_shader_program_, "u_model_view_matrix");
  CHECK_OPENGL_NO_ERROR();

  depth_u_projection_matrix_location_ =
      glGetUniformLocation(depth_shader_program_, "u_projection_matrix");
  CHECK_OPENGL_NO_ERROR();
}

void MeshRenderer::CreateDepthAndColorVertexShader() {
  const std::string vertex_shader_src =
      "#version 300 es\n"
      "uniform mat4 u_model_view_matrix;\n"
      "uniform mat4 u_projection_matrix;\n"
      "in vec4 in_position;\n"
      "in vec3 in_color;\n"
      "out float var_depth;\n"
      "out vec3 var_color;\n"
      "void main() {\n"
      "   var_color = in_color;\n"
      "   vec4 local_point = u_model_view_matrix * in_position;\n"
      "   local_point.xyz /= local_point.w;\n"
      "   var_depth = local_point.z;\n"
      "   local_point.w = 1.0;\n"
      "   gl_Position = u_projection_matrix * local_point;\n"
      "}\n";

  depth_color_vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
  const GLchar* vertex_shader_src_ptr =
      static_cast<const GLchar*>(vertex_shader_src.c_str());
  glShaderSource(depth_color_vertex_shader_, 1, &vertex_shader_src_ptr, NULL);
  glCompileShader(depth_color_vertex_shader_);

  GLint compiled;
  glGetShaderiv(depth_color_vertex_shader_, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    GLint length;
    glGetShaderiv(depth_color_vertex_shader_, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetShaderInfoLog(depth_color_vertex_shader_, length, &length, log.get());
    LOG(FATAL) << "GL Shader Compilation Error: " << log.get();
  }
}

void MeshRenderer::CreateDepthAndColorFragmentShader() {
  const std::string fragment_shader_src =
      "#version 300 es\n"
      "in highp float var_depth;\n"
      "in lowp vec3 var_color;\n"
      "layout(location = 0) out highp float out_depth;\n"
      "layout(location = 1) out lowp vec4 out_color;\n"
      "void main()\n"
      "{\n"
      "   out_depth = var_depth;\n"
      "   out_color = vec4(var_color, 1);\n"
      "}\n";

  depth_color_fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
  const GLchar* fragment_shader_src_ptr =
      static_cast<const GLchar*>(fragment_shader_src.c_str());
  glShaderSource(depth_color_fragment_shader_, 1, &fragment_shader_src_ptr, NULL);
  glCompileShader(depth_color_fragment_shader_);

  GLint compiled;
  glGetShaderiv(depth_color_fragment_shader_, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    GLint length;
    glGetShaderiv(depth_color_fragment_shader_, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetShaderInfoLog(depth_color_fragment_shader_, length, &length, log.get());
    LOG(FATAL) << "GL Shader Compilation Error: " << log.get();
  }
}

void MeshRenderer::CreateDepthAndColorProgram() {
  depth_color_shader_program_ = glCreateProgram();
  glAttachShader(depth_color_shader_program_, depth_color_fragment_shader_);
  glAttachShader(depth_color_shader_program_, depth_color_vertex_shader_);
  glLinkProgram(depth_color_shader_program_);

  GLint linked;
  glGetProgramiv(depth_color_shader_program_, GL_LINK_STATUS, &linked);
  if (!linked) {
    GLint length;
    glGetProgramiv(depth_color_shader_program_, GL_INFO_LOG_LENGTH, &length);
    std::unique_ptr<GLchar[]> log(
        reinterpret_cast<GLchar*>(new uint8_t[length]));
    glGetProgramInfoLog(depth_color_shader_program_, length, &length, log.get());
    LOG(FATAL) << "GL Program Linker Error: " << log.get();
  }

  glUseProgram(depth_color_shader_program_);
  CHECK_OPENGL_NO_ERROR();

  // Get attributes.
  depth_color_a_position_location_ = glGetAttribLocation(depth_color_shader_program_, "in_position");
  CHECK_OPENGL_NO_ERROR();
  CHECK_GE(depth_color_a_position_location_, 0) << "Attribute needs to be used";
  depth_color_a_color_location_ = glGetAttribLocation(depth_color_shader_program_, "in_color");
  CHECK_OPENGL_NO_ERROR();
  CHECK_GE(depth_color_a_color_location_, 0) << "Attribute needs to be used";

  depth_color_u_model_view_matrix_location_ =
      glGetUniformLocation(depth_color_shader_program_, "u_model_view_matrix");
  CHECK_OPENGL_NO_ERROR();

  depth_color_u_projection_matrix_location_ =
      glGetUniformLocation(depth_color_shader_program_, "u_projection_matrix");
  CHECK_OPENGL_NO_ERROR();
}

void MeshRenderer::SetupProjection(
    const Sophus::SE3f& transformation,
    const float fx, const float fy, const float cx, const float cy,
    float min_depth, float max_depth, GLint u_projection_matrix_location,
    GLint u_model_view_matrix_location) {
  CHECK_GT(max_depth, min_depth);
  CHECK_GT(min_depth, 0);

  // Row-wise projection matrix construction.
  float matrix[16];
  matrix[0] = (2 * fx) / width_;
  matrix[4] = 0;
  matrix[8] = 2 * (0.5f + cx) / width_ - 1.0f;
  matrix[12] = 0;

  matrix[1] = 0;
  matrix[5] = (2 * fy) / height_;
  matrix[9] = 2 * (0.5f + cy) / height_ - 1.0f;
  matrix[13] = 0;

  matrix[2] = 0;
  matrix[6] = 0;
  matrix[10] = (max_depth + min_depth) / (max_depth - min_depth);
  matrix[14] = -(2 * max_depth * min_depth) / (max_depth - min_depth);

  matrix[3] = 0;
  matrix[7] = 0;
  matrix[11] = 1;
  matrix[15] = 0;

  glUniformMatrix4fv(u_projection_matrix_location, 1, GL_FALSE, matrix);
  CHECK_OPENGL_NO_ERROR();

  // Model-view matrix construction.
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      matrix[i + j * 4] = transformation.rotationMatrix()(i, j);
    }
    matrix[i + 12] = transformation.translation()(i);
  }
  matrix[3] = 0;
  matrix[7] = 0;
  matrix[11] = 0;
  matrix[15] = 1;

  glUniformMatrix4fv(u_model_view_matrix_location, 1, GL_FALSE, matrix);
  CHECK_OPENGL_NO_ERROR();

  // Set viewport.
  glViewport(0, 0, width_, height_);
  CHECK_OPENGL_NO_ERROR();
}

int MeshRenderer::GetIndexCount(int width, int height) {
  constexpr int kStartIndexCount = 2;
  const int newline_index_count = (height - 2) * 4;
  const int quad_index_count = (width - 1) * (height - 1) * 4;
  return quad_index_count + newline_index_count + kStartIndexCount;
}

}  // namespace view_correction
