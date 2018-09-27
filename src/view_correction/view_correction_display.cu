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

#include "view_correction/view_correction_display.cuh"

#include <glog/logging.h>

#include "view_correction/cuda_util.h"
#include "view_correction/helper_math.h"

namespace view_correction {
constexpr float kSqrt2 = 1.4142135623731f;

__global__ void DownsampleImageToHalfSizeCUDAKernel(
    CUDABuffer_<uint8_t> target,
    cudaTextureObject_t source_texture) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int width = target.width();
  const int height = target.height();
  if (x < width && y < height) {
    target(y, x) = static_cast<uint8_t>(
        255.0f * tex2D<float>(source_texture, 2 * (x + 0.5f), 2 * (y + 0.5f))
        + 0.5f);
  }
}

void DownsampleImageToHalfSizeCUDA(
    cudaStream_t stream,
    const CUDABuffer<uint8_t>& source,
    CUDABuffer<uint8_t>* target) {
  CHECK_NOTNULL(target);
  cudaTextureObject_t source_texture;
  source.CreateTextureObject(
      cudaAddressModeClamp, cudaAddressModeClamp, cudaFilterModeLinear,
      cudaReadModeNormalizedFloat, false, &source_texture);
  
  // Run kernel.
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 gridDim(cuda_util::GetBlockCount(target->width(), kBlockWidth),
               cuda_util::GetBlockCount(target->height(), kBlockHeight));
  dim3 blockDim(kBlockWidth, kBlockHeight);
  DownsampleImageToHalfSizeCUDAKernel<<<gridDim, blockDim, 0, stream>>>(
      target->ToCUDA(), source_texture);
  
  cudaDestroyTextureObject(source_texture);
}

__global__ void ComputeGradientMagnitudeDiv2CUDAKernel(
    CUDABuffer_<uint8_t> output,
    cudaTextureObject_t image_texture) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int width = output.width();
  const int height = output.height();
  if (x < width && y < height) {
    // NOTE: Use of shared memory might lead to slight performance benefit.
    const int intensity = tex2D<uint8_t>(image_texture, x + 0, y + 0);
    const int intensity_px = tex2D<uint8_t>(image_texture, x + 1, y + 0);
    const int intensity_py = tex2D<uint8_t>(image_texture, x + 0, y + 1);
    
    const float dx = intensity_px - intensity;
    const float dy = intensity_py - intensity;
    const float gradient_magnitude = sqrtf(dx * dx + dy * dy) / kSqrt2;
    
    output(y, x) = min(255u, static_cast<uint8_t>(gradient_magnitude));
  }
}

void ComputeGradientMagnitudeDiv2CUDA(
    cudaStream_t stream,
    const CUDABuffer<uint8_t>& image,
    CUDABuffer<uint8_t>* output) {
  CHECK_NOTNULL(output);
  cudaTextureObject_t image_texture;
  image.CreateTextureObject(
      cudaAddressModeClamp, cudaAddressModeClamp, cudaFilterModePoint,
      cudaReadModeElementType, false, &image_texture);
  
  // Run kernel.
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  dim3 gridDim(cuda_util::GetBlockCount(output->width(), kBlockWidth),
               cuda_util::GetBlockCount(output->height(), kBlockHeight));
  dim3 blockDim(kBlockWidth, kBlockHeight);
  ComputeGradientMagnitudeDiv2CUDAKernel<<<gridDim, blockDim, 0, stream>>>(
      output->ToCUDA(), image_texture);
  
  cudaDestroyTextureObject(image_texture);
}


__global__ void MeshDepthmapCUDAKernel(
    const float fx_inv,
    const float fy_inv,
    const float cx_inv,
    const float cy_inv,
    CUDABuffer_<float> depthmap,
    float* vertex_buffer,
    uint8_t* color_buffer,
    uint32_t* index_buffer) {
  constexpr float kJumpThreshold = 0.070f; // In m.  TODO(puzzlepaint): 1) Make tunable. 2) This should depend on the depth instead of being constant.
  
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int width = depthmap.width();
  const int height = depthmap.height();

  if (x < width && y < height) {
    // Get depths and normalized image coordinates.
    const float2 nxy = make_float2(fx_inv * x + cx_inv, fy_inv * y + cy_inv);
    const float depth = depthmap(y, x);
    // TODO: This might benefit from shared memory or texture use.
    const float depth_top = (y > 0) ? depthmap(y - 1, x) : depth;
    const float depth_left = (x > 0) ? depthmap(y, x - 1) : depth;
    const float depth_bottom = (y < height - 1) ? depthmap(y + 1, x) : depth;
    const float depth_right = (x < width - 1) ? depthmap(y, x + 1) : depth;
    
    // Write vertex position.
    const int vertex_index = 3 * (x + y * width);
    vertex_buffer[vertex_index + 0] = depth * nxy.x;
    vertex_buffer[vertex_index + 1] = depth * nxy.y;
    vertex_buffer[vertex_index + 2] = depth;
    
    // Write vertex color according to having a depth jump at the vertex.
    const bool foreground_boundary_vertex =
        depth_bottom - depth >= kJumpThreshold ||
        depth_right - depth >= kJumpThreshold ||
        depth_top - depth >= kJumpThreshold ||
        depth_left - depth >= kJumpThreshold;
    color_buffer[x + y * width] = foreground_boundary_vertex ? 255 : 0;
    
    // Write indices.
    const int index11 = (x + y * width);
    const int index10 = index11 - 1;
    const int index01 = index11 - width;
    if (x > 0 && y > 0) {
      const float depth_top_left = depthmap(y - 1, x - 1);
      float depth_difference =
          max(max(fabs(depth_top_left - depth_left),
                  fabs(depth_top_left - depth_top)),
              max(fabs(depth_top - depth),
                  fabs(depth_left - depth)));
      const bool create_quad =
          depth_difference < kJumpThreshold &&
          depth_top_left > 0.f && depth_left > 0.f &&
          depth_top > 0.f && depth > 0.f;
      const int i = 2 + 4 * ((x - 1) + (y - 1) * width);
      if (create_quad) {
        // The last two vertices are actually unnecessary, but we need to write
        // a fixed number of vertices here, so we need to include them.
        index_buffer[i + 0] = index01;
        index_buffer[i + 1] = index11;
        index_buffer[i + 2] = index01; // Unnecessary.
        index_buffer[i + 3] = index11; // Unnecessary.
      } else {
        index_buffer[i + 0] = index10;
        index_buffer[i + 1] = index01;
        index_buffer[i + 2] = index01;
        index_buffer[i + 3] = index11;
      }
    } else if (x == 0) {
      // TODO(puzzlepaint): This could be written in an extra kernel for
      // perhaps slightly better performance.
      if (y == 0) {
        // Write start.
        index_buffer[0] = 0;
        index_buffer[1] = width;
      } else if (y < height - 1) {
        // Write newline.
        const int i = 2 + y * 4 * width - 4;
        index_buffer[i + 0] = index11 - 1 + width;
        index_buffer[i + 1] = index11;
        index_buffer[i + 2] = index11;
        index_buffer[i + 3] = index11 + width;
      }
    }
  }
}

void MeshDepthmapCUDA(
    const CUDABuffer_<float>& depthmap,
    const float fx_inv,
    const float fy_inv,
    const float cx_inv,
    const float cy_inv,
    cudaStream_t stream,
    cudaGraphicsResource_t vertex_buffer,
    cudaGraphicsResource_t color_buffer,
    cudaGraphicsResource_t index_buffer) {
  // Map buffers.
  float* vertex_buffer_pointer;
  size_t vertex_buffer_size;
  CUDA_CHECKED_CALL(cudaGraphicsMapResources(1, &vertex_buffer, stream));
  CUDA_CHECKED_CALL(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&vertex_buffer_pointer), &vertex_buffer_size,
      vertex_buffer));
  
  uint8_t* color_buffer_pointer;
  size_t color_buffer_size;
  CUDA_CHECKED_CALL(cudaGraphicsMapResources(1, &color_buffer, stream));
  CUDA_CHECKED_CALL(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&color_buffer_pointer), &color_buffer_size,
      color_buffer));
  
  uint32_t* index_buffer_pointer;
  size_t index_buffer_size;
  CUDA_CHECKED_CALL(cudaGraphicsMapResources(1, &index_buffer, stream));
  CUDA_CHECKED_CALL(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&index_buffer_pointer), &index_buffer_size,
      index_buffer));
  
  // Run kernel.
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  const dim3 grid_dim(cuda_util::GetBlockCount(depthmap.width(),
                                               kBlockWidth),
                      cuda_util::GetBlockCount(depthmap.height(),
                                               kBlockHeight));
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  MeshDepthmapCUDAKernel<<<grid_dim, block_dim, 0, stream>>>(
      fx_inv, fy_inv, cx_inv, cy_inv, depthmap, vertex_buffer_pointer,
      color_buffer_pointer, index_buffer_pointer);
  CHECK_CUDA_NO_ERROR();
  
  // Unmap buffers.
  CUDA_CHECKED_CALL(cudaGraphicsUnmapResources(1, &index_buffer, stream));
  CUDA_CHECKED_CALL(cudaGraphicsUnmapResources(1, &color_buffer, stream));
  CUDA_CHECKED_CALL(cudaGraphicsUnmapResources(1, &vertex_buffer, stream));
}

__global__ void MeshDepthmapMMCUDAKernel(
    const float fx_inv,
    const float fy_inv,
    const float cx_inv,
    const float cy_inv,
    CUDABuffer_<uint16_t> depthmap_in_mm,
    float* vertex_buffer,
    uint8_t* color_buffer,
    uint32_t* index_buffer) {
  constexpr float kJumpThreshold = 0.070f; // In m.  TODO(puzzlepaint): 1) Make tunable. 2) This should depend on the depth instead of being constant.
  
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int width = depthmap_in_mm.width();
  const int height = depthmap_in_mm.height();

  if (x < width && y < height) {
    // Get depths and normalized image coordinates.
    const float2 nxy = make_float2(fx_inv * x + cx_inv, fy_inv * y + cy_inv);
    const float depth = 0.001f * depthmap_in_mm(y, x);
    // TODO: This might benefit from shared memory or texture use.
    const float depth_top = (y > 0) ? (0.001f * depthmap_in_mm(y - 1, x)) : depth;
    const float depth_left = (x > 0) ? (0.001f * depthmap_in_mm(y, x - 1)) : depth;
    const float depth_bottom = (y < height - 1) ? (0.001f * depthmap_in_mm(y + 1, x)) : depth;
    const float depth_right = (x < width - 1) ? (0.001f * depthmap_in_mm(y, x + 1)) : depth;
    
    // Write vertex position.
    const int vertex_index = 3 * (x + y * width);
    vertex_buffer[vertex_index + 0] = depth * nxy.x;
    vertex_buffer[vertex_index + 1] = depth * nxy.y;
    vertex_buffer[vertex_index + 2] = depth;
    
    // Write vertex color according to having a depth jump at the vertex.
    const bool foreground_boundary_vertex =
        depth_bottom - depth >= kJumpThreshold ||
        depth_right - depth >= kJumpThreshold ||
        depth_top - depth >= kJumpThreshold ||
        depth_left - depth >= kJumpThreshold;
    color_buffer[x + y * width] = foreground_boundary_vertex ? 255 : 0;
    
    // Write indices.
    const int index11 = (x + y * width);
    const int index10 = index11 - 1;
    const int index01 = index11 - width;
    if (x > 0 && y > 0) {
      const float depth_top_left = (0.001f * depthmap_in_mm(y - 1, x - 1));
      float depth_difference =
          max(max(fabs(depth_top_left - depth_left),
                  fabs(depth_top_left - depth_top)),
              max(fabs(depth_top - depth),
                  fabs(depth_left - depth)));
      const bool create_quad =
          depth_difference < kJumpThreshold &&
          depth_top_left > 0.f && depth_left > 0.f &&
          depth_top > 0.f && depth > 0.f;
      const int i = 2 + 4 * ((x - 1) + (y - 1) * width);
      if (create_quad) {
        // The last two vertices are actually unnecessary, but we need to write
        // a fixed number of vertices here, so we need to include them.
        index_buffer[i + 0] = index01;
        index_buffer[i + 1] = index11;
        index_buffer[i + 2] = index01; // Unnecessary.
        index_buffer[i + 3] = index11; // Unnecessary.
      } else {
        index_buffer[i + 0] = index10;
        index_buffer[i + 1] = index01;
        index_buffer[i + 2] = index01;
        index_buffer[i + 3] = index11;
      }
    } else if (x == 0) {
      // TODO(puzzlepaint): This could be written in an extra kernel for
      // perhaps slightly better performance.
      if (y == 0) {
        // Write start.
        index_buffer[0] = 0;
        index_buffer[1] = width;
      } else if (y < height - 1) {
        // Write newline.
        const int i = 2 + y * 4 * width - 4;
        index_buffer[i + 0] = index11 - 1 + width;
        index_buffer[i + 1] = index11;
        index_buffer[i + 2] = index11;
        index_buffer[i + 3] = index11 + width;
      }
    }
  }
}

void MeshDepthmapMMCUDA(
    const CUDABuffer_<uint16_t>& depthmap_in_mm,
    const float fx_inv,
    const float fy_inv,
    const float cx_inv,
    const float cy_inv,
    cudaStream_t stream,
    cudaGraphicsResource_t vertex_buffer,
    cudaGraphicsResource_t color_buffer,
    cudaGraphicsResource_t index_buffer) {
  // Map buffers.
  float* vertex_buffer_pointer;
  size_t vertex_buffer_size;
  CUDA_CHECKED_CALL(cudaGraphicsMapResources(1, &vertex_buffer, stream));
  CUDA_CHECKED_CALL(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&vertex_buffer_pointer), &vertex_buffer_size,
      vertex_buffer));
  
  uint8_t* color_buffer_pointer;
  size_t color_buffer_size;
  CUDA_CHECKED_CALL(cudaGraphicsMapResources(1, &color_buffer, stream));
  CUDA_CHECKED_CALL(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&color_buffer_pointer), &color_buffer_size,
      color_buffer));
  
  uint32_t* index_buffer_pointer;
  size_t index_buffer_size;
  CUDA_CHECKED_CALL(cudaGraphicsMapResources(1, &index_buffer, stream));
  CUDA_CHECKED_CALL(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&index_buffer_pointer), &index_buffer_size,
      index_buffer));
  
  // Run kernel.
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  const dim3 grid_dim(cuda_util::GetBlockCount(depthmap_in_mm.width(),
                                               kBlockWidth),
                      cuda_util::GetBlockCount(depthmap_in_mm.height(),
                                               kBlockHeight));
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  MeshDepthmapMMCUDAKernel<<<grid_dim, block_dim, 0, stream>>>(
      fx_inv, fy_inv, cx_inv, cy_inv, depthmap_in_mm, vertex_buffer_pointer,
      color_buffer_pointer, index_buffer_pointer);
  CHECK_CUDA_NO_ERROR();
  
  // Unmap buffers.
  CUDA_CHECKED_CALL(cudaGraphicsUnmapResources(1, &index_buffer, stream));
  CUDA_CHECKED_CALL(cudaGraphicsUnmapResources(1, &color_buffer, stream));
  CUDA_CHECKED_CALL(cudaGraphicsUnmapResources(1, &vertex_buffer, stream));
}

__global__ void ProjectImageOntoDepthMapCUDAKernel(
    cudaTextureObject_t depth_texture,
    CUDABuffer_<uint8_t> y_image,
    CUDABuffer_<uint16_t> uv_image,
    float yuv_fx,
    float yuv_fy,
    float yuv_cx,
    float yuv_cy,
    float yuv_k1,
    float yuv_k2,
    float yuv_k3,
    float depth_fx_inv,
    float depth_fy_inv,
    float depth_cx_center_inv,
    float depth_cy_center_inv,
    CUDAMatrix3x4 depth_frame_to_yuv_frame,
    CUDABuffer_<uchar4> output) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int width = output.width();
  const int height = output.height();
  if (x < width && y < height) {
    // Unproject pixel into depth space.
    const float depth_z = tex2D<float>(depth_texture, x, y);
    const float depth_x = depth_z * (depth_fx_inv * x + depth_cx_center_inv);
    const float depth_y = depth_z * (depth_fy_inv * y + depth_cy_center_inv);
    
    // Transform into yuv camera space.
    const CUDAMatrix3x4& T = depth_frame_to_yuv_frame;
    const float yuv_x = T.row0.x * depth_x + T.row0.y * depth_y +
                        T.row0.z * depth_z + T.row0.w;
    const float yuv_y = T.row1.x * depth_x + T.row1.y * depth_y +
                        T.row1.z * depth_z + T.row1.w;
    const float yuv_z = T.row2.x * depth_x + T.row2.y * depth_y +
                        T.row2.z * depth_z + T.row2.w;
    
    // Project onto yuv image.
    // Note: ignoring the case of points behind the camera.
    const float yuv_d_nx = yuv_x / yuv_z;
    const float yuv_d_ny = yuv_y / yuv_z;
    const float r2 = yuv_d_nx * yuv_d_nx + yuv_d_ny * yuv_d_ny;
    const float factor = 1.0f + r2 * (yuv_k1 + r2 * (yuv_k2 + r2 * yuv_k3));
    const float yuv_nx = factor * yuv_d_nx;
    const float yuv_ny = factor * yuv_d_ny;
    
    // Look up color at projected position.
    // Note: not applying interpolation.
    const int px = yuv_fx * yuv_nx + yuv_cx;
    const int py = yuv_fy * yuv_ny + yuv_cy;
    uchar4 color_output;
    // NOTE: Special case for Tango tablet images. Since these images contain
    //       some metadata in the first few rows, we must skip these rows here
    //       or the inpainting-extrapolation will be terrible.
    const int kTopRowSkip = 0; //4;
    const int kBottomRowSkip = 0; //2;
    if (depth_z > 0.f && px >= 0 && py >= kTopRowSkip && px < y_image.width() && py < y_image.height() - kBottomRowSkip) {
      const uint8_t color_y = y_image(py, px);
      const uint16_t color_uv = uv_image(py / 2, px / 2);
      const uint8_t color_u = color_uv >> 8;
      const uint8_t color_v = color_uv & 0x00FF;
      
      // Convert YUV color to RGB.
      const int color_r = color_y + 1.4075f * (color_v - 128);
      const int color_g = color_y - 0.3455f * (color_u - 128) - (0.7169f * (color_v - 128));
      const int color_b = color_y + 1.7790f * (color_u - 128);

      color_output = make_uchar4(
          max(min(color_r, 255), 0),
          max(min(color_g, 255), 0),
          max(min(color_b, 255), 0), 255);
    } else {
      color_output = make_uchar4(0, 0, 0, 0);
    }
    output(y, x) = color_output;
  }
}

void ProjectImageOntoDepthMapCUDA(
    cudaStream_t stream,
    cudaTextureObject_t depth_texture,
    const CUDABuffer<uint8_t>& y_image,
    const CUDABuffer<uint16_t>& uv_image,
    float yuv_fx,
    float yuv_fy,
    float yuv_cx,
    float yuv_cy,
    float yuv_k1,
    float yuv_k2,
    float yuv_k3,
    float depth_fx_inv,
    float depth_fy_inv,
    float depth_cx_center_inv,
    float depth_cy_center_inv,
    const CUDAMatrix3x4& depth_frame_to_yuv_frame,
    CUDABuffer<uchar4>* output) {
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  const dim3 grid_dim(cuda_util::GetBlockCount(output->width(),
                                               kBlockWidth),
                      cuda_util::GetBlockCount(output->height(),
                                               kBlockHeight));
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  ProjectImageOntoDepthMapCUDAKernel<<<grid_dim, block_dim, 0, stream>>>(
      depth_texture, y_image.ToCUDA(), uv_image.ToCUDA(), yuv_fx, yuv_fy,
      yuv_cx, yuv_cy, yuv_k1, yuv_k2, yuv_k3, depth_fx_inv, depth_fy_inv,
      depth_cx_center_inv, depth_cy_center_inv, depth_frame_to_yuv_frame,
      output->ToCUDA());
  CHECK_CUDA_NO_ERROR();
}

__global__ void CopyFloatColorImageToRgb8SurfaceCUDAKernel(
    CUDABuffer_<float> color_image_r,
    CUDABuffer_<float> color_image_g,
    CUDABuffer_<float> color_image_b,
    cudaSurfaceObject_t surface) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int width = color_image_r.width();
  const int height = color_image_r.height();
  if (x < width && y < height) {
    const float r = color_image_r(y, x);
    const float g = color_image_g(y, x);
    const float b = color_image_b(y, x);
    uchar4 data = make_uchar4(255.99f * r,
                              255.99f * g,
                              255.99f * b,
                              255);
    surf2Dwrite(data, surface, x * sizeof(uchar4), y, cudaBoundaryModeTrap);
  }
}

void CopyFloatColorImageToRgb8SurfaceCUDA(
    cudaStream_t stream,
    const CUDABuffer<float>& color_image_r,
    const CUDABuffer<float>& color_image_g,
    const CUDABuffer<float>& color_image_b,
    cudaSurfaceObject_t surface) {
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  const dim3 grid_dim(cuda_util::GetBlockCount(color_image_r.width(),
                                               kBlockWidth),
                      cuda_util::GetBlockCount(color_image_r.height(),
                                               kBlockHeight));
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  CopyFloatColorImageToRgb8SurfaceCUDAKernel<<<grid_dim, block_dim, 0, stream>>>(
      color_image_r.ToCUDA(),
      color_image_g.ToCUDA(),
      color_image_b.ToCUDA(),
      surface);
  CHECK_CUDA_NO_ERROR();
}

__global__ void CopyColorImageToRgb8SurfaceCUDAKernel(
    CUDABuffer_<uchar4> color_rgbx,
    cudaSurfaceObject_t surface) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int width = color_rgbx.width();
  const int height = color_rgbx.height();
  if (x < width && y < height) {
    const uchar4 rgbx = color_rgbx(y, x);
    uchar4 data = make_uchar4(rgbx.x, rgbx.y, rgbx.z, 255);
    surf2Dwrite(data, surface, x * sizeof(uchar4), y, cudaBoundaryModeTrap);
  }
}

void CopyColorImageToRgb8SurfaceCUDA(
    cudaStream_t stream,
    const CUDABuffer<uchar4>& color_rgbx,
    cudaSurfaceObject_t surface) {
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  const dim3 grid_dim(cuda_util::GetBlockCount(color_rgbx.width(),
                                               kBlockWidth),
                      cuda_util::GetBlockCount(color_rgbx.height(),
                                               kBlockHeight));
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  CopyColorImageToRgb8SurfaceCUDAKernel<<<grid_dim, block_dim, 0, stream>>>(
      color_rgbx.ToCUDA(),
      surface);
  CHECK_CUDA_NO_ERROR();
}

__global__ void CopyValidToInvalidPixelsCUDAKernel(
    cudaTextureObject_t src_depth_texture,
    cudaTextureObject_t src_color_texture,
    CUDABuffer_<float> dest_depth,
    CUDABuffer_<uchar4> dest_color) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int width = dest_depth.width();
  const int height = dest_depth.height();
  if (x < width && y < height) {
    const float dest_depth_value = dest_depth(y, x);
    if (dest_depth_value <= 0.f) {
      const float src_depth_value = tex2D<float>(src_depth_texture, x, y);
      if (src_depth_value > 0.f) {
        dest_depth(y, x) = src_depth_value;
        dest_color(y, x) = tex2D<uchar4>(src_color_texture, x, y);
      }
    }
  }
}

void CopyValidToInvalidPixelsCUDA(
    cudaStream_t stream,
    cudaTextureObject_t src_depth_texture,
    cudaTextureObject_t src_color_texture,
    CUDABuffer<float>* dest_depth,
    CUDABuffer<uchar4>* dest_color) {
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  const dim3 grid_dim(cuda_util::GetBlockCount(dest_depth->width(),
                                               kBlockWidth),
                      cuda_util::GetBlockCount(dest_depth->height(),
                                               kBlockHeight));
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  CopyValidToInvalidPixelsCUDAKernel<<<grid_dim, block_dim, 0, stream>>>(
      src_depth_texture, src_color_texture, dest_depth->ToCUDA(), dest_color->ToCUDA());
  CHECK_CUDA_NO_ERROR();
}

__global__ void DeleteAlmostOccludedPixelsCUDAKernel(
    int radius,
    float occlusion_threshold,
    CUDABuffer_<float> src_depth_map,
    CUDABuffer_<float> dest_depth_map) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int width = dest_depth_map.width();
  const int height = dest_depth_map.height();
  if (x < width && y < height) {
    // Only testing a few selected points here. This should be okay given the
    // assumption that the depth maps are reasonably smooth (which they should
    // be).
    bool almost_occluded = false;
    float depth_threshold = src_depth_map(y, x) - occlusion_threshold;
    
    almost_occluded |= (x > radius && src_depth_map(y, x - radius) < depth_threshold);
    almost_occluded |= (y > radius && src_depth_map(y - radius, x) < depth_threshold);
    almost_occluded |= (x < width - radius && src_depth_map(y, x + radius) < depth_threshold);
    almost_occluded |= (y < height - radius && src_depth_map(y + radius, x) < depth_threshold);
    
    dest_depth_map(y, x) = almost_occluded ? 0 : (depth_threshold + occlusion_threshold);
  }
}

void DeleteAlmostOccludedPixelsCUDA(
    cudaStream_t stream,
    int radius,
    float occlusion_threshold,
    const CUDABuffer<float>& src_depth_map,
    CUDABuffer<float>* dest_depth_map) {
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  const dim3 grid_dim(cuda_util::GetBlockCount(dest_depth_map->width(),
                                               kBlockWidth),
                      cuda_util::GetBlockCount(dest_depth_map->height(),
                                               kBlockHeight));
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  DeleteAlmostOccludedPixelsCUDAKernel<<<grid_dim, block_dim, 0, stream>>>(
      radius, occlusion_threshold, src_depth_map.ToCUDA(), dest_depth_map->ToCUDA());
  CHECK_CUDA_NO_ERROR();
}

__forceinline__ __device__ float4 uchar4ToFloat4(const uchar4& input) {
  return make_float4(input.x, input.y, input.z, input.w);
}

__forceinline__ __device__ uchar4 float4ToUchar4(const float4& input) {
  return make_uchar4(input.x, input.y, input.z, input.w);
}

__global__ void ForwardReprojectToInvalidPixelsCUDAKernel(
    CUDAMatrix3x4 dest_T_src,
    float src_fx_inv,
    float src_fy_inv,
    float src_cx_inv,
    float src_cy_inv,
    CUDABuffer_<float> src_depths,
    CUDABuffer_<uchar4> src_colors,
    float dest_fx,
    float dest_fy,
    float dest_cx,
    float dest_cy,
    CUDABuffer_<float> dest_depths,
    CUDABuffer_<uchar4> dest_colors) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < src_depths.width() && y < src_depths.height() &&
      x % 2 == 0 && y % 2 == 0) {
    // Unproject.
    const float src_z = src_depths(y, x);
    if (src_z > 0) {
      const float src_x = src_z * (src_fx_inv * x + src_cx_inv);
      const float src_y = src_z * (src_fy_inv * y + src_cy_inv);
      
      // Transform.
      const CUDAMatrix3x4& T = dest_T_src;
      const float dest_x = T.row0.x * src_x + T.row0.y * src_y +
                           T.row0.z * src_z + T.row0.w;
      const float dest_y = T.row1.x * src_x + T.row1.y * src_y +
                           T.row1.z * src_z + T.row1.w;
      const float dest_z = T.row2.x * src_x + T.row2.y * src_y +
                           T.row2.z * src_z + T.row2.w;
      
      // Project.
      if (dest_z > 0) {
        const float dest_px = dest_fx * (dest_x / dest_z) + dest_cx;
        const float dest_py = dest_fy * (dest_y / dest_z) + dest_cy;
        const int ix = dest_px + 0.5f;
        const int iy = dest_py + 0.5f;
        
        if (dest_px >= 0 && dest_py >= 0 &&
            ix >= 0 && iy >= 0 &&
            ix < src_colors.width() && iy < src_colors.height() &&
            dest_depths(iy, ix) <= 0) {
          // NOTE: In principle, this should use locking to prevent simultaneous
          //       access.
          dest_depths(iy, ix) = dest_z;
          
          // Direct assignment:
          // dest_colors(iy, ix) = src_colors(y, x);
          
          // Blur:
          // (Note: Kernel without the center pixel seems to work just as well.)
          // Note: Proper rounding (addition of 0.5f before conversion to uchar)
          //       is important, otherwise the colors slowly go to black.
          dest_colors(iy, ix) = float4ToUchar4(0.2f * (
              uchar4ToFloat4(src_colors(y, x)) +
              uchar4ToFloat4(src_colors(y, max(0, x - 1))) +
              uchar4ToFloat4(src_colors(max(0, y - 1), x)) +
              uchar4ToFloat4(src_colors(y, min(src_colors.width() - 1, x + 1))) +
              uchar4ToFloat4(src_colors(min(src_colors.height() - 1, y + 1), x))) +
              make_float4(0.5f, 0.5f, 0.5f, 0.5f));
        }
      }
    }
  }
}

void ForwardReprojectToInvalidPixelsCUDA(
    cudaStream_t stream,
    const CUDAMatrix3x4& dest_T_src,
    float src_fx_inv,
    float src_fy_inv,
    float src_cx_inv,
    float src_cy_inv,
    const CUDABuffer<float>& src_depths,
    const CUDABuffer<uchar4>& src_colors,
    float dest_fx,
    float dest_fy,
    float dest_cx,
    float dest_cy,
    CUDABuffer<float>* dest_depths,
    CUDABuffer<uchar4>* dest_colors) {
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  const dim3 grid_dim(cuda_util::GetBlockCount(src_depths.width(),
                                               kBlockWidth),
                      cuda_util::GetBlockCount(src_depths.height(),
                                               kBlockHeight));
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  ForwardReprojectToInvalidPixelsCUDAKernel<<<grid_dim, block_dim, 0, stream>>>(
      dest_T_src,
      src_fx_inv,
      src_fy_inv,
      src_cx_inv,
      src_cy_inv,
      src_depths.ToCUDA(),
      src_colors.ToCUDA(),
      dest_fx,
      dest_fy,
      dest_cx,
      dest_cy,
      dest_depths->ToCUDA(),
      dest_colors->ToCUDA());
  CHECK_CUDA_NO_ERROR();
}

__global__ void ForwardReprojectToInvalidPixelsCUDAKernel(
    CUDAMatrix3x4 dest_T_src,
    float src_fx_inv,
    float src_fy_inv,
    float src_cx_inv,
    float src_cy_inv,
    CUDABuffer_<float> src_depths,
    CUDABuffer_<float> src_r,
    CUDABuffer_<float> src_g,
    CUDABuffer_<float> src_b,
    float dest_fx,
    float dest_fy,
    float dest_cx,
    float dest_cy,
    CUDABuffer_<float> dest_depths,
    CUDABuffer_<uchar4> dest_colors) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < src_depths.width() && y < src_depths.height() &&
      x % 2 == 0 && y % 2 == 0) {
    // Unproject.
    const float src_z = src_depths(y, x);
    if (src_z > 0) {
      const float src_x = src_z * (src_fx_inv * x + src_cx_inv);
      const float src_y = src_z * (src_fy_inv * y + src_cy_inv);
      
      // Transform.
      const CUDAMatrix3x4& T = dest_T_src;
      const float dest_x = T.row0.x * src_x + T.row0.y * src_y +
                           T.row0.z * src_z + T.row0.w;
      const float dest_y = T.row1.x * src_x + T.row1.y * src_y +
                           T.row1.z * src_z + T.row1.w;
      const float dest_z = T.row2.x * src_x + T.row2.y * src_y +
                           T.row2.z * src_z + T.row2.w;
      
      // Project.
      if (dest_z > 0) {
        const float dest_px = dest_fx * (dest_x / dest_z) + dest_cx;
        const float dest_py = dest_fy * (dest_y / dest_z) + dest_cy;
        const int ix = dest_px + 0.5f;
        const int iy = dest_py + 0.5f;
        
        if (dest_px >= 0 && dest_py >= 0 &&
            ix >= 0 && iy >= 0 &&
            ix < src_r.width() && iy < src_r.height() &&
            dest_depths(iy, ix) <= 0) {
          // NOTE: In principle, this should use locking to prevent simultaneous
          //       access.
          dest_depths(iy, ix) = dest_z;

          // (Note: Kernel without the center pixel seems to work just as well.)
          // Note: Proper rounding (addition of 0.5f before conversion to uchar)
          //       is important, otherwise the colors slowly go to black.
          dest_colors(iy, ix) = make_uchar4(
            0.5f + 255.99f * 0.2f * (src_r(y, x) + src_r(y, max(0, x - 1)) + src_r(max(0, y - 1), x) + src_r(y, min(src_r.width() - 1, x + 1)) + src_r(min(src_r.height() - 1, y + 1), x)),
            0.5f + 255.99f * 0.2f * (src_g(y, x) + src_g(y, max(0, x - 1)) + src_g(max(0, y - 1), x) + src_g(y, min(src_r.width() - 1, x + 1)) + src_g(min(src_r.height() - 1, y + 1), x)),
            0.5f + 255.99f * 0.2f * (src_b(y, x) + src_b(y, max(0, x - 1)) + src_b(max(0, y - 1), x) + src_b(y, min(src_r.width() - 1, x + 1)) + src_b(min(src_r.height() - 1, y + 1), x)),
            255);
        }
      }
    }
  }
}

void ForwardReprojectToInvalidPixelsCUDA(
    cudaStream_t stream,
    const CUDAMatrix3x4& dest_T_src,
    float src_fx_inv,
    float src_fy_inv,
    float src_cx_inv,
    float src_cy_inv,
    const CUDABuffer<float>& src_depths,
    const CUDABuffer<float>& src_r,
    const CUDABuffer<float>& src_g,
    const CUDABuffer<float>& src_b,
    float dest_fx,
    float dest_fy,
    float dest_cx,
    float dest_cy,
    CUDABuffer<float>* dest_depths,
    CUDABuffer<uchar4>* dest_colors) {
  constexpr int kBlockWidth = 32;
  constexpr int kBlockHeight = 32;
  const dim3 grid_dim(cuda_util::GetBlockCount(src_depths.width(),
                                               kBlockWidth),
                      cuda_util::GetBlockCount(src_depths.height(),
                                               kBlockHeight));
  const dim3 block_dim(kBlockWidth, kBlockHeight);
  ForwardReprojectToInvalidPixelsCUDAKernel<<<grid_dim, block_dim, 0, stream>>>(
      dest_T_src,
      src_fx_inv,
      src_fy_inv,
      src_cx_inv,
      src_cy_inv,
      src_depths.ToCUDA(),
      src_r.ToCUDA(),
      src_g.ToCUDA(),
      src_b.ToCUDA(),
      dest_fx,
      dest_fy,
      dest_cx,
      dest_cy,
      dest_depths->ToCUDA(),
      dest_colors->ToCUDA());
  CHECK_CUDA_NO_ERROR();
}
}  // namespace view_correction
