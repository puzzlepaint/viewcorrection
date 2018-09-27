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

#ifndef VIEW_CORRECTION_VIEW_CORRECTION_DISPLAY_CUH_
#define VIEW_CORRECTION_VIEW_CORRECTION_DISPLAY_CUH_

#include <cuda_runtime.h>

#include "view_correction/cuda_buffer.h"

namespace view_correction {

void DownsampleImageToHalfSizeCUDA(
    cudaStream_t stream,
    const CUDABuffer<uint8_t>& source,
    CUDABuffer<uint8_t>* target);

void ComputeGradientMagnitudeDiv2CUDA(
    cudaStream_t stream,
    const CUDABuffer<uint8_t>& image,
    CUDABuffer<uint8_t>* output);

// The vertex buffer must have space for at least
//     depthmap.width() * depthmap.height()
// entires, the index buffer must have space for at least
//     2 + 4 * (depthmap.height() - 2) +
//     4 * (depthmap.width() - 1) * (depthmap.height() - 1)
// entries.
void MeshDepthmapCUDA(
    const CUDABuffer_<float>& depthmap,
    const float fx_inv,
    const float fy_inv,
    const float cx_inv,
    const float cy_inv,
    cudaStream_t stream,
    cudaGraphicsResource_t vertex_buffer,
    cudaGraphicsResource_t color_buffer,
    cudaGraphicsResource_t index_buffer);

void MeshDepthmapMMCUDA(
    const CUDABuffer_<uint16_t>& depthmap_in_mm,
    const float fx_inv,
    const float fy_inv,
    const float cx_inv,
    const float cy_inv,
    cudaStream_t stream,
    cudaGraphicsResource_t vertex_buffer,
    cudaGraphicsResource_t color_buffer,
    cudaGraphicsResource_t index_buffer);

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
    CUDABuffer<uchar4>* output);

void CopyFloatColorImageToRgb8SurfaceCUDA(
    cudaStream_t stream,
    const CUDABuffer<float>& color_r,
    const CUDABuffer<float>& color_g,
    const CUDABuffer<float>& color_b,
    cudaSurfaceObject_t surface);

void CopyColorImageToRgb8SurfaceCUDA(
    cudaStream_t stream,
    const CUDABuffer<uchar4>& color_rgbx,
    cudaSurfaceObject_t surface);

void CopyValidToInvalidPixelsCUDA(
    cudaStream_t stream,
    cudaTextureObject_t src_depth_texture,
    cudaTextureObject_t src_color_texture,
    CUDABuffer<float>* dest_depth,
    CUDABuffer<uchar4>* dest_color);

void DeleteAlmostOccludedPixelsCUDA(
    cudaStream_t stream,
    int radius,
    float occlusion_threshold,
    const CUDABuffer<float>& src_depth_map,
    CUDABuffer<float>* dest_depth_map);

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
    CUDABuffer<uchar4>* dest_colors);

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
    CUDABuffer<uchar4>* dest_colors);
}  // namespace view_correction

#endif  // VIEW_CORRECTION_VIEW_CORRECTION_DISPLAY_CUH_
