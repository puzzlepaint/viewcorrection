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

#ifndef VIEW_CORRECTION_CUDA_TV_INPAINTING_FUNCTIONS_CUH_
#define VIEW_CORRECTION_CUDA_TV_INPAINTING_FUNCTIONS_CUH_

#include <cuda_runtime.h>

#include "view_correction/cuda_buffer.h"

namespace view_correction {

enum InpaintingMode {
  kIMAuto =  0,
  kIMClassic = 1,    
  kIMAdaptive = 2,
  kIMCoarseToFine = 3,
  kIMCoarseToFineAdaptive = 4,
};

// Returns the number of iterations done.
int InpaintDepthMapCUDA(
    cudaStream_t stream,    
    InpaintingMode inpainting_mode,
    bool use_tv_weights,
    int max_num_iterations,
    float max_change_rate_threshold,
    float depth_input_scaling_factor,
    cudaTextureObject_t gradient_magnitude_div_sqrt2,
    cudaTextureObject_t depth_map_input,
    CUDABuffer<bool>* tv_flag,
    CUDABuffer<bool>* tv_dual_flag,
    CUDABuffer<int16_t>* tv_dual_x,
    CUDABuffer<int16_t>* tv_dual_y,
    CUDABuffer<float>* tv_u_bar,
    CUDABuffer<float>* tv_max_change,
    CUDABuffer<float>* depth_map_output,
    CUDABuffer<uint16_t>* block_coordinates,
    CUDABuffer<unsigned char>* block_activities);

// Returns the number of iterations done.
int InpaintImageCUDA(
    cudaStream_t stream,
    int max_num_iterations,
    float max_change_rate_threshold,
    cudaTextureObject_t gradient_magnitude_div_sqrt2,
    const CUDABuffer<uchar4>& input,
    CUDABuffer<bool>* tv_flag,
    CUDABuffer<bool>* tv_dual_flag,
    CUDABuffer<float>* tv_dual_x_r,
    CUDABuffer<float>* tv_dual_x_g,
    CUDABuffer<float>* tv_dual_x_b,
    CUDABuffer<float>* tv_dual_y_r,
    CUDABuffer<float>* tv_dual_y_g,
    CUDABuffer<float>* tv_dual_y_b,
    CUDABuffer<float>* tv_u_bar_r,
    CUDABuffer<float>* tv_u_bar_g,
    CUDABuffer<float>* tv_u_bar_b,
    CUDABuffer<float>* tv_max_change,
    CUDABuffer<float>* output_r,
    CUDABuffer<float>* output_g,
    CUDABuffer<float>* output_b,
    CUDABuffer<uint16_t>* block_coordinates);

} // namespace view_correction

#endif // #ifndef VIEW_CORRECTION_CUDA_TV_INPAINTING_FUNCTIONS_CUH_
