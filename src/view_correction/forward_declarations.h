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

#ifndef VIEW_CORRECTION_FORWARD_DECLARATIONS_H_
#define VIEW_CORRECTION_FORWARD_DECLARATIONS_H_

#include <memory>

// Forward declares important classes (and declares pointer types).
namespace view_correction {

class CameraBase;
typedef std::shared_ptr<CameraBase> CameraBasePtr;
typedef std::shared_ptr<const CameraBase> CameraBaseConstPtr;

template <typename T>
class CUDABuffer;
template <typename T>
using CUDABufferPtr = std::shared_ptr<CUDABuffer<T>>;
template <typename T>
using CUDABufferConstPtr = std::shared_ptr<const CUDABuffer<T>>;

class Frame;
typedef std::shared_ptr<Frame> FramePtr;
typedef std::shared_ptr<const Frame> FrameConstPtr;

class FrameBufferManager;

class FrameDeviceBuffers;
typedef std::shared_ptr<FrameDeviceBuffers> FrameDeviceBuffersPtr;
typedef std::shared_ptr<const FrameDeviceBuffers> FrameDeviceBuffersConstPtr;

struct SceneEstimate;

class ViewCorrection;
typedef std::shared_ptr<ViewCorrection> ViewCorrectionPtr;
typedef std::unique_ptr<ViewCorrection> ViewCorrectionUniquePtr;
typedef std::shared_ptr<const ViewCorrection> ViewCorrectionConstPtr;

}  // namespace view_correction

#endif  // VIEW_CORRECTION_FORWARD_DECLARATIONS_H_
