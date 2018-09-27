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

#ifndef VIEW_CORRECTION_VIEW_CORRECTION_DISPLAY_H_
#define VIEW_CORRECTION_VIEW_CORRECTION_DISPLAY_H_

#include <chrono>
#include <cstdarg>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef ANDROID
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#else
#include <GL/glew.h>
#include <GL/gl.h>
#endif

#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

namespace view_correction {

struct ViewCorrectionDisplayImpl;


// To do: if using meshes as input, replace this with your mesh class
struct MeshStub {
  inline MeshStub()
      : empty_(true) {}
  
  void* vertex_position_data;
  void* vertex_color_data;
  void* face_data;
  int face_count;
  
  bool empty_;
  
  inline bool empty() {
    return empty_;
  }
};


struct Intrinsics {
  // Focal length
  float fx;
  float fy;
  
  // Principal point
  float cx;
  float cy;
  
  // Image size
  int width;
  int height;
};


struct TimestampedSE3f {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  Sophus::SE3f transform;
  uint64_t timestamp;
};


class DepthImage : public cv::Mat_<uint16_t> {
public:
  inline DepthImage() {}
  
  inline DepthImage(const DepthImage& other)
      : cv::Mat_<uint16_t>(other),
        timestamp_ns_(other.timestamp_ns_) {}
  
  inline void set_timestamp_ns(uint64_t timestamp_ns) {
    timestamp_ns_ = timestamp_ns;
  }
  
  inline uint64_t timestamp_ns() const {
    return timestamp_ns_;
  }
  
private:
  uint64_t timestamp_ns_;
};


class ColorImage : public cv::Mat_<cv::Vec3b> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  inline ColorImage() {}
  
  inline ColorImage(const ColorImage& other)
      : cv::Mat_<cv::Vec3b>(other),
        timestamp_ns_(other.timestamp_ns_) {}
  
  
  inline void set_timestamp_ns(uint64_t timestamp_ns) {
    timestamp_ns_ = timestamp_ns;
  }
  
  inline uint64_t timestamp_ns() const {
    return timestamp_ns_;
  }
  
  
  inline void set_G_T_C(const Sophus::SE3f& G_T_C) {
    G_T_C_ = G_T_C;
  }
  
  inline Sophus::SE3f G_T_C() const {
    return G_T_C_;
  }
  
private:
  uint64_t timestamp_ns_;
  Sophus::SE3f G_T_C_;  // camera-to-global transformation
};


class ViewCorrectionDisplay {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  enum class TargetViewMode {
    kFixedOffset = 0,
    kReceiveFromUDP
  };
  
  ViewCorrectionDisplay(
      int width, int height, int offset_x, int offset_y,
      TargetViewMode target_view_mode,
      const Intrinsics& depth_intrinsics,
      const Intrinsics& yuv_intrinsics);
  
  virtual ~ViewCorrectionDisplay();
  
  void Init();
  
  bool Render();
  
  // Must be called to update the class with new input YUV images.
  void YUVImageCallback(const ColorImage& image);
  
  // Must be called to update the class with new input depth images (if using depth map input; otherwise, use MeshCallback()).
  // Depth images are expected in millimeters in uint16_t format.
  void DepthImageCallback(const DepthImage& image);
  
  // Must be called to update the class with new scene reconstructions (if using mesh input; otherwise, use DepthImageCallback()).
  void MeshCallback(const std::shared_ptr<MeshStub>& mesh);

  
  // For demonstration purposes only, should be replaced with your own timestamp handling.
  inline void SetStartTime(std::chrono::steady_clock::time_point start_time) {
    start_time_ = start_time;
  }
  
  // For demonstration purposes only, should be replaced by pose polling in GetCurrentColorCameraPose().
  void ColorCameraPoseCallback(const Sophus::SE3f& G_T_C, uint64_t timestamp);
  
 private:
  uint64_t GetCurrentTimestamp() {
    LOG(ERROR) << "This function is a stub. You have to replace it with your function to get the current timestamp.";
    
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    uint64_t nanoseconds = std::chrono::duration<double, std::nano>(now - start_time_).count();
    
    return nanoseconds;
  }
  
  bool GetCurrentColorCameraPose(Sophus::SE3f* result, uint64_t* timestamp_of_result) {
    LOG(ERROR) << "This function is a stub. You have to replace it with your function to get the current color camera pose.";
    
    *result = input_G_T_C_;
    *timestamp_of_result = input_G_T_C_timestamp_;
    
    return true;  // return false to signal failure
  }
  
  bool GetYUVImagePose(const ColorImage& image, Sophus::SE3f* G_T_C) {
    LOG(ERROR) << "This function is a stub. You have to replace it with your function to get the YUV image's pose.";
    
    *G_T_C = image.G_T_C();
    
    return true;  // return false to signal failure
  }
  
  void InitDisplay();
  
  void DisplayOnScreen();
  
//   void InitAR();
  
  // Runs the view correction pipeline.
  // update_data: If true, runs the pipeline for the latest input data.
  //              If false, re-uses the last input (useful for evaluation).
  // inpaint_in_rgb_frame: If true, inpaints depth in the rgb frame first,
  //                       using gradient based weights. If false, skips this
  //                       step.
  bool RunPipeline(bool update_data, bool inpaint_in_rgb_frame, bool render_stereo_image);
  
  void RenderARContent(uint64_t timestamp,
                       const Sophus::SE3f& target_T_G,
                       float fx, float fy, float cx, float cy, float min_depth,
                       float max_depth);
  
  // Renders a depth map from a mesh.
  void RenderDepthImageFromMesh(const MeshStub& mesh,
                                const Sophus::SE3f& G_T_C);
  
  // Computes a meshed inpainted depth map from the input depth image and yuv image.
  void CreateMeshedInpaintedDepthMap(
      const ColorImage& rgb_image,
      uint32_t* num_src_depth_pixels_to_inpaint);
  
  // Sets up the virtual camera view relative to the source frame.
  bool SetupTargetView(
      const Sophus::SE3f& G_T_latest_C,
      bool render_stereo_image,
      Sophus::SE3f* source_frame_to_view_frame,
      float* target_fx,
      float* target_fy,
      float* target_cx,
      float* target_cy);
  

  
  // Settings.
  int target_render_width_;
  int target_render_height_;
  TargetViewMode target_view_mode_;
  
  // New input.
  std::mutex input_mutex_;
  Sophus::SE3f input_G_T_C_;
  uint64_t input_G_T_C_timestamp_;
  DepthImage input_depth_image_;
  std::shared_ptr<MeshStub> input_mesh_;
  std::vector<ColorImage> input_yuv_images_;
  uint64_t latest_depth_image_timestamp_;
  
  // Last pose used for view correction.
  Sophus::SE3f G_T_latest_C;
  uint64_t G_T_latest_C_timestamp;
  // Last mesh used for view correction.
  std::shared_ptr<MeshStub> mesh_to_render;
  
  // Intrinsics.
  const Intrinsics& depth_intrinsics_;
  const float depth_fx_;
  const float depth_fy_;
  const float depth_cx_;  // uses origin-at-pixel-center convention.
  const float depth_cy_;  // uses origin-at-pixel-center convention.
  const float depth_fx_inv_;
  const float depth_fy_inv_;
  const float depth_cx_inv_;  // uses origin-at-pixel-center convention.
  const float depth_cy_inv_;  // uses origin-at-pixel-center convention.
  const Intrinsics& yuv_intrinsics_;
  
  // Data of the last processed frame (for evaluation).
  ColorImage last_yuv_image_;
  Sophus::SE3f last_yuv_image_G_T_C_;
  
  // Display size
  int offset_x_;
  int offset_y_;
  int width_;
  int height_;
  
  // For getting the current timestamp.
  std::chrono::steady_clock::time_point start_time_;
  
  // Hidden implementation details (to not propagate CUDA includes).
  std::unique_ptr<ViewCorrectionDisplayImpl> d_;
};

typedef std::shared_ptr<ViewCorrectionDisplay> ViewCorrectionDisplayPtr;
typedef std::shared_ptr<const ViewCorrectionDisplay> ViewCorrectionDisplayConstPtr;

}  // namespace view_correction

#endif  // VIEW_CORRECTION_VIEW_CORRECTION_DISPLAY_H_
