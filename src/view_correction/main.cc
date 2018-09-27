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

#include <chrono>

#include <gflags/gflags.h>

#include "view_correction/flags.h"
#include "view_correction/opengl_util.h"
#include "view_correction/view_correction_display.h"

#include <GLFW/glfw3.h>

using namespace view_correction;
using namespace std::chrono;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  // Set target display settings (width and height can be chosen freely).
  int width = 800;
  int height = 600;
  int offset_x = 0;
  int offset_y = 0;
  
  // Set how to retrieve the viewer position. Possible values:
  // kFixedOffset uses a fixed offset from the color camera pose.
  // kReceiveFromUDP receives the viewer position from UDP messages over the network.
  ViewCorrectionDisplay::TargetViewMode view_mode =
      ViewCorrectionDisplay::TargetViewMode::kFixedOffset;
  
  // For testing, create synthetic images. This should be replaced with loading the actual data.
  // NOTE: We are using the same resolution for the color and depth images here, but they could differ.
  constexpr int kSyntheticImageWidth = 400;
  constexpr int kSyntheticImageHeight = 300;
  
  // Load depth intrinsics.
  Intrinsics depth_intrinsics;
  LOG(ERROR) << "Stub: load depth camera intrinsics here.";
  
  // Make up some intrinsics for demonstration purposes.
  depth_intrinsics.fx = kSyntheticImageHeight / 2;
  depth_intrinsics.fy = kSyntheticImageHeight / 2;
  depth_intrinsics.cx = kSyntheticImageWidth / 2;
  depth_intrinsics.cy = kSyntheticImageHeight / 2;
  depth_intrinsics.width = kSyntheticImageWidth;
  depth_intrinsics.height = kSyntheticImageHeight;
  
  // Load color intrinsics.
  Intrinsics color_intrinsics;
  LOG(ERROR) << "Stub: load color camera intrinsics here.";
  
  // Make up some intrinsics for demonstration purposes.
  color_intrinsics.fx = kSyntheticImageHeight / 2;
  color_intrinsics.fy = kSyntheticImageHeight / 2;
  color_intrinsics.cx = kSyntheticImageWidth / 2;
  color_intrinsics.cy = kSyntheticImageHeight / 2;
  color_intrinsics.width = kSyntheticImageWidth;
  color_intrinsics.height = kSyntheticImageHeight;
  
  // Create ViewCorrectionDisplay
  std::shared_ptr<ViewCorrectionDisplay> display(new ViewCorrectionDisplay(
      width, height, offset_x, offset_y,
      view_mode,
      depth_intrinsics,
      color_intrinsics));
  
  // Create OpenGL context and window and make the context current.
  // We use the GLFW library here for convenience.
  if (!glfwInit()) {
    LOG(ERROR) << "Cannot initialize GLFW. Aborting.";
    return 1;
  }
  GLFWwindow* window = glfwCreateWindow(width, height, "View correction stub", NULL, NULL);
  if (!window) {
    LOG(ERROR) << "Cannot create GLFW window. Aborting.";
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  
  // Initializing GLEW for the OpenGL context is required by the view correction code.
  glewInit();
  
  // Perform OpenGL initializations of ViewCorrectionDisplay.
  display->Init();
  
  // Main loop.
  steady_clock::time_point start_time = steady_clock::now();
  display->SetStartTime(start_time);
  
  while (!glfwWindowShouldClose(window)) {
    // Update the display with new input, if available.
    // This could also be done asynchronously from different threads.
    steady_clock::time_point now = steady_clock::now();
    uint64_t nanoseconds = duration<double, std::nano>(now - start_time).count();
    double seconds = 1e-9 * nanoseconds;
    
    
    LOG(ERROR) << "Stub: update the display with new poses";
    // TODO: Use the following function to insert your poses:
    // display->ColorCameraPoseCallback(const Sophus::SE3f& G_T_C, uint64_t timestamp);
    
    // Synthetic example for testing purposes:
    constexpr float kRadius = 0.2f;
    Sophus::SE3f G_T_C;  // camera-to-global transformation
    G_T_C.translation() = Eigen::Vector3f(kRadius * std::sin(seconds), kRadius * std::cos(seconds), 0);
    display->ColorCameraPoseCallback(G_T_C, nanoseconds);
    
    
    LOG(ERROR) << "Stub: update the display with new color images";
    // TODO: Use the following function to insert your color images:
    // NOTE, the YUV naming is outdated. The function expects RGB images.
    // display->YUVImageCallback(const ColorImage& image);
    
    // Synthetic example for testing purposes:
    ColorImage synthetic_color_image;
    synthetic_color_image.create(kSyntheticImageHeight, kSyntheticImageWidth);
    synthetic_color_image.set_timestamp_ns(nanoseconds);
    synthetic_color_image.set_G_T_C(G_T_C);
    for (int y = 0; y < kSyntheticImageHeight; ++ y) {
      for (int x = 0; x < kSyntheticImageWidth; ++ x) {
        synthetic_color_image(y, x) =
            cv::Vec3b(127 + 127 * std::sin(0.1 * x + seconds),
                      127 + 127 * std::cos(0.2 * x + seconds),
                      127 + 127 * std::sin(1 + 0.05 * x + seconds));
      }
    }
    display->YUVImageCallback(synthetic_color_image);
    
    
    // NOTE: Use the --vc_depth_source flag to define the input mode, see flags.h:
    //       Pass --vc_depth_source depth_camera or --vc_depth_source mesh.
    if (UsingMeshInput()) {
      LOG(ERROR) << "Stub: update the display with new scene reconstructions";
      // TODO: Use the following function to insert your meshes:
      // display->MeshCallback(const std::shared_ptr<MeshStub>& mesh);
      
      // Synthetic example for testing purposes:
      static std::vector<float> vertex_data = { 1, -1, 1,
                                               -1,  1, 1,
                                                1,  1, 0.2};
      static std::vector<uint8_t> vertex_colors = {255,   0,   0,
                                                     0, 255,   0,
                                                     0,   0, 255};
      static std::vector<uint32_t> index_data = {0, 1, 2};
      
      vertex_data[2] = 2 + std::sin(10 * seconds);
      vertex_data[4] = 2 + std::sin(10 * seconds);
      vertex_data[6] = 2 + std::sin(10 * seconds);
      
      std::shared_ptr<MeshStub> mesh(new MeshStub());
      mesh->vertex_position_data = vertex_data.data();
      mesh->vertex_color_data = vertex_colors.data();
      mesh->face_data = index_data.data();
      mesh->face_count = 1;
      mesh->empty_ = false;
      
      display->MeshCallback(mesh);
    } else if (UsingDepthCameraInput()) {
      LOG(ERROR) << "Stub: update the display with new depth images";
      // TODO: Use the following function to insert your depth maps:
      // display->DepthImageCallback(const DepthImage& image);
      
      // Synthetic example for testing purposes:
      DepthImage synthetic_depth_image;
      synthetic_depth_image.create(kSyntheticImageHeight, kSyntheticImageWidth);
      synthetic_depth_image.set_timestamp_ns(nanoseconds);
      for (int y = 0; y < kSyntheticImageHeight; ++ y) {
        for (int x = 0; x < kSyntheticImageWidth; ++ x) {
          float depth_meters = 1 + 0.3f * std::sin(0.2f * x + seconds);
          // NOTE: For pixels without depth measurement, set the depth to 0.
          synthetic_depth_image(y, x) = 1000 * depth_meters + 0.5f;  // convert to uint16_t millimeters
        }
      }
      display->DepthImageCallback(synthetic_depth_image);
    }
    
    // Given the latest input data, render the corrected view.
    display->Render();
    
    // GLFW buffer swap and event polling.
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  
  // Destroy the display while the OpenGL context is still active.
  display.reset();
  
  glfwTerminate();
  return 0;
}
