cmake_minimum_required(VERSION 2.8.11)

project(view_correction)

# Make CMake find the Find<Package>.cmake files.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")


################################################################################
# Dependencies and settings.

# CUDA (external)
find_package(CUDA REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/SelectCudaComputeArch.cmake)
# if(CUDA_MULTI_ARCH)
#     CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS All)
# else()
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)
# endif()
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
list(APPEND CUDA_NVCC_FLAGS "-use_fast_math")

# Eigen (external)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# GLog (external)
find_package(Glog 0.3.4 REQUIRED)

# Sophus (packaged)
include_directories(src/sophus)

# GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

# CUB (packaged)
include_directories(src/cub-1.7.4)

# OpenCV (external)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# GLFW (external)
find_package(glfw3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})


# Settings.
add_definitions("-Wall -Wextra -O2 -msse2 -msse3 -std=c++11")

# TODO: It seems gcc cannot disable this locally, therefore added it here as a
#       workaround.
add_definitions("-Wno-unknown-pragmas")

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUXX)
  add_definitions("-Wno-missing-field-initializers")
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})


################################################################################
# view_correction.

include_directories(
  ./src
)

cuda_add_executable(view_correction
  src/view_correction/cuda_buffer.cu
  src/view_correction/cuda_buffer.cuh
  src/view_correction/cuda_buffer.h
  src/view_correction/cuda_buffer_adapter.h
  src/view_correction/cuda_buffer_adapter_inl.h
  src/view_correction/cuda_buffer_inl.h
  src/view_correction/cuda_buffer_visualization.h
  src/view_correction/cuda_buffer_visualization_inl.h
  src/view_correction/cuda_convolution_inpainting_rgb.cu
  src/view_correction/cuda_convolution_inpainting_rgb.cuh
  src/view_correction/cuda_convolution_inpainting.cu
  src/view_correction/cuda_convolution_inpainting.cuh
  src/view_correction/cuda_tv_inpainting_functions.cu
  src/view_correction/cuda_tv_inpainting_functions.cuh
  src/view_correction/cuda_util.h
  src/view_correction/flags.cc
  src/view_correction/flags.h
  src/view_correction/forward_declarations.h
  src/view_correction/main.cc
  src/view_correction/mesh_renderer.cc
  src/view_correction/mesh_renderer.h
  src/view_correction/opengl_util.cc
  src/view_correction/opengl_util.h
  src/view_correction/position_receiver.cc
  src/view_correction/position_receiver.h
  src/view_correction/util.cc
  src/view_correction/util.h
  src/view_correction/view_correction_display.cc
  src/view_correction/view_correction_display.cu
  src/view_correction/view_correction_display.cuh
  src/view_correction/view_correction_display.h
)

target_link_libraries(view_correction
  ${OpenCV_LIBS}
  glfw
  ${GLEW_LIBRARIES}
  GL
  glog
  gflags
  pthread
)
