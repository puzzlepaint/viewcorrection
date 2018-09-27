# Real-Time View Correction for Mobile Devices

This repository contains the code for the following paper:

T. Schöps, M. R. Oswald, P. Speciale, S. Yang, M. Pollefeys, "Real-Time View Correction for Mobile Devices", Special issue of TVCG on ISMAR 2017. Presented at ISMAR 2017. \[[pdf](http://cvg.ethz.ch/research/view-correction/paper/Schoepst2017ISMAR.pdf)\] \[[website](http://cvg.ethz.ch/research/view-correction/)\] \[[bib](http://cvg.ethz.ch/research/view-correction/paper/Schoepst2017ISMAR.bib)\]

For this open source release as a Linux application, we removed the tight integration with the Google Tango framework of the version which was used to generate the result for the paper.
The provided code merely runs the view correction on synthetic data for testing purposes.
To use the code on real data, you have to modify the application and provide your own camera calibration and poses, color images, depth images or reconstructed meshes, and the source-target transformation.


## Building ##

Building has been tested on Ubuntu 14.04 only. It is expected that later
versions of Ubuntu also work with little effort.

The following external dependencies are required:

* CUDA (version 8 is known to work)
* Eigen
* GLEW
* GLFW
* GLog
* OpenCV

After obtaining all dependencies, the application can be built with CMake, for example as follows:
```
mkdir build_RelWithDebInfo
cd build_RelWithDebInfo
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j
```


## Running and Adapting ##

To run the test application, run the view_correction executable:
```
./view_correction
```

If you get a test output like the following, it works correctly:
![Screenshot](screenshot.jpg?raw=true)

See flags.cc for a list of program arguments.
For example, `--vc_debug` will show images of various steps in the processing pipeline.

The provided test application in main.cc can be used as a starting point to pass in your own input data.
Furthermore, you may want to modify GetCurrentTimestamp(), GetCurrentColorCameraPose(), and GetYUVImagePose() in view_correction_display.h to always return your most current values respectively estimates of these values.
The internal rendering resolution is set by the `target_render_width_` and `target_render_height_` assignment in the ViewCorrectionDisplay constructor in view_correction_display.cc.
To provide the source-target transformation, you could modify SetupTargetView() in view_correction_display.cc, for example by passing in the screen-camera calibration for your device.
It is assumed that the input images are from a pinhole camera.
