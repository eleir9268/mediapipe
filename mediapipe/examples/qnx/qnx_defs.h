#pragma once

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <ostream>
#include <thread>
#include <utility>
#include <vector>

#include <camera/camera_api.h>
#include <screen/screen.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/status.h"

typedef struct mp_camera_info {
  bool initialized;
  camera_unit_t unit;
  camera_handle_t handle;
  double framerate;
  std::mutex data_m;
  std::condition_variable data_cv;
  bool data_ready;
  cv::Mat data;
} mp_camera_info_t;

typedef struct mp_screen_info {
  bool initialized;
  screen_context_t context;
  screen_event_t   event;
  screen_window_t window;
  int size[2];
} mp_screen_info_t;

typedef struct mp_gl_info {
  bool initialized;
  EGLDisplay display;
  EGLConfig config;
  EGLContext context;
  EGLSurface surface;
  GLuint framebuffer;
  GLuint texture;
} mp_gl_info_t;

std::vector<camera_unit_t> QueryCameraUnits();
std::vector<camera_frametype_t> QueryCameraFrametypes(const mp_camera_info_t &ci);
void CameraProduceData(mp_camera_info_t &ci, camera_buffer_t* buffer_p);
cv::Mat CameraConsumeData(mp_camera_info_t &ci);
absl::Status InitCameraSink(mp_camera_info_t &ci, const camera_unit_t unit, const bool save_video);
void TeardownCameraSink(mp_camera_info_t &ci);

absl::Status InitScreenWindow(mp_screen_info_t &si);
void TeardownScreenWindow(mp_screen_info_t &si);
bool ScreenPollKeyDown(const mp_screen_info_t &si, const uint64_t timeout);

std::vector<EGLConfig> QueryEGLConfigs(mp_gl_info_t &gli);
absl::Status InitGLPipeline(mp_gl_info &gli);
absl::Status InitGLContext(mp_gl_info_t &gli, const mp_screen_info_t &si);
void TeardownGLContext(mp_gl_info_t &gli);
absl::Status GLShowMat(mp_gl_info_t &gli, const int window_width, const int window_height, const cv::Mat &output);
