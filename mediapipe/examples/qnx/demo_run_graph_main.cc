// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
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
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

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

const std::vector<EGLint> config_attrib_list = {
  EGL_SURFACE_TYPE,             EGL_WINDOW_BIT,
  EGL_RENDERABLE_TYPE,          EGL_OPENGL_ES2_BIT,
  // We want a pixel-format of RGBX8888 or RGBA8888.
  EGL_RED_SIZE,                 8,
  EGL_GREEN_SIZE,               8,
  EGL_BLUE_SIZE,                8,
  EGL_ALPHA_SIZE,               8,

  EGL_NONE,
};

const std::vector<EGLint> context_attrib_list = {
  EGL_CONTEXT_CLIENT_VERSION,   2,
  EGL_NONE,
};

const std::vector<EGLint> surface_attrib_list = {
  EGL_RENDER_BUFFER,            EGL_BACK_BUFFER,
  EGL_NONE,
};

//const std::vector<std::pair<std::string, GLuint>> vertex_attrib_list = {
//};

typedef struct mp_gl_info {
  bool initialized;
  EGLDisplay display;
  EGLConfig config;
  EGLContext context;
  EGLSurface surface;
#if 0
  GLuint vert_shader;
  GLuint frag_shader;
  GLuint program;
#endif
  GLuint framebuffer;
  GLuint texture;
} mp_gl_info_t;

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

std::vector<camera_unit_t> QueryCameraUnits() {
  std::vector<camera_unit_t> result;
  unsigned int num_units;
  int ret;

  ret = camera_get_supported_cameras(0, &num_units, nullptr);
  if (ret != CAMERA_EOK) {
    ABSL_LOG(ERROR) << "Failed to query camera units. "
      << "'camera_get_supported_cameras' returned: " << ret;
    return std::vector<camera_unit_t>();
  }

  result.resize(num_units);
  ret = camera_get_supported_cameras(num_units, &num_units, result.data());
  if (ret != CAMERA_EOK) {
    ABSL_LOG(ERROR) << "Failed to query camera units. "
      << "'camera_get_supported_cameras' returned: " << ret;
    return std::vector<camera_unit_t>();
  }

  return result;
}

std::vector<camera_frametype_t> QueryCameraFrametypes(const mp_camera_info_t &ci) {
  std::vector<camera_frametype_t> result;
  unsigned int num_frametypes;
  int ret;

  ret = camera_get_supported_vf_frame_types(ci.handle, 0, &num_frametypes, NULL);
  if (ret != CAMERA_EOK) {
    ABSL_LOG(ERROR) << "Failed to query camera frametypes. "
      << "'camera_get_supported_vf_frame_types' returned error " << ret;
    return std::vector<camera_frametype_t>();
  }

  result.resize(num_frametypes);
  ret = camera_get_supported_vf_frame_types(ci.handle, num_frametypes, &num_frametypes, result.data());
  if (ret != CAMERA_EOK) {
    ABSL_LOG(ERROR) << "Failed to query camera frametypes. "
      << "'camera_get_supported_vf_frame_types' returned error " << ret;
    return std::vector<camera_frametype_t>();
  }

  return result;
}

static void CameraStatusCallback(camera_handle_t handle, camera_devstatus_t status, uint16_t extra, void *arg)
{
    switch (status) {
    case CAMERA_STATUS_VIDEOVF:
      ABSL_LOG(INFO) << "The camera viewfinder has started streaming.";
      break;
    case CAMERA_STATUS_VIEWFINDER_ACTIVE:
      ABSL_LOG(INFO) << "The camera viewfinder is active.";
      (void) printf("STATUS CALLBACK: The viewfinder is active\n");
      break;
    case CAMERA_STATUS_VIDEO_RESUME:
      ABSL_LOG(INFO) << "Camera video encoding has started.";
      break;
    case CAMERA_STATUS_MM_ERROR:
      ABSL_LOG(ERROR) << "Camera recording has stopped due to an encoding error.";
      break;
    case CAMERA_STATUS_NOSPACE_ERROR:
      ABSL_LOG(ERROR) << "Camera recording has run out of disk space and stopped.";
      break;
    default:
      ABSL_LOG(INFO) << "Camera received status " << status << ".";
      break;
    }
}

void CameraProduceData(
  mp_camera_info_t &ci,
  camera_buffer_t* buffer_p) {
  cv::Mat frame;

  // Conversions taken from https://gitlab.com/qnx/projects/ai-camera-app/-/blob/main/FaceDetection/QSFCameraIntake.cpp
  switch(buffer_p->frametype) {
  case CAMERA_FRAMETYPE_NV12:
    {
      cv::Mat frame_raw(
          buffer_p->framedesc.nv12.height,
          buffer_p->framedesc.nv12.width, CV_8UC2,
          buffer_p->framebuf,
          buffer_p->framedesc.nv12.width);
      cv::cvtColor(frame_raw, frame, cv::COLOR_YUV2RGB_NV12);
    }
  case CAMERA_FRAMETYPE_YCBYCR:
    {
      cv::Mat frame_raw(
          buffer_p->framedesc.ycbycr.height,
          buffer_p->framedesc.ycbycr.width, CV_8UC2,
          buffer_p->framebuf,
          buffer_p->framedesc.ycbycr.width * 2);
      cv::cvtColor(frame_raw, frame, cv::COLOR_YUV2RGB_YUY2);
    }
    break;
  case CAMERA_FRAMETYPE_CBYCRY:
    {
      cv::Mat frame_raw(
          buffer_p->framedesc.cbycry.height,
          buffer_p->framedesc.cbycry.width, CV_8UC2,
          buffer_p->framebuf,
          buffer_p->framedesc.cbycry.width * 2);
      cv::cvtColor(frame_raw, frame, cv::COLOR_YUV2RGB_UYVY);
    }
    break;
  case CAMERA_FRAMETYPE_RGB888:
    frame.create(buffer_p->framedesc.rgb888.height, buffer_p->framedesc.rgb888.width, CV_8UC3);
    memcpy(
        reinterpret_cast<char*>(frame.data),
        reinterpret_cast<char*>(buffer_p->framebuf),
        buffer_p->framedesc.rgb888.height * buffer_p->framedesc.rgb888.width * 3);
    break;
  case CAMERA_FRAMETYPE_RGB8888:
    {
      const int from_to[8] { 0, 2, 1, 1, 2, 0, 3, 3 };
      cv::Mat frame_raw_argb(
          buffer_p->framedesc.rgb8888.height,
          buffer_p->framedesc.rgb8888.width, CV_8UC4,
          buffer_p->framebuf,
          buffer_p->framedesc.rgb8888.width * 4);
      cv::Mat frame_raw_bgra(frame_raw_argb.size(), frame_raw_argb.type());
      cv::mixChannels(&frame_raw_argb, 1, &frame_raw_bgra, 1, from_to, 4);
      cv::cvtColor(frame_raw_bgra, frame, cv::COLOR_RGBA2RGB);
    }
    break;
  case CAMERA_FRAMETYPE_BGR8888:
    {
      cv::Mat frame_raw(
          buffer_p->framedesc.bgr8888.height,
          buffer_p->framedesc.bgr8888.width, CV_8UC4,
          buffer_p->framebuf,
          buffer_p->framedesc.bgr8888.width * 4);
      cv::cvtColor(frame_raw, frame, cv::COLOR_BGRA2RGB);
    }
    break;
  default:
    ABSL_LOG(ERROR) << "The camera frametype is invalid.";
    return;
  }
  // We don't need data to be empty before writing the buffer.
  {
    std::lock_guard<std::mutex> data_guard(ci.data_m);

    ci.data = frame;
    ci.data_ready = true;
  }
  ci.data_cv.notify_one();
}

cv::Mat CameraConsumeData(mp_camera_info_t &ci) {
  cv::Mat ret;
  std::unique_lock data_lk(ci.data_m);
  ci.data_cv.wait(data_lk, [&]() { return ci.data_ready; });

  ret = ci.data;
  ci.data_ready = false;

  data_lk.unlock();

  return ret;
}

static void CameraViewfinderCallback(
  camera_handle_t handle,
  camera_buffer_t* buffer_p,
  void* arg) {
  mp_camera_info_t *ci_p = reinterpret_cast<mp_camera_info_t*>(arg);
  CameraProduceData(*ci_p, buffer_p);
}

absl::Status InitCameraSink(mp_camera_info_t &ci, const bool save_video) {
  std::vector<camera_unit_t> units;
  std::vector<camera_frametype_t> frametypes;
  camera_frametype_t frametype = CAMERA_FRAMETYPE_UNSPECIFIED;
  int cam_ret;
  absl::Status ret = absl::OkStatus();

  if (ci.initialized) {
    return ret;
  }

  // Set default values.
  ci.handle = static_cast<camera_handle_t>(-1);

  units = QueryCameraUnits();
  if (units.empty()) {
    ABSL_LOG(ERROR) << "Failed to find any camera units.";
    ret = absl::UnknownError("Failed to find any camera units.");
    goto failure;
  }
  ci.unit = units[0];

  cam_ret = camera_open(ci.unit, CAMERA_MODE_RO | CAMERA_MODE_ROLL | CAMERA_MODE_PWRITE, &ci.handle);
  if (cam_ret != CAMERA_EOK) {
    ABSL_LOG(ERROR) << "Failed to open camera. 'camera_open' returned error "
      << cam_ret << " (" << strerror(cam_ret) << ").";
    ret = absl::ErrnoToStatus(cam_ret, "Failed to open camera.");
    goto failure;
  }

  cam_ret = camera_get_vf_property(ci.handle, CAMERA_IMGPROP_FORMAT, &frametype);
  if (cam_ret != CAMERA_EOK) {
    ABSL_LOG(ERROR) << "Failed to set CAMERA_IMGPROP_FORMAT property. "
      << "'camera_set_vf_property' returned error " << cam_ret << " ("
      << strerror(cam_ret) << ").";
    ret = absl::ErrnoToStatus(cam_ret, "Failed to set camera property.");
    goto failure;
  }
  switch(frametype) {
  case CAMERA_FRAMETYPE_NV12:
  case CAMERA_FRAMETYPE_YCBYCR:
  case CAMERA_FRAMETYPE_CBYCRY:
  case CAMERA_FRAMETYPE_RGB888:
  case CAMERA_FRAMETYPE_RGB8888:
  case CAMERA_FRAMETYPE_BGR8888:
    break;
  default:
    ABSL_LOG(ERROR) << "The configured frametype is not supported.";
    ret = absl::UnknownError("The configured frametype is not supported.");
    goto failure;
    break;
  }

  // Don't create a window automatically.
  cam_ret = camera_set_vf_property(ci.handle, CAMERA_IMGPROP_CREATEWINDOW, false);
  if (cam_ret != CAMERA_EOK) {
    ABSL_LOG(ERROR) << "Failed to set CAMERA_IMGPROP_CREATEWINDOW property. "
      << "'camera_set_vf_property' returned error " << cam_ret << " ("
      << strerror(cam_ret) << ").";
    ret = absl::ErrnoToStatus(cam_ret, "Failed to set camera property.");
    goto failure;
  }

  if (!save_video) {
    ci.framerate = 30.0;
    cam_ret = camera_set_vf_property(ci.handle, CAMERA_IMGPROP_FRAMERATE, ci.framerate);
    if (cam_ret != CAMERA_EOK) {
      ABSL_LOG(ERROR) << "Failed to set CAMERA_IMGPROP_FRAMERATE property. "
        << "'camera_set_vf_property' returned error " << cam_ret << " ("
        << strerror(cam_ret) << ").";
      ret = absl::ErrnoToStatus(cam_ret, "Failed to set camera property.");
      goto failure;
    }

    cam_ret = camera_set_vf_property(ci.handle, CAMERA_IMGPROP_WIDTH, 640);
    if (cam_ret != CAMERA_EOK) {
      ABSL_LOG(ERROR) << "Failed to set CAMERA_IMGPROP_WIDTH property. "
        << "'camera_set_vf_property' returned error " << cam_ret << " ("
        << strerror(cam_ret) << ").";
      ret = absl::ErrnoToStatus(cam_ret, "Failed to set camera property.");
      goto failure;
    }

    cam_ret = camera_set_vf_property(ci.handle, CAMERA_IMGPROP_HEIGHT, 480);
    if (cam_ret != CAMERA_EOK) {
      ABSL_LOG(ERROR) << "Failed to set CAMERA_IMGPROP_HEIGHT property. "
        << "'camera_set_vf_property' returned error " << cam_ret << " ("
        << strerror(cam_ret) << ").";
      ret = absl::ErrnoToStatus(cam_ret, "Failed to set camera property.");
      goto failure;
    }
  } else {
    cam_ret = camera_get_vf_property(ci.handle, CAMERA_IMGPROP_WIDTH, CAMERA_IMGPROP_FRAMERATE, &ci.framerate);
    if (cam_ret != CAMERA_EOK) {
      ABSL_LOG(ERROR) << "Failed to get CAMERA_IMGPROP_FRAMERATE property. "
        << "'camera_get_vf_property' returned error " << cam_ret << " ("
        << strerror(cam_ret) << ").";
      ret = absl::ErrnoToStatus(cam_ret, "Failed to get camera property.");
      goto failure;
    }
  }

  cam_ret = camera_start_viewfinder(ci.handle, CameraViewfinderCallback, CameraStatusCallback, &ci);
  if (cam_ret != CAMERA_EOK) {
    ABSL_LOG(ERROR) << "Failed to start viewfinder. 'camera_start_viewfinder' "
      << "returned error " << cam_ret << " (" << strerror(cam_ret) << ").";
    ret = absl::ErrnoToStatus(cam_ret, "Failed to start viewfinder.");
    goto failure;
  }

  ci.initialized = true;

  return ret;

failure:
  if (ci.handle != static_cast<camera_handle_t>(-1)) {
    camera_close(ci.handle);
    ci.handle = static_cast<camera_handle_t>(-1);
  }
  return ret;
}

void TeardownCameraSink(mp_camera_info_t &ci) {
  if (ci.initialized) {
    camera_close(ci.handle);
    ci.initialized = false;
  }
}

absl::Status InitScreenWindow(mp_screen_info_t &si) {
  screen_display_t *screen_display_p = nullptr;
  int usage;
  absl::Status ret = absl::OkStatus();

  if (si.initialized) {
    return ret;
  }

  memset(&si, 0, sizeof(mp_screen_info_t));
  // These are opaque pointer types. -1 should be the safest value.
  si.context = (screen_context_t) -1;
  si.event = (screen_event_t) -1;
  si.window = (screen_window_t) -1;

  // Allocate screen handles.
  if (screen_create_context(&si.context, 0) < 0) {
    ret = absl::ErrnoToStatus(errno, "Failed to create screen context.");
    goto failure;
  }

  if (screen_create_event(&si.event) < 0) {
    ret = absl::ErrnoToStatus(errno, "Failed to create screen event.");
    goto failure;
  }

  if (screen_create_window(&si.window, si.context) < 0) {
    ret = absl::ErrnoToStatus(errno, "Failed to create screen window.");
    goto failure;
  }

  // Force the window to fullscreen by setting its size to the size of the
  // display.
  screen_display_p = (screen_display_t *) malloc(sizeof(screen_display_t));
  if (screen_display_p == NULL) {
    ret = absl::ErrnoToStatus(errno, "Failed to allocate screen display.");
    goto failure;
  }

  if (screen_get_window_property_pv(si.window, SCREEN_PROPERTY_DISPLAY, (void **) &screen_display_p) < 0) {
    ret = absl::ErrnoToStatus(errno, "Failed to get screen display.");
    goto failure;
  }

  if (screen_get_display_property_iv(*screen_display_p, SCREEN_PROPERTY_SIZE, si.size) < 0) {
    ret = absl::ErrnoToStatus(errno, "Failed to get screen display size.");
    goto failure;
  }

  ABSL_LOG(INFO) << "Defaulting to fullscreen window size of: ("
    << si.size[0] << ", " << si.size[1] << ")";
  if (screen_set_window_property_iv(si.window, SCREEN_PROPERTY_SOURCE_SIZE, si.size) < 0) {
    ret = absl::ErrnoToStatus(errno, "Failed to set window source size.");
    goto failure;
  }

  usage = SCREEN_USAGE_OPENGL_ES2 | SCREEN_USAGE_OPENGL_ES3;
  if (screen_set_window_property_iv(si.window, SCREEN_PROPERTY_USAGE, &usage) < 0) {
    ret = absl::ErrnoToStatus(errno, "Failed to set window usage.");
    goto failure;
  }

  si.initialized = true;

  if (screen_display_p != nullptr) {
    free(screen_display_p);
  }
  return ret;

failure:
  if (screen_display_p != nullptr) {
    free(screen_display_p);
  }
  if (si.window != (screen_window_t) -1) {
    screen_destroy_window(si.window);
    si.window = (screen_window_t) -1;
  }
  if (si.event != (screen_event_t) -1) {
    screen_destroy_event(si.event);
    si.event = (screen_event_t) -1;
  }
  if (si.context != (screen_context_t) -1) {
    screen_destroy_context(si.context);
    si.context = (screen_context_t) -1;
  }
  return ret;
}

void TeardownScreenWindow(mp_screen_info_t &si) {
  if (si.initialized) {
    screen_destroy_window(si.window);
    screen_destroy_event(si.event);
    screen_destroy_context(si.context);
    si.initialized = false;
  }
}

bool ScreenPollKeyDown(const mp_screen_info_t &si) {
  int type;
  int val;
  for (;;) {
    if (screen_get_event(si.context, si.event, 0) < 0) {
      return false;
    }

    if (screen_get_event_property_iv(si.event, SCREEN_PROPERTY_TYPE, &type) < 0) {
      return false;
    }

    if (type == SCREEN_EVENT_NONE) {
      return false;
    }

    // Find a key down event.
    if ((type == SCREEN_EVENT_KEYBOARD)
      && (screen_get_event_property_iv(si.event, SCREEN_PROPERTY_FLAGS, &val) == 0)
      && ((val & SCREEN_FLAG_KEY_DOWN) == SCREEN_FLAG_KEY_DOWN)) {
      return true;
    }
  }

  return false;
}

std::vector<EGLConfig> QueryEGLConfigs(mp_gl_info_t &gli) {
  EGLBoolean ret;
  std::vector<EGLConfig> result;
  EGLint num_configs;

  ret = eglChooseConfig(gli.display, config_attrib_list.data(), nullptr, 0, &num_configs);
  if (ret != EGL_TRUE) {
    ABSL_LOG(ERROR) << "Failed to query egl configs. 'eglChooseConfig' returned: "
      << eglGetError();
    return std::vector<EGLConfig>();
  }
  result.resize(num_configs);

  ret = eglChooseConfig(gli.display, config_attrib_list.data(), result.data(), num_configs, &num_configs);
  if (ret != EGL_TRUE) {
    ABSL_LOG(ERROR) << "Failed to query egl configs. 'eglChooseConfig' returned: "
      << eglGetError();
    return std::vector<EGLConfig>();
  }

  return result;
}

// https://www.khronos.org/assets/uploads/books/openglr_es_20_programming_guide_sample.pdf
static GLuint LoadGLShader(GLenum type, const char *src) {
  GLuint shader;
  GLint compiled;

  shader = glCreateShader(type);
  if(shader == 0) {
    ABSL_LOG(ERROR) << "Failed to create shader. 'glCreateShader' failed with "
      << "error " << glGetError();
    return 0;
  }
  glShaderSource(shader, 1, &src, NULL);

  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if(!compiled)
  {
    GLint infoLen = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
    if(infoLen > 1)
    {
      char* infoLog = static_cast<char*>(malloc(sizeof(char) * infoLen));
      glGetShaderInfoLog(shader, infoLen, NULL, infoLog);
      ABSL_LOG(ERROR) << "Failed to compile shader. Output:" << std::endl
        << infoLog;
      free(infoLog);
    }
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}

absl::Status InitGLPipeline(
  mp_gl_info &gli,
  const std::string vert_shader_src, const std::string frag_shader_src) {
  GLint glint = 0;
  GLint infoLen = 0;
  absl::Status ret = absl::OkStatus();

#if 0
  // Create and attach shaders.
  std::ifstream vert_shader_f(vert_shader_src);
  std::stringstream buffer;
  buffer << vert_shader_f.rdbuf();
  gli.vert_shader = LoadGLShader(GL_VERTEX_SHADER, buffer.str());
  if (!gli.vert_shader) {
    ABSL_LOG(ERROR) << "Failed to create vertex shader.";
    ret = absl::UnknownError("Failed to create vertex shader.");
    goto failure;
  }

  std::ifstream frag_shader_f(frag_shader_src);
  buffer.str(std::string());
  buffer << frag_shader_f.rdbuf();
  gli.frag_shader = LoadGLShader(GL_FRAGMENT_SHADER, buffer.str());
  if (!gli.frag_shader) {
    ABSL_LOG(ERROR) << "Failed to create fragment shader.";
    ret = absl::UnknownError("Failed to create fragment shader.");
    goto failure;
  }

  gli.program = glCreateProgram();
  if (gli.program == 0) {
    ABSL_LOG(ERROR) << "Failed to link GL program. 'glLinkProgram' failed "
      << "with error " << glint << ".";
    ret = absl::UnknownError("Failed to link GL program.");
    goto failure;
  }
  glAttachShader(gli.program, gli.vert_shader);
  glAttachShader(gli.program, gli.frag_shader);

  // Bind any attributes to the vertex shader
  for (const auto &attrib : vertex_attrib_list) {
    const std::string id = attrib.first;
    const GLuint location = attrib.second;
    glBindAttribLocation(gli.program, id, location);
    if ((glint = glGetError())) {
      ABSL_LOG(ERROR) << "Failed to bind GL attribute " << id << " at location "
        << location << ". 'glBindAttribLocation' failed with error " << glint << ".";
      ret = absl::UnknownError("Failed to bind GL attribute.");
      goto failure;
    }
  }

  // Link the program
  glLinkProgram(gli.program);
  if ((glint = glGetError())) {
    ABSL_LOG(ERROR) << "Failed to link GL program. 'glLinkProgram' failed "
      << "with error " << glint << ".";
    ret = absl::UnknownError("Failed to link GL program.");
    goto failure;
  }
  glGetProgramiv(gli.program, GL_LINK_STATUS, &glint);
  if (!glint) {
    glGetProgramiv(gli.program, GL_INFO_LOG_LENGTH, &infoLen);
    if(infoLen > 1)
    {
      char* infoLog = static_cast<char*>(malloc(sizeof(char) * infoLen));
      glGetProgramInfoLog(gli.program, infoLen, NULL, infoLog);
      ABSL_LOG(ERROR) << "Failed to link program. Output:" << std::endl
        << infoLog;
      free(infoLog);
    }
    ret = absl::UnknownError("Failed to link GL program.");
    goto failure;
  }

  // Switch to using this program
  glUseProgram(gli.program);
  if ((glint = glGetError())) {
    ABSL_LOG(ERROR) << "Failed to use GL program. 'glUseProgram' failed "
      << "with error " << glint << ".";
    ret = absl::UnknownError("Failed to use GL program.");
    goto failure;
  }
#endif

  glActiveTexture(GL_TEXTURE0);

  // Initialize the texture.
  glGenTextures(1, &gli.texture);
  if ((glint = glGetError())) {
    ABSL_LOG(ERROR) << "Failed to generate GL texture. 'glGenTextures' "
      << "failed with error " << glint << ".";
    ret = absl::UnknownError("Failed to generate GL texture.");
    goto failure;
  }

  glBindTexture(GL_TEXTURE_2D, gli.texture);
  if ((glint = glGetError())) {
    ABSL_LOG(ERROR) << "Failed to bind GL texture. 'glBindTexture' "
      << "failed with error " << glint << ".";
    ret = absl::UnknownError("Failed to bind GL texture.");
    goto failure;
  }

  // Initialize framebuffers for blitting.
  glGenFramebuffers(1, &gli.framebuffer);
  if ((glint = glGetError())) {
    ABSL_LOG(ERROR) << "Failed to create GL framebuffer. 'glGenFramebuffers' "
      << "failed with error " << glint << ".";
    ret = absl::UnknownError("Failed to create GL framebuffer.");
    goto failure;
  }
  // TODO: Does this go here?
  glBindFramebuffer(GL_READ_FRAMEBUFFER, gli.framebuffer);
  if ((glint = glGetError())) {
    ABSL_LOG(ERROR) << "Failed to bind GL read framebuffer. "
      << "'glBindFramebuffer' failed with error " << glint << ".";
    ret = absl::UnknownError("Failed to bind GL framebuffer.");
    goto failure;
  }
  glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gli.texture, 0);
  if ((glint = glGetError())) {
    ABSL_LOG(ERROR) << "Failed to attach GL texture to framebuffer. "
      << "'glFramebufferTexture2D' failed with error " << glint << ".";
    ret = absl::UnknownError("Failed to attach GL texture to framebuffer.");
    goto failure;
  }
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
  if ((glint = glGetError())) {
    ABSL_LOG(ERROR) << "Failed to bind GL write framebuffer. "
      << "'glBindFramebuffer' failed with error " << glint << ".";
    ret = absl::UnknownError("Failed to bind GL framebuffer.");
    goto failure;
  }

  return ret;

failure:
  if (gli.texture != 0) {
    glDeleteTextures(1, &gli.texture);
    gli.texture = 0;
  }
  if (gli.framebuffer != 0) {
    glDeleteFramebuffers(1, &gli.framebuffer);
    gli.framebuffer = 0;
  }
#if 0
  if (gli.program != 0) {
    glDeleteProgram(gli.program);
    gli.program = 0;
  }
  if (gli.vert_shader != 0) {
    glDeleteShader(gli.vert_shader);
    gli.vert_shader = 0;
  }
  if (gli.frag_shader != 0) {
    glDeleteShader(gli.frag_shader);
    gli.frag_shader = 0;
  }
#endif
  return ret;
}

absl::Status InitGLContext(mp_gl_info_t &gli, const mp_screen_info_t &si) {
  std::vector<EGLConfig> configs;
  absl::Status ret = absl::OkStatus();

  if (gli.initialized) {
    return ret;
  }

  memset(&gli, 0, sizeof(mp_gl_info_t));

  gli.display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  eglInitialize(gli.display, nullptr, nullptr);

  configs = QueryEGLConfigs(gli);
  if (configs.empty()) {
    ABSL_LOG(ERROR) << "Failed to find an appropriate EGL config.";
    ret = absl::UnknownError("Failed to find an appropriate EGL config.");
    goto failure;
  }
  gli.config = configs[0];

  gli.context = eglCreateContext(gli.display, gli.config, EGL_NO_CONTEXT, context_attrib_list.data());
  if (gli.context == EGL_NO_CONTEXT) {
    ABSL_LOG(ERROR) << "Failed to create EGL context. 'eglCreateContext' returned: "
      << eglGetError();
    ret = absl::UnknownError("Failed to create EGL context.");
    goto failure;
  }

  gli.surface = eglCreateWindowSurface(gli.display, gli.config, (EGLNativeWindowType)si.window, surface_attrib_list.data());
  if (gli.surface == EGL_NO_SURFACE) {
    ABSL_LOG(ERROR) << "Failed to create EGL surface. 'eglCreateWindowSurface' returned: "
      << eglGetError();
    ret = absl::UnknownError("Failed to create EGL context.");
    goto failure;
  }

  eglMakeCurrent(gli.display, gli.surface, gli.surface, gli.context);

  // TODO: Do these paths need to be relocated?
  ret = InitGLPipeline(gli, "demo.vert", "demo.frag");
  if (!ret.ok()) {
    ABSL_LOG(ERROR) << "Failed to initialize GL pipeline.";
    goto failure;
  }

  gli.initialized = true;

  return ret;

failure:
  if (gli.surface != EGL_NO_SURFACE) {
    eglDestroySurface(gli.display, gli.surface);
    gli.surface = EGL_NO_SURFACE;
  }
  if (gli.context != EGL_NO_CONTEXT) {
    eglDestroyContext(gli.display, gli.context);
    gli.context = EGL_NO_CONTEXT;
  }
  if (gli.display != EGL_NO_DISPLAY) {
    eglTerminate(gli.display);
    gli.display = EGL_NO_DISPLAY;
  }
  return ret;
}

void TeardownGLContext(mp_gl_info_t &gli) {
  if (gli.initialized) {
    glDeleteFramebuffers(1, &gli.framebuffer);
#if 0
    glDeleteProgram(gli.program);
    glDeleteShader(gli.vert_shader);
    glDeleteShader(gli.frag_shader);
#endif
    eglDestroySurface(gli.display, gli.surface);
    eglDestroyContext(gli.display, gli.context);
    eglTerminate(gli.display);
    gli.initialized = false;
  }
}

absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  absl::Status ret;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  ABSL_LOG(INFO) << "Get calculator graph config contents: "
                 << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();

  ABSL_LOG(INFO) << "Initialize the camera or load the video.";
  mp_camera_info_t ci = {};
  ret = InitCameraSink(ci, save_video);
  if (!ret.ok()) {
    return ret;
  }
#if 0
  // FIXME: Add this!
  const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (load_video) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else {
    capture.open(0);
  }
#endif

  cv::VideoWriter writer;

  ABSL_LOG(INFO) << "Initialize the screen window.";
  mp_screen_info_t si = {};
  ret = InitScreenWindow(si);
  if (!ret.ok()) {
    return ret;
  }

  mp_gl_info_t gli = {};
  ret = InitGLContext(gli, si);
  if (!ret.ok()) {
    return ret;
  }

  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                      graph.AddOutputStreamPoller(kOutputStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    // Capture sensor framework camera or video frame.
    // The frame is already in the expected format.
    cv::Mat camera_frame = CameraConsumeData(ci);
    if (camera_frame.empty()) {
#if 0
      if (!load_video) {
        ABSL_LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      ABSL_LOG(INFO) << "Empty frame, end of video reached.";
      break;
#else
      ABSL_LOG(INFO) << "Ignore empty frames from camera.";
      continue;
#endif
    }
#if 0
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }
#else
    cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
#endif

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;
    auto& output_frame = packet.Get<mediapipe::ImageFrame>();

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    if (save_video) {
      if (!writer.isOpened()) {
        ABSL_LOG(INFO) << "Prepare video writer.";
        writer.open(absl::GetFlag(FLAGS_output_video_path),
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    ci.framerate, output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      // Store the output to the display framebuffer.
      glPixelStorei(GL_UNPACK_ALIGNMENT, (output_frame_mat.step & 3) ? 1 : 4);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, output_frame_mat.cols, output_frame_mat.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, output_frame_mat.data);
      glBlitFramebuffer(0, 0, output_frame_mat.cols, output_frame_mat.rows, 0, 0, si.size[0], si.size[1], GL_COLOR_BUFFER_BIT, GL_LINEAR);
      // Press any key to exit.
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (ScreenPollKeyDown(si)) {
        grab_frames = false;
      }
    }
  }

  ABSL_LOG(INFO) << "Shutting down.";
  TeardownGLContext(gli);
  TeardownScreenWindow(si);
  TeardownCameraSink(ci);
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    ABSL_LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
