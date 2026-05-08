#include "qnx_defs.h"

static const std::vector<EGLint> config_attrib_list = {
  EGL_SURFACE_TYPE,             EGL_WINDOW_BIT,
  EGL_RENDERABLE_TYPE,          EGL_OPENGL_ES2_BIT,
  // We want a pixel-format of RGBX8888 or RGBA8888.
  EGL_RED_SIZE,                 8,
  EGL_GREEN_SIZE,               8,
  EGL_BLUE_SIZE,                8,
  EGL_ALPHA_SIZE,               8,

  EGL_NONE,
};

static const std::vector<EGLint> context_attrib_list = {
  EGL_CONTEXT_CLIENT_VERSION,   2,
  EGL_NONE,
};

static const std::vector<EGLint> surface_attrib_list = {
  EGL_RENDER_BUFFER,            EGL_BACK_BUFFER,
  EGL_NONE,
};

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

absl::Status InitCameraSink(
  mp_camera_info_t &ci,
  const camera_unit_t unit,
  const bool save_video) {
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
  ci.unit = CAMERA_UNIT_INVALID;

  if (unit == CAMERA_UNIT_INVALID) {
    ABSL_LOG(WARNING) << "No camera unit was specified. Falling back to first "
      << "available camera unit.";
  } else if ((unit <= CAMERA_UNIT_NONE) || (unit >= CAMERA_UNIT_NUM_UNITS)) {
    ABSL_LOG(WARNING) << "The specified camera unit is invalid. Falling back to "
      << "first available camera unit.";
  } else {
    ci.unit = unit;
  }

  if (ci.unit == CAMERA_UNIT_INVALID) {
    units = QueryCameraUnits();
    if (units.empty()) {
      ABSL_LOG(ERROR) << "Failed to find any camera units.";
      ret = absl::UnknownError("Failed to find any camera units.");
      goto failure;
    }
    ci.unit = units[0];
  }

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
    // A camera unit can just be a video, which does not allow setting framerate
    // or changing frame dimensions. So we treat these as suggestions instead.
    ci.framerate = 30.0;
    cam_ret = camera_set_vf_property(ci.handle, CAMERA_IMGPROP_FRAMERATE, ci.framerate);
    if (cam_ret != CAMERA_EOK) {
      camera_get_vf_property(ci.handle, CAMERA_IMGPROP_FRAMERATE, &ci.framerate);
    }

    camera_set_vf_property(ci.handle, CAMERA_IMGPROP_WIDTH, 640);

    camera_set_vf_property(ci.handle, CAMERA_IMGPROP_HEIGHT, 480);
  } else {
    cam_ret = camera_get_vf_property(ci.handle, CAMERA_IMGPROP_FRAMERATE, &ci.framerate);
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

  if (screen_get_window_property_iv(si.window, SCREEN_PROPERTY_SIZE, si.size) < 0) {
    ret = absl::ErrnoToStatus(errno, "Failed to get window size.");
    goto failure;
  }

  usage = SCREEN_USAGE_OPENGL_ES2 | SCREEN_USAGE_OPENGL_ES3;
  if (screen_set_window_property_iv(si.window, SCREEN_PROPERTY_USAGE, &usage) < 0) {
    ret = absl::ErrnoToStatus(errno, "Failed to set window usage.");
    goto failure;
  }

  si.initialized = true;

  return ret;

failure:
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

bool ScreenPollKeyDown(const mp_screen_info_t &si, const uint64_t timeout) {
  int type;
  int val;
  for (;;) {
    if (screen_get_event(si.context, si.event, timeout) < 0) {
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

absl::Status InitGLPipeline(mp_gl_info &gli) {
  GLint glint = 0;
  GLint infoLen = 0;
  absl::Status ret = absl::OkStatus();

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

  ret = InitGLPipeline(gli);
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
    eglDestroySurface(gli.display, gli.surface);
    eglDestroyContext(gli.display, gli.context);
    eglTerminate(gli.display);
    gli.initialized = false;
  }
}

absl::Status GLShowMat(
  mp_gl_info_t &gli,
  const int window_width,
  const int window_height,
  const cv::Mat &output) {
  GLint glint = 0;
  // Store the output to the display framebuffer.
  glPixelStorei(GL_UNPACK_ALIGNMENT, (output.step & 3) ? 1 : 4);
  if ((glint = glGetError())) {
    ABSL_LOG(ERROR) << "Failed set alignment for output frame. "
      << "'glPixelStorei' failed with error " << glint << ".";
    return absl::UnknownError("Failed set alignment for output frame.");
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, output.cols, output.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, output.data);
  if ((glint = glGetError())) {
    ABSL_LOG(ERROR) << "Failed to store output frame to GL texture. "
      << "'glTexImage2D' failed with error " << glint << ".";
    return absl::UnknownError("Failed to store output frame to GL texture.");
  }
  glBlitFramebuffer(0, 0, output.cols, output.rows, 0, 0, window_width, window_height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
  if ((glint = glGetError())) {
    ABSL_LOG(ERROR) << "Failed to blit framebuffer. 'glBlitFramebuffer' "
      << "failed with error " << glint << ".";
    return absl::UnknownError("Failed to blit framebuffer.");
  }

  if (eglSwapBuffers(gli.display, gli.surface) != EGL_TRUE) {
    ABSL_LOG(ERROR) << "Failed to swap EGL buffers. 'eglSwapBuffers' "
      << "failed with error " << eglGetError();
    return absl::UnknownError("Failed to swap EGL buffers.");
  }

  return absl::OkStatus();
}
