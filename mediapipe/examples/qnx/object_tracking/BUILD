# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/qnx/object_tracking:object_tracking_cpu
# RUN:
# bazel-bin/mediapipe/examples/qnx/object_tracking/object_tracking_cpu \
#   --calculator_graph_config_file=<.../object_detection_tracking_desktop_live.pbtxt.pbtxt> \
#   --camera_unit=<camera unit> \
#   --output_video_path=<output video path>
cc_binary(
    name = "object_tracking_cpu",
    data = [
        "//mediapipe/models:ssdlite_object_detection.tflite",
        "//mediapipe/models:ssdlite_object_detection_labelmap.txt",
    ],
    deps = [
        "//mediapipe/examples/qnx:demo_run_graph_main",
        "//mediapipe/graphs/tracking:desktop_calculators",
    ],
)

# BUILD:
# bazel build -c opt //mediapipe/examples/qnx/object_tracking:object_tracking_gpu
# RUN:
# bazel-bin/mediapipe/examples/qnx/object_tracking/object_tracking_gpu \
#   --calculator_graph_config_file=<.../object_detection_tracking_desktop_live.pbtxt.pbtxt> \
#   --camera_unit=<camera unit> \
#   --output_video_path=<output video path>
cc_binary(
    name = "object_tracking_gpu",
    data = [
        "//mediapipe/models:ssdlite_object_detection.tflite",
        "//mediapipe/models:ssdlite_object_detection_labelmap.txt",
    ],
    deps = [
        "//mediapipe/examples/qnx:demo_run_graph_main_gpu",
        "//mediapipe/graphs/tracking:mobile_calculators",
    ],
)
