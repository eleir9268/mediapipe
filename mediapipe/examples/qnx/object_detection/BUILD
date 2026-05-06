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
#   //mediapipe/examples/desktop/object_detection:object_detection_gpu
# RUN:
# bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_gpu \
#   --calculator_graph_config_file=<.../object_detection_desktop_tflite_graph.pbtxt> \
#   --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>
cc_binary(
    name = "object_detection_gpu",
    data = [
        "//mediapipe/graphs/object_detection:mobile_gpu.binarypb",
        "//mediapipe/models:ssdlite_object_detection.tflite",
        "//mediapipe/models:ssdlite_object_detection_labelmap.txt",
    ],
    deps = [
        "//mediapipe/graphs/object_detection:mobile_calculators",
        "//mediapipe/examples/desktop:simple_run_graph_main",
    ],
)

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/object_detection:object_detection_cpu
# RUN:
# bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_cpu \
#   --calculator_graph_config_file=<.../object_detection_desktop_live.pbtxt> \
#   --input_video_path=<input video path> \
#   --output_video_path=<output video path>
cc_binary(
    name = "object_detection_cpu",
    data = [
        "//mediapipe/graphs/object_detection:mobile_cpu.binarypb",
        "//mediapipe/models:ssdlite_object_detection.tflite",
        "//mediapipe/models:ssdlite_object_detection_labelmap.txt",
    ],
    deps = [
        "//mediapipe/graphs/object_detection:mobile_calculators",
        "//mediapipe/examples/desktop:demo_run_graph_main",
    ],
)
