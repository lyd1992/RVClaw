cmake_minimum_required(VERSION 3.16)
project(rvclaw_yolov8_mjpeg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SPACEMIT_ROBOT_ROOT "/data/spacemit_robot" CACHE PATH "SpacemiT Robot SDK root")
set(VISION_ROOT "${SPACEMIT_ROBOT_ROOT}/components/model_zoo/vision")
set(VISION_STAGING "${SPACEMIT_ROBOT_ROOT}/output/staging")

find_package(OpenCV REQUIRED PATHS /opt/opencv-spacemit/lib/cmake/opencv4 NO_DEFAULT_PATH)
find_package(yaml-cpp REQUIRED)

add_executable(yolov8_mjpeg yolov8_mjpeg.cpp)
target_include_directories(yolov8_mjpeg PRIVATE "${VISION_ROOT}/include")
target_link_directories(yolov8_mjpeg PRIVATE "${VISION_STAGING}/lib")
target_link_libraries(yolov8_mjpeg PRIVATE
  vision
  ${OpenCV_LIBS}
  onnxruntime
  spacemit_ep
  yaml-cpp
)
set_target_properties(yolov8_mjpeg PROPERTIES
  BUILD_RPATH "${VISION_STAGING}/lib;/opt/opencv-spacemit/lib"
)
