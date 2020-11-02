cmake_minimum_required(VERSION 3.10)
if (NOT DEFINED PROJECT_NAME)
    project(EAST)
endif()

set(CMAKE_CXX_STANDARD 11)
message(\n ---${CMAKE_CURRENT_SOURCE_DIR}---)

set(EAST_ALL_INCLUDE_DIRS "")
# opencv相关
find_package(OpenCV REQUIRED)
list(APPEND EAST_ALL_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})

# TensorFlow 相关
# 不搞这个Eigen3，会报错
if (NOT DEFINED Eigen3_INCLUDE_DIR)
    set(Eigen3_INCLUDE_DIR /usr/local/include/eigen3)
endif ()
list(APPEND EAST_ALL_INCLUDE_DIRS ${Eigen3_INCLUDE_DIR})

if (NOT DEFINED TF_ROOT_DIR)
    # 这个路径是本工程本地测试路径
    set(TF_ROOT_DIR /usr/local/tensorflow)
endif ()
message(TF_ROOT_DIR: ${TF_ROOT_DIR})
list(APPEND EAST_ALL_INCLUDE_DIRS
     ${TF_ROOT_DIR}
     ${TF_ROOT_DIR}/tensorflow
     ${TF_ROOT_DIR}/tensorflow/contrib/makefile/downloads/absl
     ${TF_ROOT_DIR}/tensorflow/contrib/makefile/downloads
     ${TF_ROOT_DIR}/bazel-genfiles
     ${TF_ROOT_DIR}/third-party
     ${TF_ROOT_DIR}/tensorflow/contrib/makefile/gen/protobuf/include
     ${TF_ROOT_DIR}/tensorflow/contrib/makefile/downloads/protobuf/src
     ${TF_ROOT_DIR}/tensorflow/contrib/makefile/downloads/protobuf/src/include
     )
link_directories(${TF_ROOT_DIR}/bazel-bin/tensorflow)
set(TF_LIBS tensorflow_cc tensorflow_framework)


# project相关
include_directories(${EAST_ALL_INCLUDE_DIRS})
if (NOT DEFINED EAST_LIBRARY_NAME)
    set(EAST_LIBRARY_NAME east)
endif ()

add_subdirectory(src)
if (${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(test)
endif ()
