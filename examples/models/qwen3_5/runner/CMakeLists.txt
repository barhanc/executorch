# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(qwen3_5_runner)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

find_package(gflags REQUIRED)

# find `executorch` libraries
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(LINK_LIBS executorch gflags extension_llm_runner)
set(link_libraries ${LINK_LIBS})

# quantized_ops_lib: Register quantized op kernels into the runtime
executorch_target_link_options_shared_lib(quantized_ops_lib)
list(APPEND link_libraries portable_ops_lib portable_kernels quantized_kernels quantized_ops_lib)

# stb_image: a lightweight library to load images
include(FetchContent)
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
)
FetchContent_MakeAvailable(stb)
# Add deprecated/ to use stb_image_resize.h for internal compatibility
list(APPEND _common_include_directories ${stb_SOURCE_DIR}
     ${stb_SOURCE_DIR}/deprecated
)

add_executable(qwen3_5_runner main.cpp)
target_include_directories(qwen3_5_runner PUBLIC ${_common_include_directories})
target_link_libraries(qwen3_5_runner PUBLIC ${link_libraries})
target_compile_options(qwen3_5_runner PUBLIC ${_common_compile_options})
