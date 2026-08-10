# *******************************************************************************
# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************

# RISC-V Cross-Compilation Toolchain File

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT DEFINED CMAKE_C_COMPILER)
    if(DEFINED ENV{CC} AND NOT "$ENV{CC}" STREQUAL "")
        set(CMAKE_C_COMPILER "$ENV{CC}")
    else()
        set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
    endif()
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    if(DEFINED ENV{CXX} AND NOT "$ENV{CXX}" STREQUAL "")
        set(CMAKE_CXX_COMPILER "$ENV{CXX}")
    else()
        set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
    endif()
endif()

set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_CROSSCOMPILING TRUE)
