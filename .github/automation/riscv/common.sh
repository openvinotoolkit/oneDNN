#! /bin/bash

# *******************************************************************************
# Copyright 2024-2025 Arm Limited and affiliates.
# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
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

# Common variables for RISC-V CI. Exports:
# CC, CXX, OS, CMAKE_TOOLCHAIN_FILE

set -o errexit -o pipefail -o noclobber

export OS=$(uname)

if [[ "$OS" == "Linux" ]]; then
    export MP="-j$(nproc)"
fi

if [[ "$BUILD_TOOLSET" == "gcc" ]]; then
    export CC=riscv64-linux-gnu-gcc-${GCC_VERSION}
    export CXX=riscv64-linux-gnu-g++-${GCC_VERSION}
fi

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
export CMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/../../../cmake/toolchains/riscv64.cmake"

echo "Cross-compilation mode for RISC-V"
echo "OS: $OS"
echo "CC: $CC"
echo "CXX: $CXX"
echo "CMAKE_TOOLCHAIN_FILE: $CMAKE_TOOLCHAIN_FILE"
