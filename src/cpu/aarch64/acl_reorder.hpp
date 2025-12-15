/*******************************************************************************
* Copyright 2025 Arm Ltd. and affiliates
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_AARCH64_ACL_REORDER_HPP
#define CPU_AARCH64_ACL_REORDER_HPP

// Keep include path compatibility with code that expects this header.
#include "cpu/aarch64/reorder/acl_reorder.hpp"

// Provide the expected cpu::acl namespace alias used by common headers.
namespace dnnl {
namespace impl {
namespace cpu {
namespace acl {
using aarch64::acl_reorder_fwd_t;
} // namespace acl
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
