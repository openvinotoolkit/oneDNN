/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "cpu/reorder/cpu_reorder.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// clang-format off

const impl_list_map_t &regular_f16_impl_list_map() {
    static const impl_list_map_t the_map = REG_REORDER_P({
        // f16 ->
        {{f16, data_type::undef, 0}, {
            DNNL_AARCH64_ONLY(REG_SR_DIRECT_COPY(f16, f16))

            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_brgemm_matmul_copy_reorder_t))
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64::jit_uni_reorder_direct_copy_t))
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_blk_reorder_t))
            DNNL_X64_ONLY(CPU_REORDER_INSTANCE(x64_jit_uni_reorder_t))

            DNNL_AARCH64_ONLY(CPU_REORDER_INSTANCE(aarch64::jit_uni_reorder_t))

            CPU_REORDER_INSTANCE(ref_reorder_t)
            nullptr,
        }},
    });
    return the_map;
}

// clang-format on

} // namespace cpu
} // namespace impl
} // namespace dnnl
