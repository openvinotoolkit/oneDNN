/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include <algorithm>
#include "primitive_attr.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "dnnl_thread.hpp"
#include "engine.hpp"
#include "primitive_hashing_utils.hpp"

namespace dnnl {
namespace impl {
namespace primitive_hashing {

// Combine hash of each post_ops::entry_
size_t get_post_op_hash(size_t seed, const post_ops_t &post_ops) {
    for (int i = 0; i < post_ops.len(); i++) {
        const auto &entry = post_ops.entry_[i];
        switch (entry.kind) {
            case primitive_kind::eltwise:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.eltwise.alg));
                seed = hash_combine(seed, entry.eltwise.scale);
                seed = hash_combine(seed, entry.eltwise.alpha);
                seed = hash_combine(seed, entry.eltwise.beta);
                break;
            case primitive_kind::sum:
                seed = hash_combine(seed, entry.sum.scale);
                seed = hash_combine(seed, static_cast<size_t>(entry.sum.dt));
                break;
            case primitive_kind::convolution:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.depthwise_conv.stride));
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.depthwise_conv.wei_dt));
                seed = hash_combine(seed,
                        static_cast<size_t>(entry.depthwise_conv.bias_dt));
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.depthwise_conv.dst_dt));
                break;
            case primitive_kind::binary:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.binary.alg));
                seed = hash_combine(
                        seed, get_md_hash(entry.binary.user_src1_desc));
                break;
            case primitive_kind::prelu:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.prelu.mask));
                break;
            case primitive_kind::depthwise:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.depthwise.alg));
                seed = get_array_hash(seed, entry.depthwise.offset,
                        entry.depthwise.fields_count);
                break;
            case primitive_kind::quantization:
                seed = hash_combine(
                        seed, static_cast<size_t>(entry.quantization.alg));
                seed = get_array_hash(seed, entry.quantization.per_channel,
                        entry.quantization.fields_count);
                seed = get_array_hash(seed, entry.quantization.all_default,
                        entry.quantization.fields_count);
                seed = get_array_hash(seed, entry.quantization.offset,
                        entry.quantization.fields_count);
                break;
            default: assert(!"unknown post_op");
        }
    }

    return seed;
}

} // namespace primitive_hashing
} // namespace impl
} // namespace dnnl
