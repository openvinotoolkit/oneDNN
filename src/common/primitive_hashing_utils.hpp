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

#ifndef COMMON_PRIMITIVE_HASHING_UTILS_HPP
#define COMMON_PRIMITIVE_HASHING_UTILS_HPP

#include "common/primitive_hashing.hpp"

namespace dnnl {
namespace impl {
namespace primitive_hashing {

size_t get_post_op_hash(size_t seed, const post_ops_t &post_ops);

template <typename T, typename A>
size_t get_vector_hash(size_t seed, const std::vector<T, A> &vec) {
    return get_array_hash(seed, vec.data(), vec.size());
}

} // namespace primitive_hashing
} // namespace impl
} // namespace dnnl

#endif
