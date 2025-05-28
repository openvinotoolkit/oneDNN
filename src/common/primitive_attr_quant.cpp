/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "common/primitive_attr_quant.hpp"
#include "common/primitive_hashing.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {

const quant_entry_t &default_quant_entry() {
    static const quant_entry_t default_quant_entry;
    return default_quant_entry;
}

size_t quant_entry_t::get_hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, mask_);
    seed = hash_combine(seed, static_cast<size_t>(data_type_));
    seed = hash_combine(seed, group_ndims_);
    if (group_ndims_ > 0)
        seed = primitive_hashing::get_array_hash(
                seed, group_dims_, group_ndims_);
    return seed;
}

void quant_entry_t::serialize(serialization_stream_t &sstream) const {
    sstream.append(mask_);
    sstream.append(data_type_);
    sstream.append_array(group_ndims_, group_dims_);
}

quant_entry_t quant_entry_t::deserialize(deserializer_t &d) {
    quant_entry_t e;
    d.pop(e.mask_);
    d.pop(e.data_type_);
    size_t group_ndims;
    d.pop_array(group_ndims, e.group_dims_);
    e.group_ndims_ = static_cast<int>(group_ndims);
    return e;
}

std::string quant_entry_t::get_verbose() const {
    std::string s;
    s.append(std::to_string(get_mask()));
    s.append(":").append(dnnl_dt2str(get_data_type()));
    s.append(":").append(std::to_string(type_));
    s.append(":");
    if (group_ndims_ > 0) {
                s.append(std::to_string(group_dims_[0]))
                .append("x")
                .append(std::to_string(group_dims_[1]));
    }
    s.append(":");
    if (get_ndims() > 0) {
            s.append(std::to_string(get_dims()[0]))
            .append("x")
            .append(std::to_string(get_dims()[1]));
    }

    return s;
}

std::ostream &operator<<(std::ostream &ss, const quant_entry_t &e) {
    ss << e.get_verbose();
    return ss;
}

size_t quant_entries_t::get_hash() const {
    size_t seed = 0;
    // Go through scales for all arguments.
    for (const auto &e : entries_) {
        seed = hash_combine(seed, e.first);
        seed = hash_combine(seed, e.second.get_hash());
    }
    return seed;
}

void quant_entries_t::serialize(serialization_stream_t &sstream) const {
    sstream.append(entries_.size());
    for (const auto &e : entries_) {
        sstream.append(e.first);
        sstream.append(e.second);
    }
}

template <typename T>
T deserialize_entries(deserializer_t &d) {
    T entries;
    size_t size = d.pop<size_t>();
    for (size_t i = 0; i < size; i++) {
        int arg = d.pop<int>();
        entries.set(arg, d.pop<quant_entry_t>());
    }
    return entries;
}

std::string quant_entries_t::get_verbose() const {
    std::string s;
    std::string empty_delim, attr_delim = "+";
    std::string delim = empty_delim;
    for (const auto &scale : entries_) {
        const auto &q = scale.second;
        if (q.has_default_values()) continue;

        int arg = scale.first;
        s.append(delim)
                .append(arg2str(arg))
                .append(":")
                .append(q.get_verbose());
        delim = attr_delim;
    }
    return s;
}

scales_t scales_t::deserialize(deserializer_t &d) {
    return deserialize_entries<scales_t>(d);
}

zero_points_t zero_points_t::deserialize(deserializer_t &d) {
    return deserialize_entries<zero_points_t>(d);
}

status_t quant_entry_t::set(int mask, data_type_t data_type, int group_ndims,
        const dims_t group_dims) {
    type_ = type_ | DNNL;
    is_set_ = true;
    mask_ = mask;
    data_type_ = data_type;
    group_ndims_ = group_ndims;
    if (group_ndims_ > 0) {
        utils::array_copy(group_dims_, group_dims, group_ndims_);
    }
    return status::success;
}

status_t quant_entry_t::set_scales(const dims_t dims, int ndims, data_type_t data_type, int mask) {
    type_ = type_ | OV_SCALES;
    is_set_scale = true;
    ndims_scale = ndims;
    mask_scale = mask;
    data_type_scale = data_type;
    if (ndims_scale > 0) {
        utils::array_copy(dims_scale, dims, ndims_scale);
    }
    return status::success;
}

status_t quant_entry_t::set_zero_points(const dims_t dims, int ndims, data_type_t data_type, int mask) {
    type_ = type_ | DNNL;
    is_set_wei = true;
    ndims_wei = ndims;
    mask_wei = mask;
    if (ndims_wei > 0) {
        utils::array_copy(dims_wei, dims, ndims_wei);
        group_ndims_ = ndims;
        utils::array_copy(group_dims_, dims, group_ndims_);
    }
    data_type_wei = data_type;
    return status::success;
}

status_t quant_entry_t::set_zero_points(const dims_t dims, int ndims, data_type_t data_type) {
    type_ = type_ | OV_ZERO_POINTS;
    is_set_wei = true;
    ndims_wei = ndims;
    mask_wei = 1;
    if (ndims_wei > 0) {
        utils::array_copy(dims_wei, dims, ndims_wei);
    }
    data_type_wei = data_type;
    return status::success;
}

status_t quant_entry_t::set(const quant_entry_t &other) {
    type_ = other.type_;
    is_set_ = other.is_set_;
    mask_ = other.mask_;
    data_type_ = other.data_type_;
    group_ndims_ = other.group_ndims_;
    if(group_ndims_ > 0)
        utils::array_copy(group_dims_, other.group_dims_, group_ndims_);
    is_set_scale = other.is_set_scale;
    mask_scale = other.mask_scale;
    data_type_scale = other.data_type_scale;
    ndims_scale = other.ndims_scale;
    if (ndims_scale > 0)
        utils::array_cmp(dims_scale, other.dims_scale, ndims_scale);
    is_set_wei = other.is_set_wei;
    mask_wei = other.mask_wei;
    data_type_wei = other.data_type_wei;
    ndims_wei = other.ndims_wei;
    if(ndims_wei > 0)
        utils::array_cmp(dims_wei, other.dims_wei, ndims_wei);
    return status::success;
}
int quant_entry_t::get_mask() const {
    if (is_set_wei) return mask_wei;
    if (is_set_) return mask_;
    if (is_set_scale) return mask_scale;
    return 0;
}
data_type_t quant_entry_t::get_data_type() const {
    if (is_set_wei) return data_type_wei;
    if (is_set_) return data_type_;
    if (is_set_scale) return data_type_scale;
    return data_type::undef;
}
const dims_t& quant_entry_t::get_dims() const {
    if (is_set_wei) return dims_wei;
    if (is_set_) return group_dims_;
    if (is_set_scale) return dims_scale;
    static const dims_t result = {};
    return result;
}

int quant_entry_t::get_ndims() const {
    if (is_set_wei) return ndims_wei;
    if (is_set_) return group_ndims_;
    if (is_set_scale) return ndims_scale;
    return 0;
}
// Note: keep the definition here to satisfy the
// `gtests/internals/test_comparison_operators` linking requirements which
// mandates bodies to be in the header file.
bool quant_entry_t::operator==(const quant_entry_t &rhs) const {
    bool result = (type_ == rhs.type_ && is_set_ == rhs.is_set_
            && mask_ == rhs.mask_
            && data_type_ == rhs.data_type_
            && group_ndims_ == rhs.group_ndims_
            && IMPLICATION(group_ndims_ > 0,
                utils::array_cmp(
                    group_dims_, rhs.group_dims_, group_ndims_)));

    if (!result) return false;
    result = (is_set_scale == rhs.is_set_scale
            && mask_scale == rhs.mask_scale
            && data_type_scale == rhs.data_type_scale
            && ndims_scale == rhs.ndims_scale
            && IMPLICATION(ndims_scale > 0,
                utils::array_cmp(
                    dims_scale, rhs.dims_scale, ndims_scale)));

    if (!result) return false;
    result = (is_set_wei == rhs.is_set_wei
            && mask_wei == rhs.mask_wei
            && data_type_wei == rhs.data_type_wei
            && ndims_wei == rhs.ndims_wei
            && IMPLICATION(ndims_wei > 0,
                utils::array_cmp(
                    dims_wei, rhs.dims_wei, ndims_wei)));
    return result;
}
status_t quant_entries_t::set_scales(int arg, const dims_t dims, int ndims, data_type_t data_type) {
    if (!check_arg(arg)) return status::invalid_arguments;
    CHECK(entries_[arg].set_scales(dims, ndims, data_type));
    return status::success;
}
status_t quant_entries_t::set_zero_points(int arg, const dims_t dims, int ndims, data_type_t data_type) {
    if (arg != DNNL_ARG_WEIGHTS) return status::unimplemented;
    CHECK(entries_[arg].set_zero_points(dims, ndims, data_type));
    return status::success;
}
status_t zero_points_t::set(int arg, int mask, data_type_t data_type, int group_ndims,
        const dims_t group_dims) {
    if (!check_arg(arg)) return status::invalid_arguments;
    if (arg == DNNL_ARG_WEIGHTS) {
        CHECK(entries_[arg].set_zero_points(group_dims, group_ndims, data_type, mask));
    } else {
        CHECK(entries_[arg].set(mask, data_type, group_ndims, group_dims));
    }
    return status::success;
}

} // namespace impl
} // namespace dnnl
