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

#ifndef COMMON_PRIMITIVE_ATTR_QUANT_HPP
#define COMMON_PRIMITIVE_ATTR_QUANT_HPP

// NOTE: Objects declared in this header are moved out from primitive_attr.hpp due
// to micro_sdpa primitive. Int8 support requires at least two primitive_attr
// objects to be used inside sdpa_desc_t object which triggers a deleted
// copy-ctor of primitive_attr_t, which is there because of RNN scales still
// rely on static scales and manage dynamically-allocated memory.
//
// As a result, micro_sdpa uses scales and zero-points objects directly and
// requires a dedicated header for that, otherwise, it's going to be a circular
// dependency between headers when it comes to inclusion of opdesc.hpp which
// sdpa_desc_t is a part of.

#include "common/serialization.hpp"
#include "common/utils.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>

namespace dnnl {
namespace impl {

struct quant_entry_t;
const quant_entry_t &default_quant_entry();

struct quant_entry_t : public c_compatible {
    quant_entry_t() = default;

    // `set(...)` approach is taken over constructors as the usage model assumes
    // the change of state of this object but it doesn't require its destruction
    // which would come with some performance price which prevails in this case.
    status_t set(int mask, data_type_t data_type) {
        return set(mask, data_type, 0, {});
    }
    status_t set(int mask, data_type_t data_type, int group_ndims,
            const dims_t group_dims);
    status_t set_scales(const dims_t dims, int ndims, data_type_t data_type = data_type::f32, int mask = 1);
    status_t set_zero_points(const dims_t dims, int ndims, data_type_t data_type);
    status_t set_zero_points(const dims_t dims, int ndims, data_type_t data_type, int mask);
    status_t set(const quant_entry_t &other);
    quant_entry_t &operator=(const quant_entry_t &rhs) {
        auto st = this->set(rhs);
        assert(st == status::success);
        UNUSED(st);
        return *this;
    }
    bool has_default_values() const { return *this == default_quant_entry(); }
    bool has_default_groups() const {
        return this->group_ndims_ == default_quant_entry().group_ndims_;
    }
    int get_mask() const;
    data_type_t get_data_type() const;
    const dims_t& get_dims() const;
    int get_ndims() const;
    dim_t get_group(int d) const {
        // If groups were not requested, return `1` for convenience.
        if (group_ndims_ == default_quant_entry().group_ndims_) return 1;
        // But if they were, any out of bound access would return `0` and likely
        // lead to a division by zero which is fast to catch.
        if (d >= group_ndims_) return 0;
        return group_dims_[d];
    }

    // Note: keep the definition here to satisfy the
    // `gtests/internals/test_comparison_operators` linking requirements which
    // mandates bodies to be in the header file.
    bool operator==(const quant_entry_t &rhs) const;
    size_t get_hash() const;

    void serialize(serialization_stream_t &sstream) const;

    static quant_entry_t deserialize(deserializer_t &d);

    std::string get_verbose() const;

private:
    data_type_t data_type_ = data_type::undef;
    int group_ndims_ = 0;
    dims_t group_dims_ {};
    // Note: INT_MIN is used on purpose to avoid potential issues when
    // `(mask & bit)` expression will return `true`. `INT_MIN` is represented
    // as `10...0` in bits and will avoid such situations.
    int mask_ = INT_MIN;
    bool is_set_ = false;
    // openvino extension
    enum entry_type {
        NONE = 0,
        DNNL = 1,
        OV_SCALES = 2,
        OV_ZERO_POINTS = 4
    };
    int type_ = NONE;
    // scale
    bool is_set_scale = false;
    int ndims_scale = 0;
    int mask_scale = 0;
    dims_t dims_scale {};
    data_type_t data_type_scale = data_type::undef;
    // zero_point
    bool is_set_wei = false;
    int ndims_wei = 0;
    int mask_wei = 0;
    dims_t dims_wei {};
    data_type_t data_type_wei = data_type::s32;
};

std::ostream &operator<<(std::ostream &ss, const quant_entry_t &e);

struct quant_entries_t : public c_compatible {
    quant_entries_t(data_type_t default_data_type)
        : default_data_type_(default_data_type) {}

    const quant_entry_t &get(int arg) const {
        const auto it = entries_.find(arg);
        if (it == entries_.end()) return default_quant_entry();
        return it->second;
    }

    // See `set(...)` comment for `quant_entry_t` for a design choice
    // explanation.
    virtual status_t set(int arg, int mask) {
        return set(arg, mask, default_data_type_, 0, {});
    }
    const dims_t & get_dims(int arg) const {
        return get(arg).get_dims();
    }
    int get_ndims(int arg) const {
        return get(arg).get_ndims();
    }
    virtual status_t set(int arg, int mask, data_type_t data_type, int group_ndims,
            const dims_t group_dims) {
        if (!check_arg(arg)) return status::invalid_arguments;
        CHECK(entries_[arg].set(mask, data_type, group_ndims, group_dims));
        return status::success;
    }
    status_t set_scales(int arg, const dims_t dims, int ndims, data_type_t data_type = data_type::f32);
    status_t set_zero_points(int arg, const dims_t dims, int ndims, data_type_t data_type);

    // Use this interface with `default_quant_entry` when need to remove a
    // specific entry.
    virtual status_t set(int arg, const quant_entry_t &other) {
        return entries_[arg].set(other);
    }

    // This interface is different from the one below and is just a shortcut.
    bool has_default_values(int arg) const {
        return get(arg).has_default_values();
    }

    // This interface is used to make sure that other than `supported_args` have
    // default values. It's to make sure that non-allowed arguments were not
    // passed to the library.
    bool has_default_values(const std::vector<int> &supported_args = {}) const {
        auto predicate
                = [](const quant_entry_t &s) { return s.has_default_values(); };
        return has_default_property(supported_args, predicate);
    }

    // This interface checks specific argument. It exists because quant_entry_t
    // doesn't have a notion of default data_type, only this object does.
    // Note: can be removed once the library unconditionally supports data type
    // for scales/zero-points for every implementation, then this call can be
    // removed as to make a proper load, the data type must be queried.
    bool has_default_data_type(int arg) const {
        // Note: `data_type::undef` represents `default_quant_entry`.
        return utils::one_of(
                get(arg).get_data_type(), default_data_type_, data_type::undef);
    }

    // This interface is different from the one below and is just a shortcut.
    bool has_default_groups(int arg) const {
        return get(arg).has_default_groups();
    }

    // This interface is used to make sure that other than `supported_args` have
    // default values. It's to make sure that non-allowed arguments were not
    // passed to the library.
    bool has_default_groups(const std::vector<int> &supported_args = {}) const {
        auto predicate
                = [](const quant_entry_t &s) { return s.has_default_groups(); };
        return has_default_property(supported_args, predicate);
    }

    int get_mask(int arg) const { return get(arg).get_mask(); }
    data_type_t get_data_type(int arg) const {
        return get(arg).get_data_type();
    }
    dim_t get_group(int arg, int d) const { return get(arg).get_group(d); }

    bool operator==(const quant_entries_t &rhs) const {
        return entries_ == rhs.entries_;
    }

    size_t get_hash() const;

    void serialize(serialization_stream_t &sstream) const;

    std::string get_verbose() const;

protected:
    // Sorted property of `std::map` is used for hashing.
    std::map<int, quant_entry_t> entries_;
    // Value is different depending on the inheritor.
    data_type_t default_data_type_ = data_type::undef;

    virtual bool check_arg(int arg) const = 0;

    // The function makes sure that if any argument was specified by user, that
    // only `supported_args` have their value customized, rest unsupported
    // values were not updated.
    bool has_default_property(const std::vector<int> &supported_args,
            bool (*predicate)(const quant_entry_t &)) const {
        for (const auto &s : entries_) {
            // Arg passed the condition, check the next one.
            if (predicate(s.second)) continue;

            bool allow_non_default = false;
            for (const auto &supported_arg : supported_args)
                if (s.first == supported_arg) {
                    allow_non_default = true;
                    break;
                }
            if (allow_non_default) continue;
            return false;
        }
        return true;
    }
};

struct scales_t : public quant_entries_t {
    scales_t() : quant_entries_t(default_data_type_) {};

    // This interface checks the content of all entries, and allows to ignore
    // certain arguments.
    // Note: can't be put in `quant_entries_t` because `default_data_type_` is
    // not a static member, but `has_default_property` requires `predicate`
    // to have it this way.
    bool has_default_data_type(
            const std::vector<int> &supported_args = {}) const {
        auto predicate = [](const quant_entry_t &s) {
            // Note: `data_type::undef` represents `default_quant_entry`.
            return utils::one_of(
                    s.get_data_type(), default_data_type_, data_type::undef);
        };
        return has_default_property(supported_args, predicate);
    }
    // Note: must present as compiler doesn't see an overloaded version inside a
    // base class.
    bool has_default_data_type(int arg) const {
        return quant_entries_t::has_default_data_type(arg);
    }

    static scales_t deserialize(deserializer_t &d);

private:
    static constexpr data_type_t default_data_type_ = data_type::f32;

    bool check_arg(int arg) const override {
        // regular
        for (const auto &sa : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (arg == sa) return true;
        }
        // binary
        for (const auto &sa : {DNNL_ARG_SRC_1}) {
            if (arg == sa) return true;
        }
        // concat
        if (arg & DNNL_ARG_MULTIPLE_SRC) return true;
        // depth-wise convolution post op
        for (const auto &sa : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (arg == (DNNL_ARG_ATTR_POST_OP_DW | sa)) return true;
        }
        // sdpa
        if (arg == DNNL_ARG_SRC_2) return true;
        return false;
    }
};

struct zero_points_t : public quant_entries_t {
    zero_points_t() : quant_entries_t(default_data_type_) {};

    // This interface checks the content of all entries, and allows to ignore
    // certain arguments.
    // Note: can't be put in `quant_entries_t` because `default_data_type_` is
    // not a static member, but `has_default_property` requires `predicate`
    // to have it this way.
    bool has_default_data_type(
            const std::vector<int> &supported_args = {}) const {
        auto predicate = [](const quant_entry_t &s) {
            // Note: `data_type::undef` represents `default_quant_entry`.
            return utils::one_of(
                    s.get_data_type(), default_data_type_, data_type::undef);
        };
        return has_default_property(supported_args, predicate);
    }
    // Note: must present as compiler doesn't see an overloaded version inside a
    // base class.
    bool has_default_data_type(int arg) const {
        return quant_entries_t::has_default_data_type(arg);
    }

    static zero_points_t deserialize(deserializer_t &d);
    status_t set(int arg, int mask) {
        return quant_entries_t::set(arg, mask, default_data_type_, 0, {});
    }
    status_t set(int arg, int mask, data_type_t data_type, int group_ndims,
            const dims_t group_dims);

    status_t set(int arg, const quant_entry_t &other) {
        return quant_entries_t::set(arg, other);
    }
private:
    static constexpr data_type_t default_data_type_ = data_type::s32;

    bool check_arg(int arg) const override {
        // regular
        // gemm internal primitive would use DNNL_ARG_A, DNNL_ARG_B, DNNL_ARG_C,
        // which match to DNNL_ARG_WEIGHTS, DNNL_ARG_SRC, DNNL_ARG_DST. They
        // are defined in gpu internals, thus, not spelled here.
        for (const auto &sa : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (arg == sa) return true;
        }
        // sdpa
        if (arg == DNNL_ARG_SRC_2) return true;
        return false;
    }
};

struct src_dyn_quant_params_t : public c_compatible {
    src_dyn_quant_params_t() : group_size_(0) {}
    bool has_default_values() const {
        return (group_size_ == 0);
    }
    bool defined() const {
        return true;
    }

    status_t set(uint64_t group_size) {
        group_size_ = group_size;
        return status::success;
    }

    uint64_t get() const {
        return group_size_;
    }

    bool operator==(const src_dyn_quant_params_t &rhs) const {
        using namespace utils;
        return group_size_ == rhs.group_size_;
    }

private:
    uint64_t group_size_;
};

} // namespace impl
} // namespace dnnl

#endif
