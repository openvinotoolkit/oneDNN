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

#ifndef CPU_REORDER_CPU_REORDER_HPP
#define CPU_REORDER_CPU_REORDER_HPP

#include <map>
#include <vector>

#include "cpu/reorder/simple_reorder.hpp"

#include "common/memory.hpp"
#include "common/type_helpers.hpp"
#include "common/dnnl_sel_build.hpp"

#include "cpu/cpu_engine.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_reorder.hpp"
#include "cpu/x64/wino_reorder.hpp"
#elif DNNL_AARCH64
#include "cpu/aarch64/jit_uni_reorder.hpp"
#endif

#include "cpu/rnn/rnn_reorders.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using rpd_create_f = dnnl::impl::engine_t::reorder_primitive_desc_create_f;

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using spec_reference = spec::reference;
using spec_direct_copy = spec::direct_copy;
using spec_direct_copy_except_dim_0 = spec::direct_copy_except_dim_0;
using spec_conv_conv_req_comp = spec::conv_req_comp;
template <data_type_t type_i, data_type_t type_o>
using x64_wino_reorder_t = x64::wino_reorder_t<type_i, type_o>;
constexpr bool fmt_order_keep = fmt_order::keep;
constexpr bool fmt_order_reverse = fmt_order::reverse;
constexpr bool fmt_order_any = fmt_order::any;

struct uni_reorder_t {
    struct pd_t {
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
                    return x64::jit_uni_reorder_create(reorder_pd, engine, attr, src_engine, src_md, dst_engine, dst_md);
                }
    };
};

#if defined(SELECTIVE_BUILD_ANALYZER)

DNNL_DEF_PD_BUILDER(rpd_builder,
            rpd_create_f,
            dnnl::impl::reorder_pd_t **,
            dnnl::impl::engine_t *,
            const dnnl::impl::primitive_attr_t *,
            dnnl::impl::engine_t *,
            const dnnl::impl::memory_desc_t *,
            dnnl::impl::engine_t *,
            const dnnl::impl::memory_desc_t *);

# define REG_REORDER_FN(...) REG_DNNL_FN(rpd_builder, __VA_ARGS__)

# define REG_SR_5(idt, ifmt, odt, ofmt, order) REG_DNNL_FN_7(rpd_builder, simple_reorder_t, idt, ifmt, odt, ofmt, order)
# define REG_SR_6(idt, ifmt, odt, ofmt, order, spec) REG_DNNL_FN_8(rpd_builder, simple_reorder_t, idt, ifmt, odt, ofmt, order, spec)

#else   // SELECTIVE_BUILD == ON || SELECTIVE_BUILD == OFF

# define REG_REORDER_FN REG_DNNL_FN

# define REG_SR_5(idt, ifmt, odt, ofmt, order) REG_DNNL_FN_6(simple_reorder_t, idt, ifmt, odt, ofmt, order)
# define REG_SR_6(idt, ifmt, odt, ofmt, order, spec) REG_DNNL_FN_7(simple_reorder_t, idt, ifmt, odt, ofmt, order, spec)

#endif

struct reorder_impl_key_t {
    data_type_t src_dt;
    data_type_t dst_dt; // data_type::undef if arbitrary
    int ndims; // 0 if arbitrary

    bool operator<(const reorder_impl_key_t &rhs) const {
        return value() < rhs.value();
    }

private:
    enum { MAX_DT_NUM = 10 };
    size_t value() const {
        return ((size_t)ndims * MAX_DT_NUM + (size_t)src_dt) * MAX_DT_NUM
                + (size_t)dst_dt;
    }
};

using impl_list_map_t = std::map<reorder_impl_key_t, std::vector<rpd_create_f>>;

/* regular reorders */
extern const impl_list_map_t regular_f32_bf16_impl_list_map;
extern const impl_list_map_t regular_f32_f16_impl_list_map;
extern const impl_list_map_t regular_f32_f32_impl_list_map;
extern const impl_list_map_t regular_f32_s32_impl_list_map;
extern const impl_list_map_t regular_f32_s8_impl_list_map;
extern const impl_list_map_t regular_f32_u8_impl_list_map;
extern const impl_list_map_t regular_f32_bin_impl_list_map;
extern const impl_list_map_t regular_bf16_impl_list_map;
extern const impl_list_map_t regular_f16_impl_list_map;
extern const impl_list_map_t regular_s32_impl_list_map;
extern const impl_list_map_t regular_s8_impl_list_map;
extern const impl_list_map_t regular_u8_impl_list_map;
extern const impl_list_map_t regular_bin_impl_list_map;

/* conv reorders w/ compensation */
extern const impl_list_map_t comp_f32_s8_impl_list_map;
extern const impl_list_map_t comp_bf16_s8_impl_list_map;
extern const impl_list_map_t comp_s8_s8_impl_list_map;

#define REG_SR(...) DNNL_MACRO_OVERLOAD(REG_SR, __VA_ARGS__)

#define REG_SR_BIDIR(idt, ifmt, odt, ofmt) \
    REG_SR(idt, ifmt, odt, ofmt, fmt_order_keep) \
            REG_SR(idt, ifmt, odt, ofmt, fmt_order_reverse)

#define REG_SR_DIRECT_COPY(idt, odt) \
    REG_SR(idt, any, odt, any, fmt_order_any, spec_direct_copy) \
            REG_SR(idt, any, odt, any, fmt_order_any, \
                    spec_direct_copy_except_dim_0)

#if defined(__INTEL_COMPILER) || (defined(__GNUC__) && !defined(__clang__))
/* Direct copy for icc which is faster than jitted code;
 * Direct copy for gcc which might or might not be faster than jitted
 * code, but still worth it because doesn't require jitting, i.e. much
 * faster creation time. This is tentative solution and should be
 * removed later (when we will cache jitted code?...). */
#define REG_FAST_DIRECT_COPY_F32_F32_COMMA REG_SR_DIRECT_COPY(f32, f32)
#else
#define REG_FAST_DIRECT_COPY_F32_F32_COMMA
#endif

#ifdef __INTEL_COMPILER
/* direct copy for icc, which is faster than jitted code */
#define REG_FAST_DIRECT_COPY_COMMA(sdt, ddt) REG_SR_DIRECT_COPY(sdt, ddt),
#else
#define REG_FAST_DIRECT_COPY_COMMA(sdt, ddt)
#endif

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
