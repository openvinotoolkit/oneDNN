/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/kai_depthwise_convolution.hpp"
#include "cpu/aarch64/kai_utils.hpp"

#include <algorithm>
#include <memory>

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "kai/ops/conv/common.hpp"
#include "kai/ops/conv/depthwise.hpp"
#include "kai/ops/gemm/kai_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace data_type;
using namespace kai_utils;

namespace {

bool bias_ok(const cpu_convolution_fwd_pd_t &pd) {
    return !pd.with_bias()
            || pd.invariant_bia_md()->data_type == pd.dst_md()->data_type;
}

// aclnet's GroupConvolutions are per-channel depthwise: groups == input
// channels == output channels, with a single output channel per group
// (channel_multiplier == 1). Only that case is handled here; any other grouped
// convolution is declined so it falls back to a reference implementation.
bool is_supported_depthwise(const cpu_convolution_fwd_pd_t &pd) {
    if (!pd.with_groups()) return false;
    const dim_t g = pd.G();
    if (g <= 1) return false;
    if (pd.IC() != g || pd.OC() != g) return false;
    // OCg == ICg == 1  <=>  channel_multiplier == 1.
    return true;
}

bool depthwise_dt_ok(const cpu_convolution_fwd_pd_t &pd) {
    return pd.invariant_src_md()->data_type == f16
            && pd.invariant_wei_md()->data_type == f16
            && pd.invariant_dst_md()->data_type == f16;
}

} // namespace

bool kai_depthwise_convolution_fwd_t::pd_t::set_default_formats() {
    using namespace format_tag;
    // KleidiAI depthwise kernels consume NHWC activations and HWIO weights.
    // For a grouped weight tensor [G, 1, 1, KH, KW] the hwigo tag lays the
    // group (== channel) dimension out innermost with the kernel points in
    // row-major order, which is exactly the HWIO layout the packer expects.
    return set_default_formats_common(nhwc, hwigo, nhwc);
}

std::unique_ptr<kai::ops::depthwise::IDepthwiseCommon>
kai_depthwise_convolution_fwd_t::pd_t::create_kai_depthwise() const {
    const kai::ops::PaddingValues padding {static_cast<unsigned int>(padL()),
            static_cast<unsigned int>(padT()),
            static_cast<unsigned int>(padR()),
            static_cast<unsigned int>(padB())};

    kai::ops::depthwise::DepthwiseArgs args(get_cpu_info(),
            static_cast<unsigned int>(KH()), static_cast<unsigned int>(KW()),
            static_cast<unsigned int>(KSH()), static_cast<unsigned int>(KSW()),
            static_cast<unsigned int>(KDH() + 1),
            static_cast<unsigned int>(KDW() + 1),
            static_cast<unsigned int>(MB()), static_cast<unsigned int>(IH()),
            static_cast<unsigned int>(IW()), static_cast<unsigned int>(IC()),
            static_cast<unsigned int>(OH()), static_cast<unsigned int>(OW()),
            /* channel_multiplier */ 1, padding,
            make_kai_activation(activation_type_, activation_bound_),
            /* config */ nullptr);

    return kai::ops::depthwise::depthwise<__fp16>(args);
}

status_t kai_depthwise_convolution_fwd_t::pd_t::init(engine_t *engine) {
    using primitive_mask_t = primitive_attr_t::skip_mask_t;
    MAYBE_UNUSED(engine);

    // KleidiAI's depthwise epilogue can fuse a ReLU / bounded-ReLU activation,
    // so post-ops are allowed in the attribute check; kai_activation_from_post_ops
    // rejects any post-op chain that cannot be represented, forcing a reference
    // fallback for those cases.
    bool ok = is_fwd() && set_default_alg_kind(alg_kind::convolution_direct)
            && ndims() == 4 && is_supported_depthwise(*this)
            && depthwise_dt_ok(*this) && !has_zero_dim_memory()
            && !has_runtime_dims_or_strides()
            && attr()->has_default_values(primitive_mask_t::fpmath_mode
                            | primitive_mask_t::accumulation_mode
                            | primitive_mask_t::post_ops,
                    dst_md()->data_type)
            && kai_activation_from_post_ops(
                    attr(), activation_type_, activation_bound_)
            && set_default_formats()
            && attr_.set_default_formats(dst_md()) == status::success;
    if (!ok) return status::unimplemented;

    VDISPATCH_CONV(bias_ok(*this), VERBOSE_UNSUPPORTED_DT_CFG);

    // Verify the resolved formats match what the kernel requires: NHWC
    // activations and hwigo (HWIO) weights.
    const auto src_tag
            = memory_desc_matches_one_of_tag(src_md_, format_tag::nhwc);
    const auto dst_tag
            = memory_desc_matches_one_of_tag(dst_md_, format_tag::nhwc);
    const auto wei_tag
            = memory_desc_matches_one_of_tag(weights_md_, format_tag::hwigo);
    VDISPATCH_CONV(
            src_tag != format_tag::undef, VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_CONV(
            dst_tag != format_tag::undef, VERBOSE_UNSUPPORTED_TAG_S, "dst");
    VDISPATCH_CONV(
            wei_tag != format_tag::undef, VERBOSE_UNSUPPORTED_TAG_S, "weights");

    std::unique_ptr<kai::ops::depthwise::IDepthwiseCommon> kernel
            = create_kai_depthwise();
    VDISPATCH_CONV(kernel != nullptr, VERBOSE_UNSUPPORTED_DT_CFG);

    nthr_ = std::max(1, dnnl_get_current_num_threads());
    packed_weights_size_ = kernel->get_storage_size();
    working_size_ = kernel->get_working_size(static_cast<unsigned int>(nthr_));

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_conv_permuted_weights,
            packed_weights_size_, 1, 64, 64);
    if (working_size_ != 0) {
        scratchpad.book(memory_tracking::names::key_gemm_asm_tmp_buffer,
                working_size_, 1, 64, 64);
    }

    return status::success;
}

status_t kai_depthwise_convolution_fwd_t::init(engine_t *engine) {
    MAYBE_UNUSED(engine);
    return status::success;
}

status_t kai_depthwise_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    std::unique_ptr<kai::ops::depthwise::IDepthwiseCommon> kernel
            = pd()->create_kai_depthwise();
    if (!kernel) return status::runtime_error;

    const auto scratchpad = ctx.get_scratchpad_grantor();

    const auto *src_base = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const auto *raw_wei = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    void *dst_base = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const void *bias_base = pd()->with_bias()
            ? CTX_IN_MEM(const void *, DNNL_ARG_BIAS)
            : nullptr;

    void *packed_weights = scratchpad.get<void>(
            memory_tracking::names::key_conv_permuted_weights);
    void *working_space = pd()->working_size_ != 0
            ? scratchpad.get<void>(
                      memory_tracking::names::key_gemm_asm_tmp_buffer)
            : nullptr;

    // Pack weights (and bias) once into the scratchpad. The weights are already
    // laid out as HWIO (hwigo) with the channel dimension innermost, so the
    // default kernel/row column strides resolve to the correct values.
    kernel->pack_parameters(packed_weights, bias_base, raw_wei);

    // NHWC layout strides (in elements) as resolved by set_default_formats.
    constexpr int n_dim = 0;
    constexpr int h_dim = 2;
    constexpr int w_dim = 3;
    const auto &src_strides = pd()->src_md()->format_desc.blocking.strides;
    const auto &dst_strides = pd()->dst_md()->format_desc.blocking.strides;
    const size_t ld_input_col = static_cast<size_t>(src_strides[w_dim]);
    const size_t ld_input_row = static_cast<size_t>(src_strides[h_dim]);
    const size_t ld_input_batch = static_cast<size_t>(src_strides[n_dim]);
    const size_t ld_output_col = static_cast<size_t>(dst_strides[w_dim]);
    const size_t ld_output_row = static_cast<size_t>(dst_strides[h_dim]);
    const size_t ld_output_batch = static_cast<size_t>(dst_strides[n_dim]);

    // Cap at the thread count the working space was sized for in init(); the
    // KleidiAI driver stripes a per-thread slice of that buffer by thread id.
    const int num_threads = std::min(
            pd()->nthr_, std::max(1, dnnl_get_current_num_threads()));
    parallel(num_threads, [&](int ithr, int nthr) {
        kernel->execute(src_base, ld_input_col, ld_input_row, ld_input_batch,
                packed_weights, dst_base, ld_output_col, ld_output_row,
                ld_output_batch, working_space, static_cast<unsigned int>(ithr),
                static_cast<unsigned int>(nthr));
    });

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
