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

#ifndef CPU_AARCH64_KAI_DEPTHWISE_CONVOLUTION_HPP
#define CPU_AARCH64_KAI_DEPTHWISE_CONVOLUTION_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_convolution_pd.hpp"

namespace kai::ops::depthwise {
class IDepthwiseCommon;
} // namespace kai::ops::depthwise

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// KleidiAI-backed depthwise (per-channel grouped) convolution for f16 on
// AArch64. This covers the common depthwise case (groups == input channels ==
// output channels, channel_multiplier == 1) which is what OpenVINO emits for a
// GroupConvolution with per-channel weights. Non-depthwise grouped convolutions
// are declined in init() so they fall back to a reference implementation.
struct kai_depthwise_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("depthwise:kleidiai",
                kai_depthwise_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine);
        std::unique_ptr<kai::ops::depthwise::IDepthwiseCommon>
        create_kai_depthwise() const;

        // Fused activation extracted from the attribute post-ops. Stored as
        // plain data (0 = none, 1 = ReLU, 2 = bounded ReLU) so this header does
        // not need to pull in the KleidiAI headers.
        int activation_type_ = 0;
        float activation_bound_ = 0.0f;
        // Packed-weights buffer size and total working-space size, as reported
        // by the selected KleidiAI kernel during init(). working_size_ scales
        // with nthr_ (per-thread striping), so execute() must not use more than
        // nthr_ threads or it would run past the booked working space.
        size_t packed_weights_size_ = 0;
        size_t working_size_ = 0;
        int nthr_ = 1;

    private:
        bool set_default_formats();
    };

    kai_depthwise_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
