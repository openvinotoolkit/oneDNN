/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "test_convolution_forward_common.hpp"
namespace dnnl {

using convolution_test
        = convolution_forward_test_t<uint8_t, int8_t, int32_t, int32_t>;

TEST_P(convolution_test, TestConvolution) {}

TEST(convolution_gemm_post_ops_test, GroupedInt8Depthwise) {
    SKIP_IF(get_test_engine_kind() != engine::kind::cpu,
            "The test requires host-accessible memory.");
    auto eng = get_test_engine();
    auto strm = make_stream(eng);
    const memory::dims src_dims {1, 4, 2, 4};
    const memory::dims strides {1, 1}, padding {0, 0};
    const memory::desc src_md(
            src_dims, memory::data_type::u8, memory::format_tag::nhwc);
    std::vector<uint8_t> src(32, 2);
    for (int group_channels : {16, 17}) {
        const int channels = 2 * group_channels;
        const memory::dims weights_dims {2, group_channels, 2, 1, 1};
        const memory::dims dst_dims {1, channels, 2, 4};
        const memory::desc weights_md(
                weights_dims, memory::data_type::s8, memory::format_tag::hwigo);
        const memory::desc dst_md(
                dst_dims, memory::data_type::s32, memory::format_tag::nhwc);
        std::vector<int8_t> weights(2 * channels, -1);
        std::vector<int32_t> dst(8 * channels, 1);
        std::vector<float> scales(channels, 1.f), shifts(channels);
        for (int channel = 0; channel < channels; ++channel)
            shifts[channel] = static_cast<float>(channel);
        std::vector<float> depthwise_data(scales);
        depthwise_data.insert(depthwise_data.end(), shifts.begin(), shifts.end());
        const memory::desc depthwise_md(
                {2 * channels}, memory::data_type::f32, memory::format_tag::x);
        memory depthwise_mem(depthwise_md, eng, depthwise_data.data());
        post_ops ops;
        ops.append_sum(1.f);
        ops.append_depthwise(algorithm::depthwise_scale_shift, {0, scales.size()});
        ops.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
        primitive_attr attr;
        attr.set_post_ops(ops);
        convolution_forward::primitive_desc pd(eng, prop_kind::forward_inference,
                algorithm::convolution_direct, src_md, weights_md, dst_md,
                strides, padding, padding, attr);
        while (std::string(pd.impl_info_str()).find(":gemm") == std::string::npos
                && pd.next_impl()) {}
        ASSERT_NE(std::string(pd.impl_info_str()).find(":gemm"), std::string::npos);
        memory src_mem(src_md, eng, src.data());
        memory weights_mem(weights_md, eng, weights.data());
        memory dst_mem(dst_md, eng, dst.data());
        convolution_forward(pd).execute(strm,
                {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weights_mem},
                        {DNNL_ARG_DST, dst_mem},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                depthwise_mem}});
        strm.wait();
        for (int spatial = 0; spatial < 8; ++spatial)
            for (int channel = 0; channel < channels; ++channel)
                EXPECT_EQ(dst[spatial * channels + channel], std::max(0, channel - 3));
    }
}

TEST(convolution_gemm_post_ops_test, Int8DepthwiseWithPaddingAndZeroPoint) {
    SKIP_IF(get_test_engine_kind() != engine::kind::cpu,
            "The test requires host-accessible memory.");
    auto eng = get_test_engine();
    auto strm = make_stream(eng);
    const int channels = 32;
    const memory::dims src_dims {1, 4, 2, 4};
    const memory::dims weights_dims {2, 16, 2, 3, 3};
    const memory::dims dst_dims {1, channels, 2, 4};
    const memory::dims strides {1, 1}, padding {1, 1};
    const memory::desc src_md(
            src_dims, memory::data_type::u8, memory::format_tag::nhwc);
    const memory::desc weights_md(
            weights_dims, memory::data_type::s8, memory::format_tag::hwigo);
    const memory::desc weights_any(
            weights_dims, memory::data_type::s8, memory::format_tag::any);
    const memory::desc dst_md(
            dst_dims, memory::data_type::s32, memory::format_tag::nhwc);
    const memory::desc zp_md({1}, memory::data_type::s32, memory::format_tag::x);
    std::vector<uint8_t> src(32, 2);
    std::vector<int8_t> weights(2 * channels * 9, -1);
    std::vector<int32_t> dst(8 * channels);
    std::vector<float> scales(channels, 1.f), shifts(channels);
    for (int channel = 0; channel < channels; ++channel)
        shifts[channel] = static_cast<float>(channel);
        std::vector<float> depthwise_data(scales);
        depthwise_data.insert(depthwise_data.end(), shifts.begin(), shifts.end());
        const memory::desc depthwise_md(
                        {2 * channels}, memory::data_type::f32, memory::format_tag::x);
        memory depthwise_mem(depthwise_md, eng, depthwise_data.data());
    post_ops ops;
        ops.append_depthwise(algorithm::depthwise_scale_shift, {0, scales.size()});
    primitive_attr attr;
    attr.set_post_ops(ops);
    attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
    convolution_forward::primitive_desc pd(eng, prop_kind::forward_inference,
            algorithm::convolution_direct, src_md, weights_any, dst_md, strides,
            padding, padding, attr);
    while (std::string(pd.impl_info_str()).find(":gemm") == std::string::npos
            && pd.next_impl()) {}
    ASSERT_NE(std::string(pd.impl_info_str()).find(":gemm"), std::string::npos);
    memory src_mem(src_md, eng, src.data());
    memory plain_weights_mem(weights_md, eng, weights.data());
    memory weights_mem(pd.weights_desc(), eng);
    reorder(plain_weights_mem, weights_mem)
            .execute(strm, plain_weights_mem, weights_mem);
    memory dst_mem(dst_md, eng, dst.data());
    int32_t zero_point = 1;
    memory zp_mem(zp_md, eng, &zero_point);
    convolution_forward(pd).execute(strm,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weights_mem},
                    {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, zp_mem},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                            depthwise_mem}});
    strm.wait();
    for (int spatial = 0; spatial < 8; ++spatial) {
        const int valid = spatial % 4 == 0 || spatial % 4 == 3 ? 4 : 6;
        for (int channel = 0; channel < channels; ++channel)
            EXPECT_EQ(dst[spatial * channels + channel], channel - 2 * valid);
    }
}

TEST(convolution_gemm_post_ops_test, Int8SumAndBinary) {
    SKIP_IF(get_test_engine_kind() != engine::kind::cpu,
            "The test requires host-accessible memory.");
    auto eng = get_test_engine();
    auto strm = make_stream(eng);
    const int channels = 32;
    const memory::dims src_dims {1, 4, 2, 4};
    const memory::dims weights_dims {2, 16, 2, 1, 1};
    const memory::dims dst_dims {1, channels, 2, 4};
    const memory::dims rhs_dims {1, channels, 1, 1};
    const memory::dims strides {1, 1}, padding {0, 0};
    const memory::desc src_md(
            src_dims, memory::data_type::u8, memory::format_tag::nhwc);
    const memory::desc weights_md(
            weights_dims, memory::data_type::s8, memory::format_tag::hwigo);
    const memory::desc dst_md(
            dst_dims, memory::data_type::s32, memory::format_tag::nhwc);
    const memory::desc rhs_md(
            rhs_dims, memory::data_type::f32, memory::format_tag::nhwc);
    std::vector<uint8_t> src(32, 2);
    std::vector<int8_t> weights(2 * channels, -1);
    std::vector<int32_t> dst(8 * channels, 1);
    std::vector<float> rhs(channels);
    for (int channel = 0; channel < channels; ++channel)
        rhs[channel] = static_cast<float>(channel);
    post_ops ops;
    ops.append_sum(1.f);
    ops.append_binary(algorithm::binary_add, rhs_md);
    primitive_attr attr;
    attr.set_post_ops(ops);
    convolution_forward::primitive_desc pd(eng, prop_kind::forward_inference,
            algorithm::convolution_direct, src_md, weights_md, dst_md, strides,
            padding, padding, attr);
    while (std::string(pd.impl_info_str()).find(":gemm") == std::string::npos
            && pd.next_impl()) {}
    ASSERT_NE(std::string(pd.impl_info_str()).find(":gemm"), std::string::npos);
    memory src_mem(src_md, eng, src.data());
    memory weights_mem(weights_md, eng, weights.data());
    memory dst_mem(dst_md, eng, dst.data());
    memory rhs_mem(rhs_md, eng, rhs.data());
    convolution_forward(pd).execute(strm,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weights_mem},
                    {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                            rhs_mem}});
    strm.wait();
    for (int spatial = 0; spatial < 8; ++spatial)
        for (int channel = 0; channel < channels; ++channel)
            EXPECT_EQ(dst[spatial * channels + channel], channel - 3);
}

#define TEST_PARAM_ATTR
#define U8S8
#define DIRECTION_FORWARD
#include "convolution_common.h"
#undef TEST_PARAM_ATTR

} // namespace dnnl
