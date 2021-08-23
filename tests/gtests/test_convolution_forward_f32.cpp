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

using convolution_test = convolution_forward_test_t<float, float, float, float>;

TEST_P(convolution_test, TestConvolution) {}

TEST(convolution_gemm_post_ops_test, GroupedDepthwise) {
    SKIP_IF(get_test_engine_kind() != engine::kind::cpu,
	    "The test requires host-accessible memory.");
    auto eng = get_test_engine();
    auto strm = make_stream(eng);
    const memory::dims src_dims {1, 4, 2, 3};
    const memory::dims weights_dims {2, 3, 2, 1, 1};
    const memory::dims dst_dims {1, 6, 2, 3};
    const memory::dims strides {1, 1}, padding {0, 0};
    std::vector<float> src(24, -2.f), weights(12, 1.f), dst(36);
    std::vector<float> scales {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> shifts {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    post_ops ops;
    ops.append_depthwise(algorithm::depthwise_scale_shift, scales.data(),
	    shifts.data());
    ops.append_eltwise(algorithm::eltwise_relu, 0.1f, 0.f);
    primitive_attr attr;
    attr.set_post_ops(ops);

    for (bool channels_last : {false, true}) {
	const auto src_tag = channels_last ? memory::format_tag::nhwc
					  : memory::format_tag::nchw;
	const auto weights_tag = channels_last ? memory::format_tag::hwigo
					      : memory::format_tag::goihw;
	const memory::desc src_md(src_dims, memory::data_type::f32, src_tag);
	const memory::desc weights_md(
		weights_dims, memory::data_type::f32, weights_tag);
	const memory::desc dst_md(dst_dims, memory::data_type::f32, src_tag);
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
			{DNNL_ARG_DST, dst_mem}});
	strm.wait();
	for (int channel = 0; channel < 6; ++channel) {
	    const float expected = (-4.f * scales[channel] + shifts[channel]) * 0.1f;
	    for (int spatial = 0; spatial < 6; ++spatial) {
		const int index = channels_last ? spatial * 6 + channel
						: channel * 6 + spatial;
		EXPECT_NEAR(dst[index], expected, 1e-5f);
	    }
	}
    }
}

TEST(convolution_gemm_post_ops_test, GroupedBf16Depthwise) {
    SKIP_IF(get_test_engine_kind() != engine::kind::cpu,
	    "The test requires host-accessible memory.");
    auto eng = get_test_engine();
    auto strm = make_stream(eng);
    const memory::dims src_dims {1, 4, 2, 3};
    const memory::dims weights_dims {2, 3, 2, 1, 1};
    const memory::dims dst_dims {1, 6, 2, 3};
    const memory::dims strides {1, 1}, padding {0, 0};
    std::vector<uint16_t> src(24, 0xc000), weights(12, 0x3f80);
    std::vector<float> dst(36);
    std::vector<float> scales {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> shifts {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    post_ops ops;
    ops.append_depthwise(algorithm::depthwise_scale_shift, scales.data(),
	    shifts.data());
    ops.append_eltwise(algorithm::eltwise_relu, 0.1f, 0.f);
    primitive_attr attr;
    attr.set_post_ops(ops);
    for (bool channels_last : {false, true}) {
	const auto src_tag = channels_last ? memory::format_tag::nhwc
					  : memory::format_tag::nchw;
	const auto weights_tag = channels_last ? memory::format_tag::hwigo
					      : memory::format_tag::goihw;
	const memory::desc src_md(src_dims, memory::data_type::bf16, src_tag);
	const memory::desc weights_md(
		weights_dims, memory::data_type::bf16, weights_tag);
	const memory::desc dst_md(dst_dims, memory::data_type::f32, src_tag);
	convolution_forward::primitive_desc pd(eng, prop_kind::forward_inference,
		algorithm::convolution_direct, src_md, weights_md, dst_md,
		strides, padding, padding, attr, true);
	if (!pd) GTEST_SKIP() << "BF16 convolution is unavailable";
	while (std::string(pd.impl_info_str()).find(":gemm") == std::string::npos
		&& pd.next_impl()) {}
	if (std::string(pd.impl_info_str()).find(":gemm") == std::string::npos)
	    GTEST_SKIP() << "BF16 GEMM convolution is unavailable";
	memory src_mem(src_md, eng, src.data());
	memory weights_mem(weights_md, eng, weights.data());
	memory dst_mem(dst_md, eng, dst.data());
	convolution_forward(pd).execute(strm,
		{{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weights_mem},
			{DNNL_ARG_DST, dst_mem}});
	strm.wait();
	for (int channel = 0; channel < 6; ++channel) {
	    const float expected = (-4.f * scales[channel] + shifts[channel]) * 0.1f;
	    for (int spatial = 0; spatial < 6; ++spatial) {
		const int index = channels_last ? spatial * 6 + channel
						: channel * 6 + spatial;
		EXPECT_NEAR(dst[index], expected, 1e-5f);
	    }
	}
    }
}

#define FP32
#define DIRECTION_FORWARD
#include "convolution_common.h"

} // namespace dnnl
