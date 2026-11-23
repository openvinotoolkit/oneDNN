/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <unordered_map>

#include "common/float16.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "gtest/gtest.h"

namespace dnnl {

TEST(reorder_dequantization_t, PlainLayoutsPreserveGroupedFp16Weights) {
    using dt = memory::data_type;
    using tag = memory::format_tag;
    engine cpu(engine::kind::cpu, 0);
    stream queue(cpu);
    for (const auto shape : {std::array<int, 4> {17, 30, 2, 3},
                 std::array<int, 4> {18, 33, 3, 1},
                 std::array<int, 4> {17, 33, 3, 1},
                 std::array<int, 4> {1, 33, 3, 1},
                 std::array<int, 4> {17, 1, 1, 1},
                 std::array<int, 4> {1, 1, 1, 1}}) {
        const int n = shape[0], k = shape[1];
        const int scale_groups = shape[2], zp_groups = shape[3];
        std::vector<float> scales(n * scale_groups);
        std::vector<int32_t> zero_points(n * zp_groups);
        for (const auto type : {dt::s4, dt::u4, dt::s8, dt::u8}) {
            const bool packed = type == dt::s4 || type == dt::u4;
            const bool signed_values = type == dt::s4 || type == dt::s8;
            for (int i = 0; i < n * scale_groups; ++i)
                scales[i] = 0.067f + static_cast<float>(i % 7) * 0.003f;
            for (int i = 0; i < n * zp_groups; ++i)
                zero_points[i] = (signed_values ? -2 : 3) + i % 3;
            std::vector<impl::float16_t> half_scales(scales.size());
            for (size_t i = 0; i < scales.size(); ++i)
                half_scales[i] = scales[i];
            for (const auto scale_type : {dt::f32, dt::f16}) {
                for (const auto source_tag : {tag::ab, tag::ba}) {
                    for (const auto destination_tag : {tag::ab, tag::ba}) {
                        SCOPED_TRACE(::testing::Message()
                                << n << 'x' << k << '/'
                                << static_cast<int>(type) << '/'
                                << static_cast<int>(source_tag) << '/'
                                << static_cast<int>(destination_tag) << '/'
                                << static_cast<int>(scale_type));
                        std::vector<uint8_t> input(
                                (n * k + (packed ? 1 : 0)) / (packed ? 2 : 1),
                                0);
                        std::vector<impl::float16_t> output(n * k);
                        for (int row = 0; row < n; ++row) {
                            for (int column = 0; column < k; ++column) {
                                const int value = (row * 3 + column * 7) % 16
                                        - (signed_values ? 8 : 0);
                                const int offset = source_tag == tag::ab
                                        ? row * k + column
                                        : column * n + row;
                                if (packed)
                                    input[offset / 2] = static_cast<uint8_t>(
                                            input[offset / 2]
                                            | ((value & 15)
                                                    << (4 * (offset % 2))));
                                else
                                    input[offset] = static_cast<uint8_t>(value);
                            }
                        }
                        memory::desc src({n, k}, type, source_tag);
                        memory::desc dst({n, k}, dt::f16, destination_tag);
                        primitive_attr attr;
                        attr.set_scales(DNNL_ARG_SRC, 3, {1, k / scale_groups},
                                scale_type);
                        attr.set_zero_points(
                                DNNL_ARG_SRC, 3, {1, k / zp_groups}, dt::s32);
                        reorder::primitive_desc pd(cpu, src, cpu, dst, attr);
                        EXPECT_EQ(pd.scratchpad_desc().get_size(), 0U);
                        reorder(pd).execute(queue,
                                {{DNNL_ARG_FROM,
                                         memory(src, cpu, input.data())},
                                        {DNNL_ARG_TO,
                                                memory(dst, cpu,
                                                        output.data())},
                                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                                                memory({{n, scale_groups},
                                                               scale_type,
                                                               tag::ab},
                                                        cpu,
                                                        scale_type == dt::f16
                                                                ? static_cast<
                                                                          void *>(
                                                                          half_scales
                                                                                  .data())
                                                                : static_cast<
                                                                          void *>(
                                                                          scales.data()))},
                                        {DNNL_ARG_ATTR_ZERO_POINTS
                                                        | DNNL_ARG_SRC,
                                                memory({{n, zp_groups}, dt::s32,
                                                               tag::ab},
                                                        cpu,
                                                        zero_points.data())}});
                        queue.wait();
                        for (int row = 0; row < n; ++row) {
                            for (int column = 0; column < k; ++column) {
                                const int value = (row * 3 + column * 7) % 16
                                        - (signed_values ? 8 : 0);
                                const int zero = zero_points[row * zp_groups
                                        + column / (k / zp_groups)];
                                const size_t scale_offset = row * scale_groups
                                        + column / (k / scale_groups);
                                const float scale = scale_type == dt::f16
                                        ? static_cast<float>(
                                                  half_scales[scale_offset])
                                        : scales[scale_offset];
                                const impl::float16_t expected(
                                        (value - zero) * scale);
                                const int offset = destination_tag == tag::ab
                                        ? row * k + column
                                        : column * n + row;
                                EXPECT_EQ(output[offset].raw, expected.raw);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(reorder_dequantization_t,
        BroadcastParametersAndPostOpsPreserveFp16Values) {
    using dt = memory::data_type;
    using tag = memory::format_tag;
    engine cpu(engine::kind::cpu, 0);
    stream queue(cpu);
    constexpr int n = 3, k = 5;
    const auto parameter_desc = [](int mask, dt type) {
        return memory::desc(
                {mask & 1 ? n : 1, mask & 2 ? k : 1}, type, tag::ab);
    };
    const auto parameter_offset = [](int mask, int row, int column) {
        return (mask & 1 ? row : 0) * (mask & 2 ? k : 1)
                + (mask & 2 ? column : 0);
    };
    for (const auto type : {dt::s8, dt::u8}) {
        std::vector<uint8_t> input(n * k);
        for (int i = 0; i < n * k; ++i)
            input[i] = static_cast<uint8_t>(i * 7 - (type == dt::s8 ? 48 : 0));
        for (int scale_mask = 0; scale_mask < 4; ++scale_mask) {
            for (int zero_mask = 0; zero_mask < 4; ++zero_mask) {
                for (const bool with_post_ops : {false, true}) {
                    SCOPED_TRACE(::testing::Message()
                            << static_cast<int>(type) << '/' << scale_mask
                            << '/' << zero_mask << '/' << with_post_ops);
                    const auto scales_md = parameter_desc(scale_mask, dt::f16);
                    const auto zero_md = parameter_desc(zero_mask, dt::s32);
                    std::vector<impl::float16_t> scales(
                            scales_md.get_size() / sizeof(impl::float16_t));
                    std::vector<int32_t> zeros(
                            zero_md.get_size() / sizeof(int32_t));
                    for (size_t i = 0; i < scales.size(); ++i)
                        scales[i] = 0.067f + static_cast<float>(i % 7) * 0.003f;
                    for (size_t i = 0; i < zeros.size(); ++i)
                        zeros[i] = static_cast<int32_t>(i % 3)
                                + (type == dt::u8 ? 3 : -2);
                    std::vector<uint8_t> byte_zeros(zeros.size());
                    for (size_t i = 0; i < zeros.size(); ++i)
                        byte_zeros[i] = static_cast<uint8_t>(zeros[i]);
                    const auto actual_zero_md = parameter_desc(
                            zero_mask, type == dt::u8 ? dt::u8 : dt::s32);
                    std::vector<impl::float16_t> output(
                            n * k, impl::float16_t(1.25f));
                    const memory::desc src({n, k}, type, tag::ab);
                    const memory::desc dst({n, k}, dt::f16, tag::ab);
                    primitive_attr attr;
                    attr.set_scales(DNNL_ARG_SRC, scale_mask, {}, dt::f16);
                    attr.set_zero_points(DNNL_ARG_SRC, zero_mask, {},
                            type == dt::u8 ? dt::u8 : dt::s32);
                    float destination_scale = 0.5f;
                    if (with_post_ops) {
                        attr.set_scales_mask(DNNL_ARG_DST, 0);
                        post_ops operations;
                        operations.append_sum(2.f);
                        attr.set_post_ops(operations);
                    }
                    const reorder::primitive_desc pd(cpu, src, cpu, dst, attr);
                    std::unordered_map<int, memory> arguments {
                            {DNNL_ARG_FROM, memory(src, cpu, input.data())},
                            {DNNL_ARG_TO, memory(dst, cpu, output.data())},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                                    memory(scales_md, cpu, scales.data())},
                            {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
                                    memory(actual_zero_md, cpu,
                                            type == dt::u8
                                                    ? static_cast<void *>(
                                                              byte_zeros.data())
                                                    : static_cast<void *>(
                                                              zeros.data()))}};
                    if (with_post_ops)
                        arguments.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                                memory({{1}, dt::f32, tag::a}, cpu,
                                        &destination_scale));
                    reorder(pd).execute(queue, arguments);
                    queue.wait();
                    for (int row = 0; row < n; ++row) {
                        for (int column = 0; column < k; ++column) {
                            const int value = (row * k + column) * 7
                                    - (type == dt::s8 ? 48 : 0);
                            const int zero = zeros[parameter_offset(
                                    zero_mask, row, column)];
                            const float scale = scales[parameter_offset(
                                    scale_mask, row, column)];
                            const impl::float16_t expected(
                                    ((value - zero) * scale
                                            + (with_post_ops ? 2.5f : 0.f))
                                    / (with_post_ops ? destination_scale
                                                     : 1.f));
                            EXPECT_EQ(
                                    output[row * k + column].raw, expected.raw);
                        }
                    }
                }
            }
        }
    }
}

TEST(reorder_dequantization_t, ScalarS32ZeroPointsPreserveRange) {
    using dt = memory::data_type;
    using tag = memory::format_tag;
    engine cpu(engine::kind::cpu, 0);
    stream queue(cpu);
    for (const auto type : {dt::s4, dt::u4, dt::s8, dt::u8}) {
        for (const bool with_sum : {false, true}) {
            if (with_sum && (type == dt::s4 || type == dt::u4)) continue;
            for (int32_t zero : {std::numeric_limits<int32_t>::min(),
                         std::numeric_limits<int32_t>::max()}) {
                SCOPED_TRACE(::testing::Message()
                        << static_cast<int>(type) << '/' << zero << '/'
                        << with_sum);
                uint8_t input = 0;
                float scale = 1.e-9f;
                impl::float16_t output(1.25f);
                const memory::desc src({1, 1}, type, tag::ab);
                const memory::desc dst({1, 1}, dt::f16, tag::ab);
                primitive_attr attr;
                attr.set_scales(DNNL_ARG_SRC, 3, {1, 1}, dt::f32);
                attr.set_zero_points(DNNL_ARG_SRC, 3, {1, 1}, dt::s32);
                if (with_sum) {
                    post_ops operations;
                    operations.append_sum(2.f);
                    attr.set_post_ops(operations);
                }
                const reorder::primitive_desc pd(cpu, src, cpu, dst, attr);
                reorder(pd).execute(queue,
                        {{DNNL_ARG_FROM, memory(src, cpu, &input)},
                                {DNNL_ARG_TO, memory(dst, cpu, &output)},
                                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                                        memory({{1, 1}, dt::f32, tag::ab}, cpu,
                                                &scale)},
                                {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
                                        memory({{1, 1}, dt::s32, tag::ab}, cpu,
                                                &zero)}});
                queue.wait();
                const impl::float16_t expected(-static_cast<float>(zero) * scale
                        + (with_sum ? 2.5f : 0.f));
                EXPECT_EQ(output.raw, expected.raw);
            }
        }
    }
}

TEST(reorder_dequantization_t, DenseFp16MatchesReferenceWithNegativeScales) {
    using dt = memory::data_type;
    using tag = memory::format_tag;
    engine cpu(engine::kind::cpu, 0);
    stream queue(cpu);
    constexpr int n = 7, k = 35;
    std::vector<impl::float16_t> scales(n * 7);
    std::vector<int32_t> zero_points(n * 5);
    for (size_t i = 0; i < scales.size(); ++i)
        scales[i] = i % 2 ? -0.125f : 0.25f;
    for (const auto type : {dt::s4, dt::u4, dt::s8, dt::u8}) {
        const bool packed = type == dt::s4 || type == dt::u4;
        const bool signed_values = type == dt::s4 || type == dt::s8;
        for (size_t i = 0; i < zero_points.size(); ++i)
            zero_points[i] = signed_values ? -2 : 3;
        for (const auto source_tag : {tag::ab, tag::ba}) {
            for (const auto destination_tag : {tag::ab, tag::ba}) {
                SCOPED_TRACE(::testing::Message()
                        << static_cast<int>(type) << '/'
                        << static_cast<int>(source_tag) << '/'
                        << static_cast<int>(destination_tag));
                std::vector<uint8_t> input(
                        (n * k + (packed ? 1 : 0)) / (packed ? 2 : 1), 0);
                for (int row = 0; row < n; ++row) {
                    for (int column = 0; column < k; ++column) {
                        const int value = (row * 3 + column * 7) % 16
                                - (signed_values ? 8 : 0);
                        const int offset = source_tag == tag::ab
                                ? row * k + column
                                : column * n + row;
                        if (packed)
                            input[offset / 2] |= static_cast<uint8_t>(
                                    (value & 15) << (4 * (offset % 2)));
                        else
                            input[offset] = static_cast<uint8_t>(value);
                    }
                }
                std::array<std::vector<impl::float16_t>, 2> outputs {
                        std::vector<impl::float16_t>(n * k),
                        std::vector<impl::float16_t>(n * k)};
                const memory::desc src({n, k}, type, source_tag);
                const memory::desc dst({n, k}, dt::f16, destination_tag);
                for (int reference = 0; reference < 2; ++reference) {
                    primitive_attr attr;
                    attr.set_scales(DNNL_ARG_SRC, 3, {1, 5}, dt::f16);
                    attr.set_zero_points(DNNL_ARG_SRC, 3, {1, 7}, dt::s32);
                    if (reference) {
                        post_ops operations;
                        operations.append_sum(0.f);
                        attr.set_post_ops(operations);
                    }
                    reorder::primitive_desc pd(cpu, src, cpu, dst, attr);
                    reorder(pd).execute(queue,
                            {{DNNL_ARG_FROM, memory(src, cpu, input.data())},
                                    {DNNL_ARG_TO,
                                            memory(dst, cpu,
                                                    outputs[reference].data())},
                                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                                            memory({{n, 7}, dt::f16, tag::ab},
                                                    cpu, scales.data())},
                                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
                                            memory({{n, 5}, dt::s32, tag::ab},
                                                    cpu, zero_points.data())}});
                    queue.wait();
                }
                for (int i = 0; i < n * k; ++i)
                    EXPECT_EQ(outputs[0][i].raw, outputs[1][i].raw);
            }
        }
    }
}

} // namespace dnnl
