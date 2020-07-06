/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#include <mkldnn_types.h>
#include "mkldnn_types.h"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"
#include "jit_generator.hpp"

#include "jit_uni_eltwise.hpp"
#include "jit_avx512_core_bf16cvt.hpp"

#define GET_OFF(field) offsetof(jit_eltwise_args, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble(size_t start_idx,
        size_t end_idx) {
    preserved_vecs_count = 0;
    vecs_to_preserve = (size_t)aux_vecs_count(alg_);
    start_idx_tail = start_idx;

    // For sse42 mask register has to be Xmm(0)
    if (isa == sse42 && vecs_to_preserve > 0) {
        size_t idx = 0;
        assert(idx < start_idx);
        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    for (size_t idx = preserved_vecs_count; idx < vecs_count; idx++) {
        if (preserved_vecs_count >= vecs_to_preserve) break;
        if (start_idx <= idx && idx < end_idx) continue;

        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    size_t preserved_vecs_count_tail = vecs_to_preserve - preserved_vecs_count;
    for (size_t i = 0; i < preserved_vecs_count_tail; i++) {
        preserved_vec_idxs[preserved_vecs_count++] = start_idx_tail++;
    }

    assert(preserved_vecs_count == vecs_to_preserve);

    if (save_state_) {
        h->push(p_table);

        if (preserved_vecs_count)
            h->sub(h->rsp, preserved_vecs_count * vlen);

        for (size_t i = 0; i < preserved_vecs_count; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen],
                    Vmm(preserved_vec_idxs[i]));

        load_table_addr();
    }

    assign_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble_tail(size_t start_idx)
{
    size_t tail_vecs_to_preserve = start_idx_tail - start_idx;
    if (tail_vecs_to_preserve == 0) return;

    const int idx_off = vecs_to_preserve - tail_vecs_to_preserve;

    if (save_state_) {
        if (idx_off)
            h->add(h->rsp, idx_off * vlen);

        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(Vmm(preserved_vec_idxs[idx_off + i]),
                    h->ptr[h->rsp + i * vlen]);
    }

    for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
        preserved_vec_idxs[idx_off + i] += tail_vecs_to_preserve;

    if (save_state_) {
        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen],
                    Vmm(preserved_vec_idxs[idx_off + i]));

        if (idx_off)
            h->sub(h->rsp, idx_off * vlen);
    }

    assign_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_postamble() {
    if (!save_state_) return;

    for (size_t i = 0; i < preserved_vecs_count; ++i)
        h->uni_vmovups(Vmm(preserved_vec_idxs[i]),
                h->ptr[h->rsp + i * vlen]);

    if (preserved_vecs_count)
        h->add(h->rsp, preserved_vecs_count * vlen);

    h->pop(p_table);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::assign_regs() {
    vmm_mask = Vmm(preserved_vec_idxs[0]);
    vmm_aux0 = Vmm(preserved_vec_idxs[0]);
    vmm_aux1 = Vmm(preserved_vec_idxs[1]);
    vmm_aux2 = Vmm(preserved_vec_idxs[2]);
    vmm_aux3 = Vmm(preserved_vec_idxs[3]);
    vmm_aux4 = Vmm(preserved_vec_idxs[4]);
}

// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Stores a mask by applying cmpps on two inputs w/ a given predicate.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_cmp_mask(const Vmm &vmm_src,
        const Xbyak::Operand &compare_operand, int cmp_predicate) {
    if (has_avx512()) {
        h->vcmpps(k_mask, vmm_src, compare_operand, cmp_predicate);
    } else {
        h->uni_vcmpps(vmm_mask, vmm_src, compare_operand, cmp_predicate);
    }
}

// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Blends a result of second input into a first input w/ a stored mask.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::blend_with_mask(
        const Vmm &vmm_dst, const Xbyak::Operand &src) {
    if (has_avx512()) {
        h->vblendmps(vmm_dst | k_mask, vmm_dst, src);
    } else {
        h->uni_vblendvps(vmm_dst, vmm_dst, src, vmm_mask);
    }
}

// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Tests a mask for all zeros. If all zeroes occur, set ZF = 1.
// Nicely combines with jump_if_zero (jz).
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::test_mask() {
    if (has_avx512()) {
        h->kortestw(k_mask, k_mask);
    } else {
        h->uni_vtestps(vmm_mask, vmm_mask);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector(const Vmm &vmm_src) {
    // get mask of values lower than log(FLT_MIN) to zero them in the output
    if (utils::one_of(isa, avx512_common, avx512_core))
        h->vcmpps(k_mask, vmm_src, table_val(11), _cmp_lt_os);
    else if (isa == avx2)
        h->vcmpltps(vmm_mask, vmm_src, table_val(11));
    else if (isa == sse42) {
        h->uni_vmovups(vmm_mask, vmm_src);
        h->cmpltps(vmm_mask, table_val(11));
    }

    h->uni_vminps(vmm_src, vmm_src, table_val(10));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(11));
    h->uni_vmovups(vmm_aux1, vmm_src);
    //calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(2));
    h->uni_vaddps(vmm_src, vmm_src, table_val(1));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux2, vmm_src, _op_floor);

    //keep fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux2); //vmm_src = fx

    //x = x - fx * ln2
    h->uni_vfnmadd231ps(vmm_aux1, vmm_aux2, table_val(3));

    // compute 2^n
    h->uni_vcvtps2dq(vmm_aux2, vmm_src);
    h->uni_vpaddd(vmm_aux2, vmm_aux2, table_val(4));
    h->uni_vpslld(vmm_aux2, vmm_aux2, 23); //Vmm(6) = 2^-fx

    // use vmm_src as tmp vmm_zero when applying mask
    h->uni_vpxor(vmm_src, vmm_src, vmm_src);
    // set zeroes according to the mask
    if (utils::one_of(isa, avx512_common, avx512_core))
        h->vblendmps(vmm_aux2 | k_mask, vmm_aux2, vmm_src);
    else
        h->uni_vblendvps(vmm_aux2, vmm_aux2, vmm_src, vmm_mask);

    // y = p5
    h->uni_vmovups(vmm_src, table_val(9));
    // y = y * x + p4
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(8));
    // y = y * x + p3
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(7));
    // y = y * x + p2
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(6));
    // y = y * x + p1
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(0));
    // y = y * x + p0
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(5));  //exp(q)
    // y = y * 2^n
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector(const Vmm &vmm_src)
{
    const int alpha_off = 0, zero_off = 1;

    h->uni_vmovups(vmm_aux1, vmm_src);
    if (isa == sse42) {
        h->movups(vmm_mask, vmm_src);
        h->mulps(vmm_src, table_val(alpha_off));
        h->cmpps(vmm_mask, table_val(zero_off), _cmp_nle_us);
        h->blendvps(vmm_src, vmm_aux1);
    } else if (isa == avx2) {
        h->vmulps(vmm_src, vmm_src, table_val(alpha_off));
        h->vcmpgtps(vmm_mask, vmm_aux1, table_val(zero_off));
        h->vblendvps(vmm_src, vmm_src, vmm_aux1, vmm_mask);
    } else if (isa == avx512_common) {
        h->vmulps(vmm_src, vmm_src, table_val(alpha_off));
        h->vcmpps(k_mask, vmm_aux1, table_val(zero_off), _cmp_nle_us);
        h->vblendmps(vmm_src | k_mask, vmm_src, vmm_aux1);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_zero_ns_compute_vector(
        const Vmm &vmm_src) {
    const int zero_off = 1;
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero_off));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector(const Vmm &vmm_src) {
    const int alpha_off = 25, zero_off = 26;

    // compute exponent
    h->uni_vmovups(vmm_aux3, vmm_src);
    exp_compute_vector(vmm_src);

    // alpha * (exp(x) - 1)
    h->uni_vsubps(vmm_src, vmm_src, table_val(0));
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha_off));

    // combine with mask
    if (isa == sse42) {
        h->pxor(vmm_mask, vmm_mask);
        h->cmpps(vmm_mask,  vmm_aux3, _cmp_le_os);
        h->blendvps(vmm_src, vmm_aux3);
    } else if (isa == avx2) {
        h->uni_vcmpgtps(vmm_mask, vmm_aux3, table_val(zero_off));
        h->uni_vblendvps(vmm_src, vmm_src, vmm_aux3, vmm_mask);
    } else if (isa == avx512_common) {
        h->vcmpps(k_mask, vmm_aux3, table_val(zero_off), _cmp_nle_us);
        h->vblendmps(vmm_src | k_mask, vmm_src, vmm_aux3);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector(const Vmm &vmm_src)
{
    // # comes from Taylor expansion error bound
    //  > linear_sat_point = single(sqrt(3) * 1b-12);
    // # comes from the exp formula cancellation
    //  > exp_bound_point = (single(log(3)/2));
    // # comes from rounding accuracy in float
    //  > one_sat_point = round(atanh(1 - 1b-25), single, RU);
    //  > P = fpminimax(f, [|1, 3, 5, 7, 9|], [|24... |],
    //            [linear_sat_point, exp_bound_point], relative, floating);
    //  > err_bound = D(sup(supnorm(P, tanh(x),
    //          [linear_sat_point, exp_bound_point], relative, theta)));
    //    0x1.fffd6f00b9539p-25
    //  > P;
    //    x * (0x1.fffffep-1 + x^0x1p1 * (-0x1.55539ep-2 + x^0x1p1 *
    //        (0x1.10be3ep-3 + x^0x1p1 * (-0x1.ae57b4p-5
    //        + x^0x1p1 * 0x1.09fa1p-6))))

    // register mapping
    // vmm_src contains input
    // vmm_aux0 contains mask of currently valid results.
    //     1 is need computation, 0 is already computed
    // vmm_aux1 contains current output
    // vmm_aux2, vmm_aux3 contains auxiliary values
    // vmm_aux4 contains the original sign of inputs

    Label end_tanh_label;

    auto test_exit =[&](Xbyak::Address threshold){
        // is not necessary for >AVX, but should not matter on perf
        h->uni_vmovups(vmm_aux0, vmm_src);
        if (isa == avx512_common){
            h->vcmpps(k_mask, vmm_aux0, threshold, 0x5);
            h->kortestw(k_mask, k_mask);
        } else {
            h->uni_vcmpgeps(vmm_aux0, vmm_aux0, threshold);
            h->uni_vtestps(vmm_aux0, vmm_aux0);
        }
        h->jz(end_tanh_label, Xbyak::CodeGenerator::T_NEAR);
    };

    auto blend_results=[&](Vmm vmm_partial_res){
        if (isa == avx512_common)
            h->vblendmps(vmm_aux1 | k_mask, vmm_aux1, vmm_partial_res);
        else
            h->uni_vblendvps(vmm_aux1, vmm_aux1, vmm_partial_res, vmm_aux0);
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x postive
    // and reapply sign at the end
    // mov is not necessary for >AVX, but should not matter for performance
    h->uni_vmovups(vmm_aux4, vmm_src);
    h->uni_vandps(vmm_aux4, vmm_aux4, table_val(12));
    h->uni_vandps(vmm_src, vmm_src, table_val(17));

    // if x < linear_sat_point for all inputs, we just return the input
    h->uni_vmovups(vmm_aux1, vmm_src);
    test_exit(table_val(13));

    // if one of the mask is one, we have to compute an better approx
    h->uni_vmovups(vmm_aux2, vmm_src);
    h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_aux2);
    h->uni_vmovups(vmm_aux3, table_val(22));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(21));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(20));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(19));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux2, table_val(18));
    h->uni_vmulps(vmm_aux3, vmm_aux3, vmm_src);

    // we blend only the result that need update
    blend_results(vmm_aux3);

    // if x < exp_bound_point, we go to return point
    test_exit(table_val(14));

    // if not we use a better approx 1 - 2 / (1 + exp(2x))
    // compute 2x
    h->uni_vmovups(vmm_aux3, vmm_src);
    h->uni_vaddps(vmm_aux3, vmm_aux3, vmm_aux3);

    // Compute exp(2x)
    // We need to save kmask, vmm_aux0, vmm_aux1, vmm_aux2 and vmm_src as exp
    // uses them.
    // vmm_src is not more read afterwards, so we do not have to save it
    auto stack_size = 4 * vlen + (isa == avx512_common) * 4;
    h->sub(h->rsp, stack_size);
    h->uni_vmovups(h->ptr[h->rsp + 0 * vlen], vmm_aux0);
    h->uni_vmovups(h->ptr[h->rsp + 1 * vlen], vmm_aux1);
    h->uni_vmovups(h->ptr[h->rsp + 2 * vlen], vmm_aux2);
    h->uni_vmovups(h->ptr[h->rsp + 3 * vlen], vmm_src);
    if (isa == avx512_common)
        h->kmovw(h->ptr[h->rsp + 4 * vlen], k_mask);

    exp_compute_vector(vmm_aux3);

    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp + 0 * vlen]);
    h->uni_vmovups(vmm_aux1, h->ptr[h->rsp + 1 * vlen]);
    h->uni_vmovups(vmm_aux2, h->ptr[h->rsp + 2 * vlen]);
    h->uni_vmovups(vmm_src, h->ptr[h->rsp + 3 * vlen]);
    if (isa == avx512_common)
        h->kmovw(k_mask, h->ptr[h->rsp + 4 * vlen]);
    h->add(h->rsp, stack_size);

    // 1 + exp(2x)
    h->uni_vaddps(vmm_aux3, vmm_aux3, table_val(0));

    // 1 - 2 / (1 + exp(2x))
    h->uni_vmovups(vmm_aux2, table_val(16));
    h->uni_vdivps(vmm_aux2, vmm_aux2, vmm_aux3);
    h->uni_vaddps(vmm_aux2, vmm_aux2, table_val(0));

    // we blend only the result that need update
    blend_results(vmm_aux2);

    // finally, we saturate to 1 if needed
    // TODO: maybe move that up if most inputs saturate in practice
    if (isa == avx512_common)
        h->vcmpps(k_mask, vmm_aux0, table_val(15), 0x5);
    else {
        h->uni_vmovups(vmm_aux0, vmm_src);
        h->uni_vcmpgeps(vmm_aux0, vmm_aux0, table_val(15));
    }
    h->uni_vmovups(vmm_aux2, table_val(0));
    blend_results(vmm_aux2);

    h->L(end_tanh_label);
    {
        // we apply the sign of x to the result and we are done
        h->uni_vmovups(vmm_src, vmm_aux1);
        h->uni_vpxor(vmm_src, vmm_src, vmm_aux4);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_compute_vector(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux0, vmm_src);

    // compute G(x) = a * x * (1 + b * x * x)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vmovups(vmm_aux1, table_val(23));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(0));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
    h->uni_vmulps(vmm_src, vmm_src, table_val(24));

    // save x on stack as tanh uses vmm_aux0
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_aux0);

    // compute tanh G(x)
    tanh_compute_vector(vmm_src);

    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    // compute 0.5 * x * (1 + tanh)
    h->uni_vaddps(vmm_src, vmm_src, table_val(0));
    h->uni_vmulps(vmm_src, vmm_src, table_val(1));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector(
        const Vmm &vmm_src) {
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector(const Vmm &vmm_src) {
    // compute abs(x) = _mm_and_ps(x, 01111..111));
    h->uni_vandps(vmm_src, vmm_src, table_val(0));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector(const Vmm &vmm_src)
{
    if (isa == avx512_common) {
        h->vcmpps(k_mask, vmm_src, table_val(0), _cmp_nle_us);
        h->uni_vsqrtps(vmm_aux1, vmm_src);
        h->uni_vmovups(vmm_src, table_val(0));
        h->vblendmps(vmm_src | k_mask, vmm_src, vmm_aux1);
    } else {
        h->uni_vmovups(vmm_mask, vmm_src);
        h->uni_vcmpgtps(vmm_mask, vmm_mask, table_val(0));
        h->uni_vsqrtps(vmm_aux1, vmm_src);
        h->uni_vmovups(vmm_src, table_val(0));
        h->uni_vblendvps(vmm_src, vmm_src, vmm_aux1, vmm_mask);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector(
        const Vmm &vmm_src) {
    // compute x = alpha * x + beta;
    h->uni_vmovups(vmm_aux0, table_val(0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux0, table_val(1));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector(
        const Vmm &vmm_src) {
    // compute bounded relu */
    h->uni_vmaxps(vmm_src, vmm_src, table_val(1));
    h->uni_vminps(vmm_src, vmm_src, table_val(0));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector(
        const Vmm &vmm_src) {
    // duplicate src
    h->uni_vmovups(vmm_aux2, vmm_src);

    h->uni_vminps(vmm_src, vmm_src, table_val(24));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(25));
    h->uni_vmovups(vmm_aux1, vmm_src);
    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(2));
    h->uni_vaddps(vmm_src, vmm_src, table_val(1));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux0, vmm_src, _op_floor);

    // keep fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux0); //vmm_src = fx
    // calculation fx * ln2
    h->uni_vmulps(vmm_aux0, vmm_aux0, table_val(3));
    // x = x - fx * ln2
    h->uni_vsubps(vmm_aux1, vmm_aux1, vmm_aux0);
    // y = p5
    h->uni_vmovups(vmm_aux3, table_val(22));
    // y = y * x + p4
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(21));
    // y = y * x + p3
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(20));
    // y = y * x + p2
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(19));
    // y = y * x + p1
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(0));
    // y = y * x + p0
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(17));

    // compute 2^(-n)
    if (isa == avx512_common) {
        h->vmulps(vmm_aux1, vmm_src, table_val(23));
        h->vcvtps2dq(vmm_aux1, vmm_aux1);
    } else {
        h->uni_vcvtps2dq(vmm_aux1, vmm_src);
        h->uni_vpsignd(vmm_aux1, vmm_aux1, table_val(23));
    }

    h->uni_vpaddd(vmm_aux1, vmm_aux1, table_val(4));
    h->uni_vpslld(vmm_aux1, vmm_aux1, 23); //vmm_aux1 = 2^-fx
    // calculate ln(1 + y)
    h->uni_vaddps(vmm_aux3, vmm_aux3, vmm_aux1);
    // x = y; y is free; keep x for further computations
    h->uni_vmovups(vmm_src, vmm_aux3);
    // frexp()
    h->uni_vpsrld(vmm_src, vmm_src, 23);
    h->uni_vcvtdq2ps(vmm_src, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->uni_vsubps(vmm_src, vmm_src, table_val(5));

    h->uni_vandps(vmm_aux3, vmm_aux3, table_val(6));
    // got y. (mantisa)  0.5 < y < 1
    h->uni_vorps(vmm_aux3, vmm_aux3, table_val(7));
    // y  = y - 1
    h->uni_vsubps(vmm_aux3, vmm_aux3, table_val(0));
    // y = p8
    h->uni_vmovups(vmm_aux1, table_val(16));
    // y = y * x + p7
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(15));
    // y = y * x + p6
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(14));
    // y = y * x + p5
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(13));
    // y = y * x + p4
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(12));
    // y = y * x + p3
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(11));
    // y = y * x + p2
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(10));
    // y = y * x + p1
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(9));
    // y = y * x + p0 ; p0 = 0
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(8));
    //calculate ln(2) * n
    h->uni_vmulps(vmm_src, vmm_src, table_val(3));
    h->uni_vaddps(vmm_aux1, vmm_aux1, vmm_src);
    h->uni_vaddps(vmm_aux1, vmm_aux1, vmm_aux0);

    // get vmm_mask = src > max logf
    h->uni_vmovups(vmm_mask, vmm_aux2);
    if (isa == avx512_common) {
        // y = (x < max log f) ? soft_relu(x) : x
        h->vcmpps(k_mask, vmm_mask, table_val(24), _cmp_nle_us);
        h->vblendmps(vmm_aux1 | k_mask, vmm_aux1, vmm_aux2);
    } else {
        // y = (x < max log f) ? soft_relu(x) : x
        h->uni_vcmpgtps(vmm_mask, vmm_mask, table_val(24));
        h->uni_vblendvps(vmm_aux1, vmm_aux1, vmm_aux2, vmm_mask);
    }

    h->uni_vmovups(vmm_src, vmm_aux1);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector(
        const Vmm &vmm_src) {
    // we store the original sign and make x negative
    // IMPORTANT: we assume vmm_aux0 to be xmm0, as for sse4.1 path it is required
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);
    h->uni_vandps(vmm_aux3, vmm_aux3, table_val(12));
    h->uni_vorps(vmm_src, vmm_src, table_val(12));

    exp_compute_vector(vmm_src);
    // dup exp(x)
    h->uni_vmovups(vmm_aux1, vmm_src);
    // (exp(x) + 1)
    h->uni_vaddps(vmm_aux1, vmm_aux1, table_val(0));
    // y = exp(x) / (exp(x) + 1)
    h->uni_vdivps(vmm_src, vmm_src, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    h->uni_vmovups(vmm_aux2, table_val(0));
    h->uni_vsubps(vmm_aux2, vmm_aux2, vmm_src);
    if (isa == avx512_common) {
        h->vptestmd(k_mask, vmm_aux3, vmm_aux3);
        h->vblendmps(vmm_aux2 | k_mask, vmm_aux2, vmm_src);
    } else {
        h->uni_vmovups(vmm_aux0, vmm_aux3);// The mask should be xmm0 for sse4.1
        h->uni_vblendvps(vmm_aux2, vmm_aux2, vmm_src, vmm_aux0);
    }
    h->uni_vmovups(vmm_src, vmm_aux2);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clamp_compute_vector(
        const Vmm &vmm_src) {
    // compute clamp */
    h->uni_vmaxps(vmm_src, vmm_src, table_val(1));
    h->uni_vminps(vmm_src, vmm_src, table_val(0));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector(
        const Vmm &vmm_src) {
    const int alpha_off = 25;
    // Save src data on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);
    // x*alpha
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha_off));
    // sigmoid(x*alpha)
    logistic_compute_vector(vmm_src);
    // x*sigmoid(alpha*x)
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector(
        const Vmm &vmm_src) {
    // From J.-M. Muller and others, Handbook of Floating-Point Arithmetic, 2010
    // Here is a brief mathematics to approximate log(x):
    // log(x) = E * log(2) + log(y), where -log(2)/2 <= log(y) <= log(2)/2;
    // log(y) = log(1 + z) - log(r_i), where z = y * r_i - 1, r_i approximates
    //   1 / y, i is index of one of precomputed values;
    // log(1 + z) ~~ polynomial(z), =>
    // if (x is normal)
    //     log(x) ~~ E * log(2) + polynomial(z) - log(r_i),
    // where log(r_i) is table value.
    //
    // If (x == 0) result = -inf;
    // If (x < 0) result = qnan;

    // save source on stack to check neg and zero values at the end
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);

    // compute i
    const int approx_order = 5;
    h->uni_vpsrld(vmm_aux1, vmm_src, n_mantissa_bits - approx_order);
    h->uni_vandps(vmm_aux1, vmm_aux1, table_val(log_five_bit_offset));
    h->uni_vpslld(vmm_aux1, vmm_aux1, 1); // multiply i by 2

    // compute anticancellation i
    h->uni_vpsrld(vmm_aux2, vmm_aux1, approx_order);

    // get E, don't care about sign as only positive numbers are considered
    h->uni_vpsrld(vmm_aux3, vmm_src, n_mantissa_bits);
    h->uni_vpaddd(vmm_aux3, vmm_aux3, vmm_aux2);
    h->uni_vcvtdq2ps(vmm_aux3, vmm_aux3);

    // get m (mantissa)
    h->uni_vxorps(vmm_aux2, vmm_aux2, table_val(exponent_bias));
    h->uni_vpslld(vmm_aux2, vmm_aux2, n_mantissa_bits);
    h->uni_vandps(vmm_src, vmm_src, table_val(log_mantissa_mask));
    h->uni_vorps(vmm_src, vmm_src, vmm_aux2);

    // At first, adjust indices for table structure which broadcasts elements
    if (has_avx512()) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 4); // multiply by simd_w = 16
    } else if (isa == avx2) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 3); // multiply by simd_w = 8
    } else if (isa == sse42) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 2); // multiply by simd_w = 4
    }

    const auto it = entry_map_.find(log_predefined_vals);
    assert(it != entry_map_.end());
    const auto table_start_idx = (*it).second.off;

    auto gather_table_values = [&](const Vmm &vmm_dst, const Vmm &vmm_idxs,
                                       size_t offt = 0) {
        Xbyak::Address table_idx = h->ptr[p_table + table_start_idx + offt
                + vmm_idxs * sizeof(float)];
        if (has_avx512()) {
            h->kmovw(k_mask, table_val(log_full_k_reg_mask));
            h->vgatherdps(vmm_dst | k_mask, table_idx);
        } else if (isa == avx2) {
            h->uni_vmovups(vmm_mask, table_val(sign_mask));
            h->vgatherdps(vmm_dst, table_idx, vmm_mask);
        } else if (isa == sse42) {
            Xbyak::Reg64 reg_tmp
                    = p_table.getIdx() != h->r9.getIdx() ? h->r9 : h->r10;

            const int gpr_size = 8;
            // save reg_tmp state as we are not allowed to spoil it.
            h->sub(h->rsp, gpr_size);
            h->mov(h->ptr[h->rsp], reg_tmp);

            // rest of code puts indices on stack, fetching a table number based
            // on an index, replaces index with the value, and, finally, moves
            // fetched values into vector register.
            h->sub(h->rsp, vlen);
            h->uni_vmovups(h->ptr[h->rsp], vmm_idxs);

            for (size_t i = 0; i < vlen / sizeof(float); ++i) {
                h->mov(reg_tmp.cvt32(), h->ptr[h->rsp + i * sizeof(float)]);
                h->shl(reg_tmp.cvt32(), 2); // multiply by simd_w
                table_idx = h->ptr[p_table + table_start_idx + offt + reg_tmp];
                h->mov(reg_tmp.cvt32(), table_idx);
                h->mov(h->ptr[h->rsp + i * sizeof(float)], reg_tmp.cvt32());
            }

            h->uni_vmovups(vmm_dst, h->ptr[h->rsp]);
            h->add(h->rsp, vlen);
            // restore GPR state
            h->mov(reg_tmp, h->ptr[h->rsp]);
            h->add(h->rsp, gpr_size);
        }
    };

    // get r_i, same as table(i)
    gather_table_values(vmm_aux2, vmm_aux1, 0);

    // compute relative error (rel_err = m * r_i - 1)
    h->uni_vfmsub213ps(vmm_aux2, vmm_src, table_val(one));

    // compute polynomial(rel_err)
    h->uni_vmovups(vmm_src, table_val(log_pol, 3));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 2));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 1));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);

    // get log(r_i) = table(i+1)
    gather_table_values(vmm_aux2, vmm_aux1, vlen);

    // compute partial result (pres = E * ln(2) - log(r_i))
    h->uni_vfmadd231ps(vmm_aux2, vmm_aux3, table_val(ln2f));

    // compute (result = polynomial + pres) w/ TwoSum algorithm
    // TODO: restore this instead of version below when asserts are gone
    // h->uni_vaddps(vmm_aux1, vmm_src, vmm_aux2); // res_hi = pol + pres
    // h->uni_vsubps(vmm_aux3, vmm_aux1, vmm_aux2); // res_lo = res_hi - pres
    // h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src); // res_lo = res_lo - pol
    // h->uni_vaddps(vmm_src, vmm_aux1, vmm_aux3); // res_hi = pol + pres

    h->uni_vmovups(vmm_aux1, vmm_src);
    h->uni_vaddps(vmm_aux1, vmm_aux1, vmm_aux2); // res_hi = pol + pres
    h->uni_vmovups(vmm_aux3, vmm_aux1);
    h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_aux2); // res_lo = res_hi - pres
    h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src); // res_lo = res_lo - pol
    h->uni_vmovups(vmm_src, vmm_aux1);
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux3); // res_hi = pol + pres

    // Check original source for zero and neg values. skip blend w/ extreme
    // values if all src values were positive.
    h->uni_vmovups(vmm_aux1, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    Xbyak::Label end_log_label;
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_le_os);
    test_mask();
    h->jz(end_log_label);

    // Blend extreme values into src if reach here.
    // First zero for -inf values...
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_eq_oq);
    blend_with_mask(vmm_src, table_val(log_minus_inf));

    // ...then negative for qnan values.
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(log_qnan));

    h->L(end_log_label);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_prepare_table() {
    const unsigned int cvals[] = {
            0x3f800000, // [0] 1.0f
            0x3f000000, // [1] 0.5f
            0x3fb8aa3b, // [2] log2ef = 1.44269502f
            0x3f317218, // [3] ln2f =   0.69314718f
            0x0000007f, // [4] 0x7f
            // exp(x) polynom
            0x3f800001, // [5] p0 = 1.0000001f
            0x3efffe85, // [6] p2 = 0.4999887f
            0x3e2aaa3e, // [7] p3 = 0.16666505f
            0x3d2bb1b1, // [8] p4 = 0.041917507f
            0x3c091ec1, // [9] p5 = 0.008369149f
            0x42b17218, //[10] logf(FLT_MAX)
            0xc2aeac50, //[11] logf(FLT_MIN)
            // tanh(x) constants,
            0x80000000, //[12] mask to extract sign
            0x39ddb3d7, //[13] arg below which tanh(x) = x
            0x3f0c9f54, //[14] arg below which pol approx is valid
            0x41102cb4, //[15] arg after which tanh(x) = 1
            0xc0000000, //[16] -2.0f
            0x7fffffff, //[17] mask to make positive
            // tanh pol approx
            0x3f7fffff, //[18] p0
            0xbeaaa9cf, //[19] p1
            0x3e085f1f, //[20] p2
            0xbd572bda, //[21] p3
            0x3c84fd08, //[22] p4
            // gelu approx constants
            0x3d372713, //[23] 0.044715
            0x3f4c4229, //[24] sqrt(2/pi)
    };

    for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(cvals[i]);
    }

    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(0);
}


template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_prepare_table() {
    // Assumption: entries can be inserted with dd, so they should be 4 bytes.
    assert(sizeof(table_entry_val_t) == 4);

    // Assumption: iterating on entry_map_ here has the same order as
    // when we set the offsets. We verify that in asserts.
    // table_entry_val_t is assumed to be 32 bits
#ifndef NDEBUG
    size_t off = 0;
    key_t curr_key = undef_key;
    int key_occurences = 0;
#endif

    // Run through the map and insert values stored there
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        const auto &te = (*it).second; // get map entry for a given key
        const auto len = te.bcast ? vlen : sizeof(table_entry_val_t);
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
            h->dd(te.val);

#ifndef NDEBUG
        // we check that the precomputed offsets match the registered ones
        const auto &key = (*it).first; // get map entry key
        if (key != curr_key) {
            curr_key = key;
            key_occurences = 0;
        }
        key_occurences++;
        auto expected_off = table_off(key, key_occurences - 1);
        assert(off == expected_off);
        MAYBE_UNUSED(expected_off);
        off += len;
#endif
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_prepare_table() {
    const unsigned int cvals[] = {
            0x3f800000, // [0] 1.0f
            0x3f000000, // [1] 0.5f
            0x3fb8aa3b, // [2] log2ef = 1.44269502f
            0x3f317218, // [3] ln2f =   0.69314718f
            0x0000007f, // [4] 0x7f
            0x42fc0000, // [5] 126
            0x807fffff, // [6] and with (to get 0.5 * mantissa)
            0x3f000000, // [7] or with (to get 0.5 * mantissa)
            // ln(1 + x) polynomial
            0xb2b4637d, // [8]  p0 = 0.0000000244f
            0x3f7fff8e, // [9]  p1 = 0.9999976971f
            0xbf001759, //[10]  p2 = -0.5002478215f
            0x3ea70608, //[11]  p3 = 0.3272714505f
            0xbea3d7bf, //[12]  p4 = -0.3153830071f
            0xbe361d04, //[13]  p5 = -0.1701777461f
            0xbfa8f1e6, //[14]  p6 = -1.3254635147f
            0xbfe1e812, //[15]  p7 = -1.7971917960f
            0xbfc4d30e, //[16]  p8 = -1.5652673123f
            // exp(x) polynomial
            0x3f800001, //[17]  p0 = 1.0000001f
            0x3f800000, //[18]  p1 = 1.0f
            0x3efffe85, //[19]  p2 = 0.4999887f
            0x3e2aaa3e, //[20]  p3 = 0.16666505f
            0x3d2bb1b1, //[21]  p4 = 0.041917507f
            0x3c091ec1, //[22]  p5 = 0.008369149f
            0xbf800000, //[23] is required for sign changing
            // TODO: update values [24] and [25] from comments as they are more precise
            0x42b0c0a5, //[24] max logf = 88.3762589f //0x42b17218, //[24] logf(FLT_MAX)
            0xc1766666  //[25] min logf = -14.5f      //0xc2aeac50, //[25] logf(FLT_MIN)
    };

    for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
        for (size_t d = 0; d < vlen / sizeof(float); ++d) {
            h->dd(cvals[i]);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(0x7fffffff);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(beta_));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(0);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clamp_prepare_table() {
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(alpha_));
    for (size_t d = 0; d < vlen / sizeof(float); ++d) h->dd(float2int(beta_));
}

template <cpu_isa_t isa>
int jit_uni_eltwise_injector_f32<isa>::aux_vecs_count(alg_kind_t alg_) {
    switch (alg_) {
    case alg_kind::eltwise_relu: return (alpha_ == 0.f) ? 0 : 2;
    case alg_kind::eltwise_elu: return 4;
    case alg_kind::eltwise_tanh: return 5;
    case alg_kind::eltwise_square: return 0;
    case alg_kind::eltwise_abs: return 0;
    case alg_kind::eltwise_sqrt: return 2;
    case alg_kind::eltwise_linear: return 1;
    case alg_kind::eltwise_bounded_relu: return 0;
    case alg_kind::eltwise_soft_relu: return 4;
    case alg_kind::eltwise_logistic: return 4;
    case alg_kind::eltwise_exp: return 3;
    case alg_kind::eltwise_gelu: return 5;
    case alg_kind::eltwise_clamp: return 0;
    case alg_kind::eltwise_swish: return 4;
    case alg_kind::eltwise_log: return 5;
    default: assert(!"unsupported eltwise algorithm");
    }

    return 0;
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_body(size_t start_idx,
        size_t end_idx) {
    using namespace alg_kind;
    for (size_t idx = start_idx; idx < end_idx; idx++) {
        switch (alg_) {
        case eltwise_relu:
            if (alpha_ == 0.f) relu_zero_ns_compute_vector(Vmm(idx));
            else relu_compute_vector(Vmm(idx));
            break;
        case eltwise_elu: elu_compute_vector(Vmm(idx)); break;
        case eltwise_tanh: tanh_compute_vector(Vmm(idx)); break;
        case eltwise_square: square_compute_vector(Vmm(idx)); break;
        case eltwise_abs: abs_compute_vector(Vmm(idx)); break;
        case eltwise_sqrt: sqrt_compute_vector(Vmm(idx)); break;
        case eltwise_linear: linear_compute_vector(Vmm(idx)); break;
        case eltwise_bounded_relu: bounded_relu_compute_vector(Vmm(idx)); break;
        case eltwise_soft_relu: soft_relu_compute_vector(Vmm(idx)); break;
        case eltwise_logistic: logistic_compute_vector(Vmm(idx)); break;
        case eltwise_exp: exp_compute_vector(Vmm(idx)); break;
        case eltwise_gelu: gelu_compute_vector(Vmm(idx)); break;
        case eltwise_clamp: clamp_compute_vector(Vmm(idx)); break;
        case eltwise_swish: swish_compute_vector(Vmm(idx)); break;
        case eltwise_log: log_compute_vector(Vmm(idx)); break;
        default: assert(!"unsupported eltwise algorithm");
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx) {
    assert(start_idx < end_idx && end_idx <= vecs_count);

    injector_preamble(start_idx, end_idx);
    compute_body(start_idx_tail, end_idx);
    injector_preamble_tail(start_idx);
    compute_body(start_idx, start_idx_tail);
    injector_postamble();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::prepare_table(bool gen_table) {
    using namespace alg_kind;

    h->align(64);
    h->L(l_table);

    if (gen_table) {
        switch (alg_) {
        case eltwise_relu: relu_prepare_table(); break;
        case eltwise_elu:
        case eltwise_tanh:
        case eltwise_logistic:
        case eltwise_exp:
        case eltwise_gelu:
        case eltwise_swish:
            elu_prepare_table(); break;
        case eltwise_log:
            log_prepare_table(); break;
        case eltwise_soft_relu: soft_relu_prepare_table(); break;
        case eltwise_abs: abs_prepare_table(); break;
        case eltwise_sqrt: sqrt_prepare_table(); break;
        case eltwise_linear: linear_prepare_table(); break;
        case eltwise_bounded_relu: bounded_relu_prepare_table(); break;
        case eltwise_square: break;
        case eltwise_clamp: clamp_prepare_table(); break;
        default: assert(!"unsupported eltwise algorithm");
    }
    }
}

template struct jit_uni_eltwise_injector_f32<avx512_common>;
template struct jit_uni_eltwise_injector_f32<avx2>;
template struct jit_uni_eltwise_injector_f32<sse42>;


struct jit_eltwise_args {
    const void *from;
    const void *for_comparison;
    const void *to;
    size_t work_amount;
};

struct jit_uni_eltwise_kernel_f32 : public c_compatible {
    const eltwise_desc_t &desc_;

    void (*ker_)(const jit_eltwise_args *);
    void operator()(const jit_eltwise_args *args) { assert(ker_); ker_(args); }

    jit_uni_eltwise_kernel_f32(const eltwise_desc_t &desc)
        : desc_(desc), ker_(nullptr) {}
    virtual ~jit_uni_eltwise_kernel_f32() {}

protected:
    bool is_bwd() const { return desc_.prop_kind == prop_kind::backward_data; }
};

/* jit kernels */
namespace {

template <cpu_isa_t isa>
struct jit_uni_relu_kernel_f32 : public jit_uni_eltwise_kernel_f32,
    public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_relu_kernel_f32)

    void compute_step(bool vectorize, const int uf, const int shift) {
        for (int i = 0; i < uf; i++) {
            auto addr_fwd = ptr[reg_from + i * shift];
            auto addr_bwd = ptr[reg_for_comparison + i * shift];
            if (vectorize) {
                if (is_bf16_) {
                    vmovups(Ymm_src(i + 1), addr_fwd);
                    vpermw(Vmm(i + 1) | k_mask_cvt | T_z, zmm_idx, Zmm_src(i + 1));
                } else {
                    uni_vmovups(Vmm(i + 1), addr_fwd);
                }
                if (is_bwd()) {
                    if (is_bf16_) {
                        vmovups(Ymm_src(uf + i + 1), addr_bwd);
                        vpermw(Vmm(uf + i + 1) | k_mask_cvt | T_z,
                                zmm_idx, Zmm_src(uf + i + 1));
                    } else {
                        uni_vmovups(Vmm(uf + i + 1), addr_bwd);
                    }
                }
            } else {
                if (is_bf16_) {
                    vmovdqu16(Ymm_src(i + 1) | k_tail_mask, addr_fwd);
                    vpermw(Vmm(i + 1) | k_mask_cvt | T_z, zmm_idx,
                            Zmm_src(i + 1));
                } else {
                    movss(Xmm(i + 1), addr_fwd);
                }
                if (is_bwd()) {
                    if (is_bf16_) {
                        vmovdqu16(Ymm_src(uf + i + 1) | k_tail_mask, addr_bwd);
                        vpermw(Vmm(uf + i + 1) | k_mask_cvt | T_z, zmm_idx,
                                Zmm_src(uf + i + 1));
                    } else {
                        movss(Xmm(uf + i + 1), addr_bwd);
                    }
                }
            }
        }

        if (isa == sse42) {
            for (int i = 0; i < uf; i++) {
                movups(Vmm(2 * uf + i + 1), Vmm(i + 1));
                mulps(Vmm(2 * uf + i + 1), vmm_ns);

                Vmm mask = Vmm(0);
                if (is_bwd()) {
                    movups(mask, Vmm(uf + i + 1));
                    cmpps(mask, vmm_zero, _cmp_nle_us);
                } else {
                    movups(mask, Vmm(i + 1));
                    cmpps(mask, vmm_zero, _cmp_nle_us);
                }
                blendvps(Vmm(2 * uf + i + 1), Vmm(i + 1));
            }
        } else {
            for (int i = 0; i < uf; i++) {
                vmulps(Vmm(2 * uf + i + 1), Vmm(i + 1), vmm_ns);
                if (isa == avx2) {
                    if (is_bwd())
                        vcmpgtps(vmm_mask, Vmm(uf + i + 1), vmm_zero);
                    else
                        vcmpgtps(vmm_mask, Vmm(i + 1), vmm_zero);

                    vblendvps(Vmm(2 * uf + i + 1), Vmm(2 * uf + i + 1),
                              Vmm(i + 1), vmm_mask);

                } else {
                    if (is_bwd())
                        vcmpps(k_mask, Vmm(uf + i + 1), vmm_zero, _cmp_nle_us);
                    else
                        vcmpps(k_mask, Vmm(i + 1), vmm_zero, _cmp_nle_us);
                    vblendmps(Vmm(2 * uf + i + 1) | k_mask, Vmm(2 * uf + i + 1),
                              Vmm(i + 1));
                }
            }
        }
        auto store_data =[&] (opmask_t _kmask, int i) {
            if (!mayiuse(avx512_core_bf16))
                bf16_emu_->r_vcvtneps2bf16(Ymm_src(2 * uf + i + 1),
                    Zmm(2 * uf + i + 1));
            else
                vcvtneps2bf16(Ymm_src(2 * uf + i + 1), Vmm(2 * uf + i + 1));
            vmovdqu16(ptr[reg_to + i * shift] | _kmask, Ymm_src(2 * uf + i + 1));
        };

        for (int i = 0; i < uf; i++) {
            if (vectorize)
                if(is_bf16_)
                    store_data(k_full_mask, i);
                else
                    uni_vmovups(ptr[reg_to + i * shift], Vmm(2 * uf + i + 1));
            else
                if (is_bf16_)
                    store_data(k_tail_mask, i);
                else
                    movss(ptr[reg_to + i * shift], Xmm(2 * uf + i + 1));
        }
    }

    ~jit_uni_relu_kernel_f32() { delete bf16_emu_; }

    jit_uni_relu_kernel_f32(const eltwise_desc_t &desc)
        : jit_uni_eltwise_kernel_f32(desc)
        , jit_generator()
        , bf16_emu_(nullptr) {
        assert(desc.alg_kind == alg_kind::eltwise_relu);
        assert(isa == sse42 || isa == avx2 || isa == avx512_common);

        Reg64 param = abi_param1;

        is_bf16_ = (desc.data_desc.data_type == data_type::bf16);
        if (!mayiuse(avx512_core_bf16) && is_bf16_)
            bf16_emu_ = new bf16_emulation_t(this,
                    bf16_emu_reserv_1, bf16_emu_reserv_2,
                    bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_6);

        const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
        const int loop_dec[] = {simd_w, 1};
        const int uf[] = {1, 1};

        int _shift = (is_bf16_) ? sizeof(mkldnn_bfloat16_t) : sizeof(float);
        int _vlen = (is_bf16_)
            ? cpu_isa_traits<isa>::vlen / 2
            : cpu_isa_traits<isa>::vlen;

        const int shift[] = {_vlen, _shift};
        const bool loop_vectorize[] = {true, false};

        preamble();

        if (is_bf16_) {
            mov(mask_reg, 0xAAAAAAAA);
            kmovd(k_mask_cvt, mask_reg);

            mov(mask_reg, 0x1);
            kmovd(k_tail_mask, mask_reg);

            mov(mask_reg, 0xffff);
            kmovd(k_full_mask, mask_reg);
        }
        if (!mayiuse(avx512_core_bf16) && is_bf16_)
            bf16_emu_->init_vcvtneps2bf16();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        if (is_bwd())
            mov(reg_for_comparison, ptr[param + GET_OFF(for_comparison)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        if (is_bf16_) {
            mov(p_idx_table, idx_table);
            vmovups(zmm_idx, ptr[p_idx_table]);
        }

        mov(imm_addr64, float2int(desc.alpha));
        movq(xmm_ns, imm_addr64);
        uni_vbroadcastss(vmm_ns, xmm_ns);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        Label loop_label[3];

        for (int id = 0; id < 2; id++) {
            L(loop_label[id]);
            cmp(reg_work_amount, uf[id] * loop_dec[id] - 1);
            jle(loop_label[id + 1], T_NEAR);

            compute_step(loop_vectorize[id], uf[id], shift[id]);

            add(reg_from, uf[id] * shift[id]);
            add(reg_to, uf[id] * shift[id]);
            if (is_bwd())
                add(reg_for_comparison, uf[id] * shift[id]);

            sub(reg_work_amount, uf[id] * loop_dec[id]);
            jmp(loop_label[id]);
        }

        L(loop_label[2]);
        postamble();

        if (is_bf16_) {
            align(64);
            L(idx_table);
            const uint16_t _idx[] = { 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,
                                      9,9,10,10,11,11,12,12,13,13,14,14,15,15 };
            for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
                dw(_idx[i]);
        }

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                                             isa == avx2, Ymm, Zmm>::type;
    using opmask_t = const Xbyak::Opmask;

    Reg64 reg_from = rax;
    Reg64 reg_for_comparison = is_bwd() ? rdx : reg_from;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;

    Reg32 mask_reg = r14d;
    Reg32 reg32_tmp = mask_reg;
    Reg64 reg_idx = r15;
    Reg64 p_idx_table = r13;

    Xmm xmm_ns = Xmm(14);

    Vmm vmm_ns = Vmm(isa == avx512_common ? 30 : 14);
    Vmm vmm_zero = Vmm(isa == avx512_common ? 31 : 15);

    Vmm vmm_mask = Vmm(isa == avx512_common ? 28 : 12);
    Opmask k_mask = Opmask(1);

    inline Ymm Ymm_src(int i) {
        return Ymm(15 + i);
    }
    inline Zmm Zmm_src(int i) {
        return Zmm(15 + i);
    }
    Zmm zmm_idx = Zmm(29);

    Zmm bf16_emu_reserv_1 = Zmm(24);
    Zmm bf16_emu_reserv_2 = Zmm(25);
    Zmm bf16_emu_reserv_3 = Zmm(26);
    Reg64 bf16_emu_reserv_4 = r14;
    Zmm bf16_emu_reserv_5 = Zmm(27);
    Zmm bf16_emu_reserv_6 = Zmm(27);

    opmask_t k_mask_cvt = k7;
    opmask_t k_tail_mask = k6;
    opmask_t k_full_mask = k5;

    Label idx_table;

    bool is_bf16_;

    bf16_emulation_t *bf16_emu_;
};

template <cpu_isa_t isa>
struct jit_uni_kernel_fwd_f32: public jit_uni_eltwise_kernel_f32,
    public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_kernel_fwd_f32)

    jit_uni_kernel_fwd_f32(const eltwise_desc_t &desc)
        : jit_uni_eltwise_kernel_f32(desc)
        , jit_generator()
        , bf16_emu_(nullptr) {

        bool is_bf16_ = (desc.data_desc.data_type == data_type::bf16);

        if (!mayiuse(avx512_core_bf16) && is_bf16_)
            bf16_emu_ = new bf16_emulation_t(this,
                    bf16_emu_reserv_1, bf16_emu_reserv_2,
                    bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5, bf16_emu_reserv_6);

        eltwise_injector_ = new jit_uni_eltwise_injector_f32<isa>(this,
                desc.alg_kind, desc.alpha, desc.beta, false, r9, Opmask(1));

        using namespace alg_kind;

        assert(is_bwd() == false);
        assert(utils::one_of(desc.alg_kind, eltwise_tanh, eltwise_elu,
                    eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
                    eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
                    eltwise_exp, eltwise_gelu, eltwise_clamp, eltwise_swish));

        preamble();

        if (is_bf16_) {
            mov(mask_reg, 0xAAAAAAAA);
            kmovd(k_mask, mask_reg);

            mov(mask_reg, 0x1);
            kmovd(k_tail_mask, mask_reg);

            mov(mask_reg, 0xffff);
            kmovd(k_full_mask, mask_reg);
        }
        if (!mayiuse(avx512_core_bf16) && is_bf16_)
            bf16_emu_->init_vcvtneps2bf16();

        Reg64 param = abi_param1;
        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        if (is_bf16_) {
            mov(p_idx_table, idx_table);
            vmovups(zmm_idx, ptr[p_idx_table]);
        }
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        eltwise_injector_->load_table_addr();

        Label reminder_loop_start, reminder_loop_end;
        Label vectorized_loop_start, vectorized_loop_end;

        cmp(reg_work_amount, simd_w);
        jl(reminder_loop_start, T_NEAR);

        L(vectorized_loop_start);

        auto store_data =[&] (opmask_t _kmask) {
            if (!mayiuse(avx512_core_bf16))
                bf16_emu_->r_vcvtneps2bf16(ymm_src, zmm_src_1);
            else
                vcvtneps2bf16(ymm_src, vmm_src);
            vmovdqu16(ptr[reg_to] | _kmask, ymm_src);
        };

        if (is_bf16_) {
            vmovups(ymm_src, ptr[reg_from]);
            vpermw(vmm_src | k_mask  | T_z, zmm_idx, zmm_src);
            eltwise_injector_->compute_vector(vmm_src.getIdx());
            store_data(k_full_mask);
        } else {
            uni_vmovups(vmm_src, ptr[reg_from]);
            eltwise_injector_->compute_vector(vmm_src.getIdx());
            uni_vmovups(ptr[reg_to], vmm_src);
        }
        auto shift = (is_bf16_) ? vlen / 2 : vlen;
        add(reg_from, shift);
        add(reg_to, shift);

        sub(reg_work_amount, simd_w);
        cmp(reg_work_amount, simd_w);
        jge(vectorized_loop_start, T_NEAR);

        L(vectorized_loop_end);

        L(reminder_loop_start);

        cmp(reg_work_amount, 0);
        jle(reminder_loop_end, T_NEAR);
        if (is_bf16_) {
            vmovups(ymm_src | k_tail_mask, ptr[reg_from]);
            vpermw(vmm_src | k_mask | T_z, zmm_idx, zmm_src);
            eltwise_injector_->compute_vector(vmm_src.getIdx());
            store_data(k_tail_mask);
        } else {
            movss(xmm_src, ptr[reg_from]);
            eltwise_injector_->compute_vector(xmm_src.getIdx());
            movss(ptr[reg_to], xmm_src);
        }
        auto size_step = (is_bf16_) ? sizeof(mkldnn_bfloat16_t) : sizeof(float);
        add(reg_from, size_step);
        add(reg_to, size_step);

        dec(reg_work_amount);
        jmp(reminder_loop_start, T_NEAR);

        L(reminder_loop_end);

        postamble();

        eltwise_injector_->prepare_table();

        if (is_bf16_) {
            align(64);
            L(idx_table);
            const uint16_t _idx[] = { 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,
                                      9,9,10,10,11,11,12,12,13,13,14,14,15,15 };
            for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
                dw(_idx[i]);
        }

        ker_ = (decltype(ker_))this->getCode();
    }

    ~jit_uni_kernel_fwd_f32() {
        delete eltwise_injector_;
        delete bf16_emu_;
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                isa == avx2, Ymm, Zmm>::type;
    using opmask_t = const Xbyak::Opmask;

    const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int vlen   = cpu_isa_traits<isa>::vlen;

    Reg64 reg_from = rax;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;
    Reg32 mask_reg = edx;
    Reg64 reg_idx = r15;
    Reg32 reg32_tmp = r14d;
    Reg64 p_idx_table = r13;

    Xmm xmm_src = Xmm(1);
    Vmm vmm_src = Vmm(1);
    Zmm zmm_src_1 = Zmm(1);

    Ymm ymm_src = Ymm(30);
    Zmm zmm_src = Zmm(30);
    Zmm zmm_idx = Zmm(31);

    Zmm bf16_emu_reserv_1 = Zmm(26);
    Zmm bf16_emu_reserv_2 = Zmm(27);
    Zmm bf16_emu_reserv_3 = Zmm(28);
    Reg64 bf16_emu_reserv_4 = r14;
    Zmm bf16_emu_reserv_5 = Zmm(29);
    Zmm bf16_emu_reserv_6 = Zmm(29);

    opmask_t k_mask = k7;
    opmask_t k_tail_mask = k6;
    opmask_t k_full_mask = k5;

    Label idx_table;

    jit_uni_eltwise_injector_f32<isa> *eltwise_injector_;
    bf16_emulation_t *bf16_emu_;
};

} /* namespace */

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::pd_t::init() {
    using namespace alg_kind;

    assert(engine()->kind() == engine_kind::cpu);
    bool ok = true && mayiuse(isa)
        && utils::one_of(desc()->prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference)
        && desc()->data_desc.data_type == d_type
        && !has_zero_dim_memory()
        && utils::one_of(desc()->alg_kind, eltwise_relu, eltwise_tanh,
                eltwise_elu, eltwise_square, eltwise_abs, eltwise_sqrt,
                eltwise_linear, eltwise_bounded_relu, eltwise_soft_relu,
                eltwise_logistic, eltwise_exp, eltwise_gelu, eltwise_clamp, eltwise_swish)
        && memory_desc_wrapper(src_pd()).is_dense(true)
        && IMPLICATION(!memory_desc_wrapper(src_pd()).is_dense(false),
                math::eltwise_fwd_preserves_zero(desc()->alg_kind, true))
        && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::jit_uni_eltwise_fwd_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs), kernel_(nullptr) {
    const auto &desc = *pd()->desc();
    switch (desc.alg_kind) {
    case alg_kind::eltwise_relu:
        kernel_ = new jit_uni_relu_kernel_f32<isa>(desc); break;
    default:
        kernel_ = new jit_uni_kernel_fwd_f32<isa>(desc);
    }
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::~jit_uni_eltwise_fwd_t()
{ delete kernel_; }

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_eltwise_fwd_t<isa, d_type>::execute_forward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper data_d(pd()->src_pd());

    const size_t nelems = data_d.nelems(true);

    src += data_d.blocking_desc().offset_padding;
    dst += data_d.blocking_desc().offset_padding;

    const int cache_line = 16;
    parallel(0, utils::div_up(nelems, cache_line), [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        auto arg = jit_eltwise_args();
        arg.from = (const void*)&src[start];
        arg.for_comparison = (const void*)&src[start];
        arg.to = (const void*)&dst[start];
        arg.work_amount = end - start;
        if (arg.work_amount)
            (*kernel_)(&arg);
    });
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::pd_t::init() {
    assert(engine()->kind() == engine_kind::cpu);

    bool ok = true
        && desc()->prop_kind == prop_kind::backward_data
        && utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu)
        && src_pd()->desc()->data_type == d_type
        && !has_zero_dim_memory()
        && mayiuse(isa)
        && memory_desc_wrapper(src_pd()).is_dense()
        && memory_desc_wrapper(diff_dst_pd()) == memory_desc_wrapper(src_pd())
        && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::jit_uni_eltwise_bwd_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs), kernel_(nullptr) {
    const auto &desc = *pd()->desc();
    switch (desc.alg_kind) {
    case alg_kind::eltwise_relu:
        kernel_ = new jit_uni_relu_kernel_f32<isa>(desc); break;
    default: assert(!"unknown eltwise alg_kind");
    }
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::~jit_uni_eltwise_bwd_t()
{ delete kernel_; }

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_eltwise_bwd_t<isa, d_type>::execute_backward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper data_d(pd()->src_pd());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_pd());

    const size_t nelems = data_d.nelems();

    src += data_d.blocking_desc().offset_padding;
    diff_dst += diff_data_d.blocking_desc().offset_padding;
    diff_src += diff_data_d.blocking_desc().offset_padding;

    const int cache_line = 16;

    parallel(0, utils::div_up(nelems, cache_line), [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        auto arg = jit_eltwise_args();
        arg.from = (const void*)&diff_dst[start];
        arg.to = (const void*)&diff_src[start];
        arg.for_comparison = (const void*)&src[start];
        arg.work_amount = end - start;
        if (arg.work_amount) {
            (*kernel_)(&arg);
        }
    });
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::register_table_entries() {
    // This function is responsible to pick all necessary constants
    // for a given algorithm, compute right offset for them to be used
    // in table_val() and save the hexadecimal value of them, which
    // will be finally used in prepare_table(). We rely on fact that
    // the map iterator order is deterministic for a fixed map.

    // common values used in several algorithms
    static const table_t common_values {{zero, {0x00000000, true}},
            {half, {0x3f000000, true}}, {one, {0x3f800000, true}},
            {two, {0x40000000, true}}, {minus_one, {0xbf800000, true}},
            {minus_two, {0xc0000000, true}}, {ln2f, {0x3f317218, true}},
            {positive_mask, {0x7fffffff, true}},
            {sign_mask, {0x80000000, true}},
            {exponent_bias, {0x0000007f, true}}};

    // log(x) constants
    static const table_t log_consts {
            {log_minus_inf, {0xff800000, true}},
            {log_qnan, {0x7fc00000, true}},
            {log_mantissa_mask, {0x007fffff, true}},
            {log_full_k_reg_mask, {0x0000ffff, true}},
            {log_five_bit_offset, {0x0000001f, true}},
    };

    // log(x) polynomial approximation
    static const table_t log_polynomial {
            {log_pol, {0xbf000000, true}}, // p1 = -0.5f
            {log_pol, {0x3eaaaaab, true}}, // p2 =  0.333333343f
            {log_pol, {0xbe8004ab, true}}, // p3 = -0.250035613f
            {log_pol, {0x3e4cc8a3, true}}, // p4 =  0.199984118f
    };

    // log(x) pre-defined values. First goes index}, then val[index].
    static const table_t log_predefined_values {
            {log_predefined_vals, {0x3f800000, true}}, //  0: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, //  1: -88.029693603515625
            {log_predefined_vals, {0x3f780000, true}}, //  2: 0.96875
            {log_predefined_vals,
                    {0xc2affef2, true}}, //  3: -87.9979400634765625
            {log_predefined_vals, {0x3f700000, true}}, //  4: 0.9375
            {log_predefined_vals,
                    {0xc2afee29, true}}, //  5: -87.9651565551757812
            {log_predefined_vals, {0x3f680000, true}}, //  6: 0.90625
            {log_predefined_vals,
                    {0xc2afdccd, true}}, //  7: -87.9312515258789062
            {log_predefined_vals, {0x3f600000, true}}, //  8: 0.875
            {log_predefined_vals,
                    {0xc2afcad6, true}}, //  9: -87.8961639404296875
            {log_predefined_vals, {0x3f580000, true}}, // 10: 0.84375
            {log_predefined_vals,
                    {0xc2afb837, true}}, // 11: -87.859794616699218
            {log_predefined_vals, {0x3f580000, true}}, // 12: 0.84375
            {log_predefined_vals,
                    {0xc2afb837, true}}, // 13: -87.859794616699218
            {log_predefined_vals, {0x3f500000, true}}, // 14: 0.8125
            {log_predefined_vals,
                    {0xc2afa4e4, true}}, // 15: -87.822052001953125
            {log_predefined_vals, {0x3f480000, true}}, // 16: 0.78125
            {log_predefined_vals,
                    {0xc2af90cf, true}}, // 17: -87.782829284667968
            {log_predefined_vals, {0x3f480000, true}}, // 18: 0.78125
            {log_predefined_vals,
                    {0xc2af90cf, true}}, // 19: -87.782829284667968
            {log_predefined_vals, {0x3f400000, true}}, // 20: 0.75
            {log_predefined_vals,
                    {0xc2af7be9, true}}, // 21: -87.742012023925781
            {log_predefined_vals, {0x3f400000, true}}, // 22: 0.75
            {log_predefined_vals,
                    {0xc2af7be9, true}}, // 23: -87.742012023925781
            {log_predefined_vals, {0x3f380000, true}}, // 24: 0.71875
            {log_predefined_vals,
                    {0xc2af661e, true}}, // 25: -87.699447631835937
            {log_predefined_vals, {0x3f380000, true}}, // 26: 0.71875
            {log_predefined_vals,
                    {0xc2af661e, true}}, // 27: -87.699447631835937
            {log_predefined_vals, {0x3f300000, true}}, // 28: 0.6875
            {log_predefined_vals,
                    {0xc2af4f5c, true}}, // 29: -87.654998779296875
            {log_predefined_vals, {0x3f300000, true}}, // 30: 0.6875
            {log_predefined_vals,
                    {0xc2af4f5c, true}}, // 31: -87.654998779296875
            {log_predefined_vals, {0x3fa80000, true}}, // 32: 1.3125
            {log_predefined_vals,
                    {0xc2b09a6f, true}}, // 33: -88.301628112792968
            {log_predefined_vals, {0x3fa80000, true}}, // 34: 1.3125
            {log_predefined_vals,
                    {0xc2b09a6f, true}}, // 35: -88.301628112792968
            {log_predefined_vals, {0x3fa00000, true}}, // 36: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 37: -88.252838134765625
            {log_predefined_vals, {0x3fa00000, true}}, // 38: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 39: -88.252838134765625
            {log_predefined_vals, {0x3fa00000, true}}, // 40: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 41: -88.252838134765625
            {log_predefined_vals, {0x3f980000, true}}, // 42: 1.1875
            {log_predefined_vals,
                    {0xc2b06731, true}}, // 43: -88.201545715332031
            {log_predefined_vals, {0x3f980000, true}}, // 44: 1.1875
            {log_predefined_vals,
                    {0xc2b06731, true}}, // 45: -88.201545715332031
            {log_predefined_vals, {0x3f900000, true}}, // 46: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 47: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 48: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 49: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 50: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 51: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 52: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 53: -88.147476196289062
            {log_predefined_vals, {0x3f880000, true}}, // 54: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 55: -88.090316772460937
            {log_predefined_vals, {0x3f880000, true}}, // 56: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 57: -88.090316772460937
            {log_predefined_vals, {0x3f880000, true}}, // 58: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 59: -88.090316772460937
            {log_predefined_vals, {0x3f800000, true}}, // 60: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, // 61: -88.029693603515625
            {log_predefined_vals, {0x3f800000, true}}, // 62: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, // 63: -88.029693603515625
    };

    // This object takes care about which constants and polynomials to include.
    struct need_t {
        need_t(alg_kind_t alg) {
            using namespace alg_kind;
            switch (alg) {
                case eltwise_log: log_ = true; break;
                default: break;
            }
        }

        bool log_ = false;

        bool log() const { return log_; }
    };

    need_t need(alg_);

    auto push_arg_entry_of = [&](const key_t key, const table_entry_val_t val,
                                     const bool broadcast) {
        mapped_table_entry_t te {0, val, broadcast};
        entry_map_.insert(std::make_pair(key, te));
    };

    auto push_entries_of = [&](const table_t &t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto key = (*it).first;
            auto te = (*it).second; // copy values from table
            push_arg_entry_of(key, te.val, te.bcast);
        }
    };

    push_arg_entry_of(alpha, float2int(alpha_), true);
    push_arg_entry_of(beta, float2int(beta_), true);
    push_entries_of(common_values);
    if (need.log()) push_entries_of(log_consts);
    if (need.log()) push_entries_of(log_polynomial);
    if (need.log()) push_entries_of(log_predefined_values);

    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        auto &te = (*it).second;
        te.off = off;
        off += te.bcast ? vlen : sizeof(table_entry_val_t);
    }
}

template struct jit_uni_eltwise_fwd_t<sse42, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<sse42, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx2, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx2, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx512_common, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<avx512_common, data_type::bf16>;
template struct jit_uni_eltwise_bwd_t<avx512_common, data_type::f32>;
template struct jit_uni_eltwise_bwd_t<avx512_common, data_type::bf16>;

}
}
}
