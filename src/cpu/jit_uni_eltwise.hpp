/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef CPU_JIT_UNI_ELTWISE_HPP
#define CPU_JIT_UNI_ELTWISE_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_eltwise_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_eltwise_injector_f32 {
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    jit_uni_eltwise_injector_f32(jit_generator *host, alg_kind_t alg,
            float alpha, float beta, bool save_state = true,
            Xbyak::Reg64 p_table = Xbyak::util::rax,
            Xbyak::Opmask k_mask = Xbyak::Opmask(1))
        : alg_(alg), alpha_(alpha), beta_(beta), h(host)
        , save_state_(save_state), p_table(p_table), k_mask(k_mask)
    {
        using namespace alg_kind;
        assert(utils::one_of(isa, sse42, avx2, avx512_common));
        assert(utils::one_of(alg_, eltwise_relu, eltwise_tanh, eltwise_elu,
                    eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
                    eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
                    eltwise_exp, eltwise_gelu, eltwise_clamp, eltwise_swish,
                    eltwise_log));
        register_table_entries();
    }

    // note that eltwise.scale is ignored
    jit_uni_eltwise_injector_f32(jit_generator *host,
            const post_ops_t::entry_t::eltwise_t &eltwise,
            bool save_state = true, Xbyak::Reg64 p_table = Xbyak::util::rax,
            Xbyak::Opmask k_mask = Xbyak::Opmask(1))
        : jit_uni_eltwise_injector_f32(host, eltwise.alg, eltwise.alpha,
                eltwise.beta, save_state, p_table, k_mask) {}

    void compute_vector_range(size_t start_idx, size_t end_idx);
    void compute_vector(size_t idx) { compute_vector_range(idx, idx + 1); }
    void prepare_table(bool gen_table=true);
    void load_table_addr() { h->mov(p_table, l_table); }

    const alg_kind_t alg_;
    const float alpha_;
    const float beta_;

    jit_generator * const h;

    const bool save_state_;
    const Xbyak::Reg64 p_table;
    const Xbyak::Opmask k_mask;
    Xbyak::Label l_table;

private:
    // if only the injector was inherited from jit_generator...
    enum {
        _cmp_eq_oq = jit_generator::_cmp_eq_oq,
        _cmp_lt_os = jit_generator::_cmp_lt_os,
        _cmp_le_os = jit_generator::_cmp_le_os,
        _cmp_nle_us = jit_generator::_cmp_nle_us,
        _op_floor = jit_generator::_op_floor,
    };

    static constexpr bool has_avx512() {
        return utils::one_of(isa, avx512_common, avx512_core);
    }

    size_t vlen = cpu_isa_traits<isa>::vlen;

    const static size_t preserved_vecs_max = 5;

    static constexpr int n_mantissa_bits = 23;

    size_t vecs_to_preserve = 0;
    size_t vecs_count = isa == avx512_common ? 32 : 16;
    size_t preserved_vecs_count = 0;
    size_t preserved_vec_idxs[preserved_vecs_max] = {0};
    size_t start_idx_tail = 0;

    Vmm vmm_mask, vmm_aux0, vmm_aux1, vmm_aux2, vmm_aux3, vmm_aux4;

    Xbyak::Address table_val(int index)
    { return h->ptr[p_table + index * vlen]; }

    int aux_vecs_count(alg_kind_t alg);
    void compute_body(size_t start_idx, size_t end_idx);
    void injector_preamble(size_t start_idx, size_t end_idx);
    void injector_preamble_tail(size_t start_idx);
    void injector_postamble();
    void assign_regs();
    void compute_cmp_mask(const Vmm &vmm_src,
            const Xbyak::Operand &compare_operand, int cmp_predicate);
    void blend_with_mask(const Vmm &vmm_dst, const Xbyak::Operand &src);
    void test_mask();

    void exp_compute_vector(const Vmm &vmm_src);
    void relu_compute_vector(const Vmm &vmm_src);
    void relu_zero_ns_compute_vector(const Vmm &vmm_src);
    void elu_compute_vector(const Vmm &vmm_src);
    void tanh_compute_vector(const Vmm &vmm_src);
    void square_compute_vector(const Vmm &vmm_src);
    void abs_compute_vector(const Vmm &vmm_src);
    void sqrt_compute_vector(const Vmm &vmm_src);
    void linear_compute_vector(const Vmm &vmm_src);
    void bounded_relu_compute_vector(const Vmm &vmm_src);
    void soft_relu_compute_vector(const Vmm &vmm_src);
    void logistic_compute_vector(const Vmm &vmm_src);
    void gelu_compute_vector(const Vmm &vmm_src);
    void clamp_compute_vector(const Vmm &vmm_src);
    void swish_compute_vector(const Vmm &vmm_src);
    void log_compute_vector(const Vmm &vmm_src);

    void relu_prepare_table();
    void elu_prepare_table();
    void soft_relu_prepare_table();
    void abs_prepare_table();
    void sqrt_prepare_table();
    void linear_prepare_table();
    void bounded_relu_prepare_table();
    void clamp_prepare_table();
    void log_prepare_table();

    enum key_t {
        alpha = 0, // alpha argument
        beta, // beta argument
        zero, // 0.f
        half, // 0.5f
        one, // 1.f  or  mask for exponent bits
        two, // 2.f
        minus_one, // -1.f  or  changes sign to opposite
        minus_two, // -2.f
        ln2f, // 0.69314718f
        positive_mask, // changes sign to positive
        sign_mask, // gets sign value
        exponent_bias, // (127 = 2^7 - 1), gets exponent bits
        exp_log2ef, // 1.44269502f - formula-based for approx
        exp_ln_flt_max_f, // logf(FLT_MAX) - max normal value
        exp_ln_flt_min_f, // logf(FLT_MIN) - min normal value
        exp_pol, // see correspondent table for float values
        tanh_idx_bias, // bias applied during index computation
        tanh_idx_mask, // mask applied to extract index
        tanh_linear_ubound, // arg below which tanh(x) = x
        tanh_saturation_lbound, // arg after which tanh(x) = 1.f
        tanh_pol_table, // table of polynomial coefficients
        soft_relu_one_twenty_six, // 126.f
        soft_relu_mantissa_sign_mask, // mask for mantissa bits and sign
        soft_relu_pol, // see correspondent table for float values
        gelu_tanh_fitting_const, // 0.044715f
        gelu_tanh_fitting_const_times_three, // 0.134145f
        gelu_tanh_sqrt_two_over_pi, // sqrtf(2.f/pi) = 0.797884f
        gelu_erf_approx_const, // 0.3275911f - implementation based for approx
        gelu_erf_one_over_sqrt_two, // 1.f / sqrtf(2.f)
        gelu_erf_one_over_sqrt_pi, // 1.f / sqrtf(pi) = 0.564190f
        gelu_erf_pol, // see correspondent table for float values
        log_minus_inf, // -inf
        log_qnan, // qnan
        log_mantissa_mask, // gets mantissa bits
        log_full_k_reg_mask, // sets k_register with all bits of 1
        log_full_vector_reg_mask, // sets vector register will all bits of 1
        log_five_bit_offset, // 5 bits off (31 = 2^5 - 1)
        log_pol, // see correspondent table for float values
        log_predefined_vals, // see correspondent table for float values
        undef_key,
    };

    size_t table_off(key_t key, size_t key_off_val_shift = 0) {
        // assumption: all table entries sharing the same key also
        // share their broadcast property
        // TODO: enforce through data structure
        const auto it = entry_map_.find(key); // search an entry for a key
        assert(it != entry_map_.end());
        const auto &te = (*it).second;
        const auto scale = te.bcast ? vlen : sizeof(table_entry_val_t);
        return te.off + key_off_val_shift * scale;
    }
    Xbyak::Address table_val(key_t key, size_t key_off_val_shift = 0) {
        auto off = table_off(key, key_off_val_shift);
        return h->ptr[p_table + off];
    }

    // we accept only 32bit hexadecimal table values to avoid any rounding
    using table_entry_val_t = uint32_t;
    using table_entry_offset_t = size_t; // offsets are in bytes wrt p_table
    using table_entry_bcast_t = bool; // true => bcast value

    struct table_entry_t {
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };
    struct mapped_table_entry_t {
        table_entry_offset_t off;
        table_entry_val_t val;
        table_entry_bcast_t bcast;
    };

    using table_t = std::multimap<key_t, table_entry_t>;
    using mapped_table_t = std::multimap<key_t, mapped_table_entry_t>;

    void register_table_entries();
    mapped_table_t entry_map_;
};

struct jit_uni_eltwise_kernel_f32;

template <cpu_isa_t isa, impl::data_type_t d_type>
struct jit_uni_eltwise_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_eltwise_fwd_pd_t {
        pd_t(engine_t *engine, const eltwise_desc_t *adesc,
                const primitive_attr_t *attr,
                const eltwise_fwd_pd_t *hint_fwd_pd)
            : cpu_eltwise_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_eltwise_fwd_t<isa, d_type>);

        virtual status_t init() override;
    };

    jit_uni_eltwise_fwd_t(const pd_t *apd, const input_vector &inputs,
                       const output_vector &outputs);
    ~jit_uni_eltwise_fwd_t();

    typedef typename prec_traits<d_type>::type data_t;

    virtual void execute(event_t *e) const
    {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_uni_eltwise_kernel_f32 *kernel_;
};

template <cpu_isa_t isa, impl::data_type_t d_type>
struct jit_uni_eltwise_bwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_eltwise_bwd_pd_t {
        pd_t(engine_t *engine, const eltwise_desc_t *adesc,
                const primitive_attr_t *attr,
                const eltwise_fwd_pd_t *hint_fwd_pd)
            : cpu_eltwise_bwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_eltwise_bwd_t<isa, d_type>);

        virtual status_t init() override;
    };

    jit_uni_eltwise_bwd_t(const pd_t *apd, const input_vector &inputs,
                       const output_vector &outputs);
    ~jit_uni_eltwise_bwd_t();

    typedef typename prec_traits<d_type>::type data_t;

    virtual void execute(event_t *e) const
    {
        execute_backward();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_uni_eltwise_kernel_f32 *kernel_;
};

}
}
}

#endif
