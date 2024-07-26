#ifndef CPU_X64_INJECTORS_JIT_CHANNEL_LOAD_HPP
#define CPU_X64_INJECTORS_JIT_CHANNEL_LOAD_HPP

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename Vmm>
void load_channel_vector(jit_generator_t *host, const Vmm &dst,
        const Xbyak::Reg64 &src, int offset, const Xbyak::RegExp &base_ptr,
        size_t field_offset, dim_t count) {
    const int lanes = vreg_traits_t<Vmm>::vlen / sizeof(float);
    const Xbyak::Reg64 candidates[] = {host->rax, host->rcx, host->rdx,
            host->r8, host->r9, host->r10, host->r11};
    int first = -1, second = -1;
    for (int idx = 0; idx < 7; ++idx) {
        const int reg = candidates[idx].getIdx();
        if (reg == src.getIdx()
                || (base_ptr.getBase().getBit() && reg == base_ptr.getBase().getIdx())
                || (base_ptr.getIndex().getBit() && reg == base_ptr.getIndex().getIdx()))
            continue;
        if (first < 0) first = idx;
        else { second = idx; break; }
    }
    assert(first >= 0 && second >= 0 && count > 0);
    const auto end = candidates[first];
    const auto tmp = candidates[second];
    const bool stack_base = base_ptr.getBase().getBit()
            && base_ptr.getBase().getIdx() == host->rsp.getIdx();
    Xbyak::Label full, done;

    host->push(end);
    host->push(tmp);
    host->mov(end, host->ptr[base_ptr + (stack_base ? 2 * sizeof(size_t) : 0)]);
    host->add(end, field_offset + count * sizeof(float));
    host->lea(tmp, host->ptr[src + offset + (lanes - 1) * sizeof(float)]);
    host->cmp(tmp, end);
    host->jb(full, Xbyak::CodeGenerator::T_NEAR);

    host->sub(host->rsp, lanes * sizeof(float));
    for (int lane = 0; lane < lanes; ++lane) {
        Xbyak::Label skip;
        host->mov(host->dword[host->rsp + lane * sizeof(float)], 0);
        host->lea(tmp, host->ptr[src + offset + lane * sizeof(float)]);
        host->cmp(tmp, end);
        host->jae(skip, Xbyak::CodeGenerator::T_NEAR);
        host->mov(tmp.cvt32(), host->dword[tmp]);
        host->mov(host->dword[host->rsp + lane * sizeof(float)], tmp.cvt32());
        host->L(skip);
    }
    host->uni_vmovups(dst, host->ptr[host->rsp]);
    host->add(host->rsp, lanes * sizeof(float));
    host->jmp(done, Xbyak::CodeGenerator::T_NEAR);

    host->L(full);
    host->uni_vmovups(dst, host->ptr[src + offset]);
    host->L(done);
    host->pop(tmp);
    host->pop(end);
}

template <typename Vmm>
void load_channel_vector(jit_generator_t *host, const Vmm &dst,
        const Xbyak::Reg64 &src, int offset, const float *base, dim_t count) {
    const int lanes = vreg_traits_t<Vmm>::vlen / sizeof(float);
    const Xbyak::Reg64 end = src.getIdx() == host->rax.getIdx() ? host->rcx : host->rax;
    const Xbyak::Reg64 tmp = src.getIdx() == host->rdx.getIdx() ? host->r8 : host->rdx;
    Xbyak::Label full, done;

    host->push(end);
    host->push(tmp);
    host->mov(end, reinterpret_cast<size_t>(base + count));
    host->lea(tmp, host->ptr[src + offset + (lanes - 1) * sizeof(float)]);
    host->cmp(tmp, end);
    host->jb(full, Xbyak::CodeGenerator::T_NEAR);

    host->sub(host->rsp, lanes * sizeof(float));
    for (int lane = 0; lane < lanes; ++lane) {
        Xbyak::Label skip;
        host->mov(host->dword[host->rsp + lane * sizeof(float)], 0);
        host->lea(tmp, host->ptr[src + offset + lane * sizeof(float)]);
        host->cmp(tmp, end);
        host->jae(skip, Xbyak::CodeGenerator::T_NEAR);
        host->mov(tmp.cvt32(), host->dword[tmp]);
        host->mov(host->dword[host->rsp + lane * sizeof(float)], tmp.cvt32());
        host->L(skip);
    }
    host->uni_vmovups(dst, host->ptr[host->rsp]);
    host->add(host->rsp, lanes * sizeof(float));
    host->jmp(done, Xbyak::CodeGenerator::T_NEAR);

    host->L(full);
    host->uni_vmovups(dst, host->ptr[src + offset]);
    host->L(done);
    host->pop(tmp);
    host->pop(end);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif