#ifndef CPU_X64_INJECTORS_JIT_CHANNEL_LOAD_HPP
#define CPU_X64_INJECTORS_JIT_CHANNEL_LOAD_HPP

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

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