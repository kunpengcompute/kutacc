#include "linear.h"
#include "math/fast_exp.h"
#include "utils/bf16.h"
namespace kutacc{

void transitionblock_out(bfloat16_t* act, bfloat16_t* weights, bfloat16_t* out, int batch, int seq_len, int nchannels, int nheads)
{
    auto [tm, tn] = kutacc::compute_tm_tn(batch * seq_len, nheads);

    kutacc::MatrixTilingBlock tiling(tm, tn, nchannels);
    std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[batch * seq_len * nchannels]);
    kutacc::bf16_gemm_pack(batch * seq_len, nchannels, std::get<0>(tiling), std::get<2>(tiling), (__bf16 *)act, pack_a.get());
    std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[nchannels * nheads]);
    kutacc::bf16_gemm_pack(nheads, nchannels, std::get<1>(tiling), std::get<2>(tiling), (__bf16 *)weights, pack_b.get());
    std::unique_ptr<__bf16[]> tmpc(new __bf16[batch * seq_len * nheads]);
    kutacc::bf16_packed_gemm(batch * seq_len, nheads, nchannels, tiling, 
        pack_a.get(),
        pack_b.get(),
        (__bf16 *)out,
        tmpc.get());
}
} //namespace kutacc
