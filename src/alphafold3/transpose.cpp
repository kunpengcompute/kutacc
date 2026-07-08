#include "kutacc.h"
#include "common.h"
namespace kutacc{

void mha_transpose_s_h(bfloat16_t* in, bfloat16_t* o, int B, int H, int N, int D)
{
    int N_block = (N%block_n)?(N/block_n+1):(N/block_n);
    // #pragma omp parallel for collapse(3)
    // for(int bidx = 0; bidx < B; ++bidx)
    // {
    //     for(int hidx = 0; hidx < H; ++hidx)
    //     {
    //         for(int nidx = 0; nidx < N_block; ++nidx)
    //         {
    kutacc::parallel_for(0, B * H * N_block , 1, [&](int64_t start, int64_t end){
        for(int task_idx = start; task_idx < end; task_idx ++)
        {
            int bidx = task_idx / (H * N_block);
            int hidx = (task_idx % (H * N_block)) / N_block;
            int nidx = task_idx % N_block;

            bfloat16_t* in_h_n = in+bidx*N*H*D+nidx*block_n*H*D+hidx*D;
            bfloat16_t* o_h_n = o+bidx*H*N*D+hidx*N*D+nidx*block_n*D;
            int block_size_n = (nidx<N_block-1)?block_n:(N-block_n*nidx);

            int in_step = H*D;
            bfloat16_t* in0 = in_h_n;
            bfloat16_t* in1 = in0+in_step;
            bfloat16_t* in2 = in1+in_step;
            bfloat16_t* in3 = in2+in_step;
            bfloat16_t* o0 = o_h_n;
            bfloat16_t* o1 = o0+D;
            bfloat16_t* o2 = o1+D;
            bfloat16_t* o3 = o2+D;
            for(int tn = 0; tn < block_size_n; tn += 4)
            {
                for(int d = 0; d < D; d += SVL_bf16)
                {
                    svbool_t p16all = svwhilelt_b16(0, D - d);
                    svbfloat16_t in0_d = svld1_bf16(p16all, in0+d);
                    svbfloat16_t in1_d = svld1_bf16(p16all, in1+d);
                    svbfloat16_t in2_d = svld1_bf16(p16all, in2+d);
                    svbfloat16_t in3_d = svld1_bf16(p16all, in3+d);

                    svst1_bf16(p16all, o0+d, in0_d);
                    svst1_bf16(p16all, o1+d, in1_d);
                    svst1_bf16(p16all, o2+d, in2_d);
                    svst1_bf16(p16all, o3+d, in3_d);
                }
                in0 += 4*in_step;
                in1 += 4*in_step;
                in2 += 4*in_step;
                in3 += 4*in_step;
                o0 += 4*D;
                o1 += 4*D;
                o2 += 4*D;
                o3 += 4*D;
            }
        }
    });
}
} //namespace kutacc