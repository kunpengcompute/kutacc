#include "linear.h"
// #include "tensor/tensor.h"
#include "activation/sigmoid.h"
#include "utils/bf16.h"
#include "wrapper/wrapper.h"
namespace kutacc{

void pack_kernel_eq1(bfloat16_t* a, bfloat16_t* packed_a, int B, int N)
{
    // SME_ON();
    svbool_t p32all = svptrue_b32();
    for(int bidx = 0; bidx < B; bidx += SVL_fp32)
    {
        svbool_t pg = svwhilelt_b32(0, B-bidx);
        int sv_len = std::min(SVL_fp32, B-bidx);
        for(int nidx = 0; nidx < N; nidx += 4*SVL_bf16)
        {
            if(N-nidx>=4*SVL_bf16)
            {
                bfloat16_t* a0 = a+bidx*N+nidx;
                bfloat16_t* a1 = a0+SVL_bf16;
                bfloat16_t* a2 = a1+SVL_bf16;
                bfloat16_t* a3 = a2+SVL_bf16;
                for(int trow = 0; trow < sv_len; trow++)
                {
                    svld1_hor_za32(0, trow, p32all, (float*)(a0+trow*N));
                    svld1_hor_za32(1, trow, p32all, (float*)(a1+trow*N));
                    svld1_hor_za32(2, trow, p32all, (float*)(a2+trow*N));
                    svld1_hor_za32(3, trow, p32all, (float*)(a3+trow*N));
                }

                bfloat16_t* p_a0 = packed_a+bidx*N+nidx*sv_len;
                bfloat16_t* p_a1 = p_a0+SVL_bf16*sv_len;
                bfloat16_t* p_a2 = p_a1+SVL_bf16*sv_len;
                bfloat16_t* p_a3 = p_a2+SVL_bf16*sv_len;
                for(int tcol = 0; tcol < SVL_fp32; tcol++)
                {
                    svst1_ver_za32(0, tcol, pg, (float*)(p_a0+tcol*2*sv_len)); 
                    svst1_ver_za32(1, tcol, pg, (float*)(p_a1+tcol*2*sv_len)); 
                    svst1_ver_za32(2, tcol, pg, (float*)(p_a2+tcol*2*sv_len)); 
                    svst1_ver_za32(3, tcol, pg, (float*)(p_a3+tcol*2*sv_len)); 	
                }
            }
            else
            {
                bfloat16_t* a0 = a+bidx*N+nidx;
                bfloat16_t* a1 = a0+SVL_bf16;
                for(int trow = 0; trow < sv_len; trow++)
                {
                    svld1_hor_za32(0, trow, p32all, (float*)(a0+trow*N));
                    svld1_hor_za32(1, trow, p32all, (float*)(a1+trow*N));
                }

                bfloat16_t* p_a0 = packed_a+bidx*N+nidx*sv_len;
                bfloat16_t* p_a1 = p_a0+SVL_bf16*sv_len;
                for(int tcol = 0; tcol < SVL_fp32; tcol++)
                {
                    svst1_ver_za32(0, tcol, pg, (float*)(p_a0+tcol*2*sv_len)); 
                    svst1_ver_za32(1, tcol, pg, (float*)(p_a1+tcol*2*sv_len)); 
                }                
            }
        }        
    }
    // SME_OFF();
}

void einsum_eq1(bfloat16_t* aptr, bfloat16_t* bptr, bfloat16_t* o, int C, int B, int N)
{
    int max_threads = kutacc::get_thread_num();
    int avg_task = C / max_threads;
    int busy_thread = C-avg_task*max_threads;

    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
        SME_ON();
        int tid = kutacc::get_thread_id();
        int task_num, start_idx, end_idx;
        if(tid < busy_thread)
        {
            task_num = avg_task+1;
            start_idx = tid*task_num;
            end_idx = start_idx+task_num;
        }
        else
        {
            task_num = avg_task;
            start_idx = busy_thread*(task_num+1)+(tid-busy_thread)*task_num;
            end_idx = start_idx+task_num;
        }
        alignas(64) bfloat16_t packed_a[N*N];
        alignas(64) bfloat16_t packed_b[B*N];
        svbool_t p16all = svptrue_b16();
        svfloat32_t zero_vec_f32 = svdup_f32(0.0f);
        svbfloat16_t zero_vec_b16 = svdup_bf16(0.0);
        for(int cidx = start_idx; cidx < end_idx; ++cidx)
        {
            bfloat16_t* a_cidx = aptr+cidx*N*N;
            bfloat16_t* b_cidx = bptr+cidx*B*N;
            bfloat16_t* o_cidx = o+cidx*N*B;
            pack_kernel_eq1(a_cidx, packed_a, N, N);
            pack_kernel_eq1(b_cidx, packed_b, B, N);
            //N should be the m*64
            for(int nidx = 0; nidx < N; nidx += 4*SVL_fp32)
            {
                bfloat16_t* a0 = packed_a+nidx*N;
                bfloat16_t* a1 = a0+N*SVL_fp32;
                bfloat16_t* a2 = a1+N*SVL_fp32;
                bfloat16_t* a3 = a2+N*SVL_fp32;
                for(int bidx = 0; bidx < B; bidx += SVL_fp32)
                {
                    svzero_za();
                    svbool_t pgb = svwhilelt_b16(0,2*(B-bidx));
                    int sv_len = std::min(SVL_fp32, B-bidx);
                    bfloat16_t* b0 = packed_b+bidx*N;
                    for(int tn = 0; tn < N; tn += 2)
                    {
                        svbfloat16_t b_d = svld1_bf16(pgb, b0+tn*sv_len);
                        svbfloat16_t a_d0 = svld1_bf16(p16all, a0+tn*SVL_fp32);
                        svbfloat16_t a_d1 = svld1_bf16(p16all, a1+tn*SVL_fp32);
                        svbfloat16_t a_d2 = svld1_bf16(p16all, a2+tn*SVL_fp32);
                        svbfloat16_t a_d3 = svld1_bf16(p16all, a3+tn*SVL_fp32);

                        svmopa_za32_bf16_m(0, p16all, pgb, a_d0, b_d);
                        svmopa_za32_bf16_m(1, p16all, pgb, a_d1, b_d);
                        svmopa_za32_bf16_m(2, p16all, pgb, a_d2, b_d);
                        svmopa_za32_bf16_m(3, p16all, pgb, a_d3, b_d);
                    }
                    
                    bfloat16_t* o0 = o_cidx+nidx*B+bidx;
                    bfloat16_t* o1 = o0+SVL_fp32*B;
                    bfloat16_t* o2 = o1+SVL_fp32*B;
                    bfloat16_t* o3 = o2+SVL_fp32*B;
                    svbool_t pgb_st = svwhilelt_b32(0,B-bidx);
                    svbool_t pg_st = svwhilelt_b16(0,B-bidx);
                    #pragma unroll(4)
                    for(int trow = 0; trow < SVL_fp32; trow++)
                    {
                        svfloat32_t d0 = svread_hor_za32_f32_m(zero_vec_f32, pgb_st, 0, trow);
                        svfloat32_t d1 = svread_hor_za32_f32_m(zero_vec_f32, pgb_st, 1, trow);
                        svfloat32_t d2 = svread_hor_za32_f32_m(zero_vec_f32, pgb_st, 2, trow);
                        svfloat32_t d3 = svread_hor_za32_f32_m(zero_vec_f32, pgb_st, 3, trow);
                        svbfloat16_t b_d0 = svcvt_bf16_f32_x(pgb_st, d0);
                        svbfloat16_t b_d1 = svcvt_bf16_f32_x(pgb_st, d1);
                        svbfloat16_t b_d2 = svcvt_bf16_f32_x(pgb_st, d2);
                        svbfloat16_t b_d3 = svcvt_bf16_f32_x(pgb_st, d3);
                        b_d0 = svuzp1_bf16(b_d0, zero_vec_b16);
                        b_d1 = svuzp1_bf16(b_d1, zero_vec_b16);
                        b_d2 = svuzp1_bf16(b_d2, zero_vec_b16);
                        b_d3 = svuzp1_bf16(b_d3, zero_vec_b16);

                        svst1_bf16(pg_st, o0+trow*B, b_d0);  
                        svst1_bf16(pg_st, o1+trow*B, b_d1);   
                        svst1_bf16(pg_st, o2+trow*B, b_d2);  
                        svst1_bf16(pg_st, o3+trow*B, b_d3);                         
                    }
                }
            }
        }
        SME_OFF();  
    });
}

void pack_kernel_eq2(bfloat16_t* a, bfloat16_t* packed_a, int N, int B)
{
    for(int bidx = 0; bidx < B; bidx += SVL_fp32)
    {
        svbool_t pg = svwhilelt_b16(0, B-bidx);
        int sv_len = std::min(SVL_fp32, B-bidx);
        svbool_t pg_st = svwhilelt_b16(0, 2*sv_len);
        bfloat16_t* a_nidx = a+bidx;
        bfloat16_t* p_a_nidx = packed_a+N*bidx;
        #pragma unroll(8)
        for(int nidx = 0; nidx < N; nidx += 2)
        {
            svbfloat16_t a_d1 = svld1_bf16(pg, a_nidx + nidx*B);
            svbfloat16_t a_d2 = svld1_bf16(pg, a_nidx + (nidx+1)*B);
            a_d1 = svzip1_bf16(a_d1, a_d2);
            svst1_bf16(pg_st, p_a_nidx+nidx*sv_len, a_d1);
        }
    }
}

void einsum_eq2(bfloat16_t* aptr, bfloat16_t* bptr, bfloat16_t* o, int C, int B, int N)
{
    int max_threads = kutacc::get_thread_num();
    int avg_task = C / max_threads;
    int busy_thread = C-avg_task*max_threads;

    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
        SME_ON();
        int tid = kutacc::get_thread_id();
        int task_num, start_idx, end_idx;
        if(tid < busy_thread)
        {
            task_num = avg_task+1;
            start_idx = tid*task_num;
            end_idx = start_idx+task_num;
        }
        else
        {
            task_num = avg_task;
            start_idx = busy_thread*(task_num+1)+(tid-busy_thread)*task_num;
            end_idx = start_idx+task_num;
        }
        alignas(64) bfloat16_t packed_a[N*N];
        alignas(64) bfloat16_t packed_b[N*B];
        svbool_t p16all = svptrue_b16();
        svbool_t p32all = svptrue_b32();
        svfloat32_t zero_vec_f32 = svdup_f32(0.0f);
        for(int cidx = start_idx; cidx < end_idx; ++cidx)
        {
            bfloat16_t* a_cidx = aptr+cidx*N*N;
            bfloat16_t* b_cidx = bptr+cidx*N*B;
            bfloat16_t* o_cidx = o+cidx*B*N;
            pack_kernel_eq2(a_cidx, packed_a, N, N);
            pack_kernel_eq2(b_cidx, packed_b, N, B);
            //N should be the m*64
            for(int nidx = 0; nidx < N; nidx += 4*SVL_fp32)
            {
                bfloat16_t* a0 = packed_a+nidx*N;
                bfloat16_t* a1 = a0+N*SVL_fp32;
                bfloat16_t* a2 = a1+N*SVL_fp32;
                bfloat16_t* a3 = a2+N*SVL_fp32;
                for(int bidx = 0; bidx < B; bidx += SVL_fp32)
                {
                    svzero_za();
                    svbool_t pgb = svwhilelt_b16(0,2*(B-bidx));
                    int sv_len = (SVL_fp32<(B-bidx))?SVL_fp32:(B-bidx);
                    bfloat16_t* b0 = packed_b+bidx*N;
                    for(int tn = 0; tn < N; tn += 2)
                    {
                        svbfloat16_t b_d = svld1_bf16(pgb, b0+tn*sv_len);
                        svbfloat16_t a_d0 = svld1_bf16(p16all, a0+tn*SVL_fp32);
                        svbfloat16_t a_d1 = svld1_bf16(p16all, a1+tn*SVL_fp32);
                        svbfloat16_t a_d2 = svld1_bf16(p16all, a2+tn*SVL_fp32);
                        svbfloat16_t a_d3 = svld1_bf16(p16all, a3+tn*SVL_fp32);

                        svmopa_za32_bf16_m(0, pgb, p16all, b_d, a_d0);
                        svmopa_za32_bf16_m(1, pgb, p16all, b_d, a_d1);
                        svmopa_za32_bf16_m(2, pgb, p16all, b_d, a_d2);
                        svmopa_za32_bf16_m(3, pgb, p16all, b_d, a_d3);
                    }
                    #pragma unroll(4)
                    for(int trow = 0; trow < sv_len; trow++)
                    {
                        svfloat32_t d0 = svread_hor_za32_f32_m(zero_vec_f32, p32all, 0, trow);
                        svfloat32_t d1 = svread_hor_za32_f32_m(zero_vec_f32, p32all, 1, trow);
                        svfloat32_t d2 = svread_hor_za32_f32_m(zero_vec_f32, p32all, 2, trow);
                        svfloat32_t d3 = svread_hor_za32_f32_m(zero_vec_f32, p32all, 3, trow);
                        svbfloat16_t b_d0 = svcvt_bf16_f32_x(p32all, d0);
                        svbfloat16_t b_d1 = svcvt_bf16_f32_x(p32all, d1);
                        svbfloat16_t b_d2 = svcvt_bf16_f32_x(p32all, d2);
                        svbfloat16_t b_d3 = svcvt_bf16_f32_x(p32all, d3);
                        b_d0 = svuzp1_bf16(b_d0, b_d1);
                        b_d2 = svuzp1_bf16(b_d2, b_d3);
                        svst1_bf16(p16all, o_cidx+(bidx+trow)*N+nidx, b_d0);  
                        svst1_bf16(p16all, o_cidx+(bidx+trow)*N+nidx+SVL_bf16, b_d2);                          
                    }
                }
            }
        }
        SME_OFF();  
    });
}
void trianglemultiplication(bfloat16_t* input, bfloat16_t* input_act, bfloat16_t* output_w, bfloat16_t* gating_w, bfloat16_t* output, bfloat16_t* gate, int64_t c_0, int64_t c_1, int64_t c_2){
    int64_t nchannels = c_2;
    auto [tm, tn] = kutacc::compute_tm_tn(c_0 * c_1, c_2);
    kutacc::MatrixTilingBlock tiling(tm, tn, c_2);
    {
        std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[c_0 * c_1 * c_2]);
        std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[c_2 * c_2]);
        std::unique_ptr<__bf16[]> tmpc(new __bf16[c_0 * c_1 * c_2]);
        kutacc::bf16_gemm_pack(c_0 * c_1, c_2, std::get<0>(tiling), std::get<2>(tiling), (__bf16 *)input, pack_a.get());
        kutacc::bf16_gemm_pack(c_2, c_2, std::get<1>(tiling), std::get<2>(tiling), (__bf16 *)output_w, pack_b.get());
        kutacc::bf16_packed_gemm(c_0 * c_1, c_2, c_2, tiling, 
            pack_a.get(),
            pack_b.get(),
            (__bf16 *)output,
            tmpc.get());
    }
    {
        std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[c_0 * c_1 * c_2]);
        std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[c_2 * c_2]);
        std::unique_ptr<__bf16[]> tmpc(new __bf16[c_0 * c_1 * c_2]);
        kutacc::bf16_gemm_pack(c_0 * c_1, c_2, std::get<0>(tiling), std::get<2>(tiling), (__bf16 *)input_act, pack_a.get());
        kutacc::bf16_gemm_pack(c_2, c_2, std::get<1>(tiling), std::get<2>(tiling), (__bf16 *)gating_w, pack_b.get());
        kutacc::bf16_packed_gemm(c_0 * c_1, c_2, c_2, tiling, 
            pack_a.get(),
            pack_b.get(),
            (__bf16 *)gate,
            tmpc.get());
    }
    {
        kutacc::parallel_for(0, c_0, 1, [&](int64_t start, int64_t end) {
        for (int bi = start; bi < end; bi++) {
            int64_t vl = svcntw();
            svbool_t pg = svptrue_b32();
        
            for (int64_t qi = 0; qi < c_1; qi++) {
                // 获取当前input和gate的数据指针
                auto input_data = output + bi * c_1 * c_2 + qi * c_2;
                auto gate_data = gate + bi * c_1 * c_2  + qi * c_2;
                
                // 使用循环处理所有通道，每次处理4个向量长度
                for (int64_t ci = 0; ci < nchannels; ci += vl * 4) {
                    // 加载input数据
                    svfloat32_t w0 = kutacc::svld1<float, bfloat16_t>(pg, input_data + ci);
                    svfloat32_t w1 = kutacc::svld1<float, bfloat16_t>(pg, input_data + ci + vl);
                    svfloat32_t w2 = kutacc::svld1<float, bfloat16_t>(pg, input_data + ci + vl * 2);
                    svfloat32_t w3 = kutacc::svld1<float, bfloat16_t>(pg, input_data + ci + vl * 3);
                    
                    // 加载gate数据
                    svfloat32_t g0 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci);
                    svfloat32_t g1 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl);
                    svfloat32_t g2 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl * 2);
                    svfloat32_t g3 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl * 3);
                    
                    // 计算input *= sigmoid(gate)
                    w0 = svmul_x(pg, w0, kutacc::Sigmoid::call(pg, g0));
                    w1 = svmul_x(pg, w1, kutacc::Sigmoid::call(pg, g1));
                    w2 = svmul_x(pg, w2, kutacc::Sigmoid::call(pg, g2));
                    w3 = svmul_x(pg, w3, kutacc::Sigmoid::call(pg, g3));
                    
                    // 存回input
                    kutacc::svst1<bfloat16_t, float>(pg, input_data + ci, w0);
                    kutacc::svst1<bfloat16_t, float>(pg, input_data + ci + vl, w1);
                    kutacc::svst1<bfloat16_t, float>(pg, input_data + ci + vl * 2, w2);
                    kutacc::svst1<bfloat16_t, float>(pg, input_data + ci + vl * 3, w3);
            }
        } 
    }
});
    }
}

} //namespace kutacc
