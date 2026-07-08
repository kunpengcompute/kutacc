#include "../linear.h"
#include "activation/sigmoid.h"
#include "math/fast_exp.h"
#include "utils/bf16.h"

namespace kutacc {
inline void pack_qk_hd16(bfloat16_t* q_h_i, bfloat16_t* q_block, int bloc_q)
{
    const int D = 16;
    svbool_t p32all = svptrue_b32();
    for(int row = 0; row < bloc_q; row += 4*SVL_fp32)
    {
        bfloat16_t* q0 = q_h_i+row*D;
        bfloat16_t* q1 = q0+SVL_fp32*D;
        bfloat16_t* q2 = q1+SVL_fp32*D;
        bfloat16_t* q3 = q2+SVL_fp32*D;
        bfloat16_t* qb0 = q_block+row*D;
        bfloat16_t* qb1 = qb0+SVL_fp32*D;
        bfloat16_t* qb2 = qb1+SVL_fp32*D;
        bfloat16_t* qb3 = qb2+SVL_fp32*D;
        for(int trow = 0; trow < SVL_fp32; trow++)
        {
            svld1_hor_za32(0, trow, p32all, (float*)(q0+trow*D));
            svld1_hor_za32(1, trow, p32all, (float*)(q1+trow*D));
            svld1_hor_za32(2, trow, p32all, (float*)(q2+trow*D));
            svld1_hor_za32(3, trow, p32all, (float*)(q3+trow*D));
        }
        for(int tcol = 0; tcol < 8; tcol++)
        {
            svst1_ver_za32(0, tcol, p32all, (float*)(qb0+tcol*SVL_bf16));  
            svst1_ver_za32(2, tcol, p32all, (float*)(qb2+tcol*SVL_bf16)); 
            svst1_ver_za32(1, tcol, p32all, (float*)(qb1+tcol*SVL_bf16));
            svst1_ver_za32(3, tcol, p32all, (float*)(qb3+tcol*SVL_bf16)); 	
        }
    }
}

inline void pack_qk_hd32(bfloat16_t* q_h_i, bfloat16_t* q_block, int bloc_q)
{
    const int D = 32;
    svbool_t p32all = svptrue_b32();
    for(int row = 0; row < bloc_q; row += 4*SVL_fp32)
    {
        bfloat16_t* q0 = q_h_i+row*D;
        bfloat16_t* q1 = q0+SVL_fp32*D;
        bfloat16_t* q2 = q1+SVL_fp32*D;
        bfloat16_t* q3 = q2+SVL_fp32*D;
        bfloat16_t* qb0 = q_block+row*D;
        bfloat16_t* qb1 = qb0+SVL_fp32*D;
        bfloat16_t* qb2 = qb1+SVL_fp32*D;
        bfloat16_t* qb3 = qb2+SVL_fp32*D;
        for(int trow = 0; trow < SVL_fp32; trow++)
        {
            svld1_hor_za32(0, trow, p32all, (float*)(q0+trow*D));
            svld1_hor_za32(1, trow, p32all, (float*)(q1+trow*D));
            svld1_hor_za32(2, trow, p32all, (float*)(q2+trow*D));
            svld1_hor_za32(3, trow, p32all, (float*)(q3+trow*D));
        }
        for(int tcol = 0; tcol < SVL_fp32; tcol++)
        {
            svst1_ver_za32(0, tcol, p32all, (float*)(qb0+tcol*SVL_bf16));  
            svst1_ver_za32(2, tcol, p32all, (float*)(qb2+tcol*SVL_bf16)); 
            svst1_ver_za32(1, tcol, p32all, (float*)(qb1+tcol*SVL_bf16));
            svst1_ver_za32(3, tcol, p32all, (float*)(qb3+tcol*SVL_bf16)); 	
        }
    }
}


void mask_b2f(bool* mask, float* mask_data, int B, int N)
{
    kutacc::parallel_for(0, B*N, 1, [&](int64_t start, int64_t end) {
        for (int64_t task_idx = start; task_idx < end; task_idx++){
            mask_data[task_idx] = mask[task_idx]?0:data_min_grid;
        }
    });
}

void gridself_layernorm_bias_bf16_pack(float* x, bfloat16_t* x_bf16, bfloat16_t* x_r, float* ln_weight,  float* ln_bias, int batch, int seq_len, int nchannels, float eps)
{
    svbool_t pt16_w = svwhilelt_b16(0,16);
    int M = batch * seq_len, N = nchannels;
    int max_threads = kutacc::get_thread_num();
    int task_num = M / SVL_fp32;
    int avg_task = task_num / max_threads;
    int busy_thread = task_num-avg_task*max_threads;
	kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
        int tid = kutacc::get_thread_id();
        int task_num, start_idx, end_idx;
        if(tid < busy_thread)
        {
            task_num = avg_task+1;
            start_idx = tid*task_num*SVL_fp32;
            end_idx = start_idx+task_num*SVL_fp32;
        }
        else
        {
            task_num = avg_task;
            start_idx = busy_thread*(task_num+1)*SVL_fp32+(tid-busy_thread)*task_num*SVL_fp32;
            end_idx = start_idx+task_num*SVL_fp32;
        }

        for(int i = start_idx; i < end_idx; i++)
        {
            svbool_t p32all = svptrue_b32();
            svfloat32_t sum_v = svdup_f32(0);
            svfloat32_t sum2_v = svdup_f32(0.0f);
            float* x_i = x+i*N;
            float* x_j = x_i;
            bfloat16_t *x_bf16_i = x_bf16 + i*N;
            float* w_j = ln_weight;
            float* b_j = ln_bias;
            for(int j = 0; j < N; j += SVL_fp32)
            {
                svbool_t pg = svwhilelt_b32(j, N);
                svfloat32_t x_v = svld1_f32(pg, x_j);
                sum_v = svadd_m(pg, sum_v, x_v);
                sum2_v = svmla_m(pg, sum2_v, x_v, x_v);
                x_j += SVL_fp32;
            }

            float sum = svaddv(p32all, sum_v);
            float sum2 = svaddv(p32all, sum2_v);
            float mean = sum/N;
            float var = std::max((sum2-sum*sum/N)/N, 0.f);
            float rstd = 1/std::sqrt(var + eps);        
            svfloat32_t bias = svdup_f32(-rstd * mean);  
            x_j = x_i; 

            for (int j = 0; j < N; j += SVL_fp32) {
                svbool_t pg = svwhilelt_b32(j, N);
                svfloat32_t x_v = svld1_f32(pg, x_j);
                svfloat32_t w_v = svld1_f32(pg, w_j);
                svfloat32_t b_v = svld1_f32(pg, b_j);
                x_v = svmla_x(pg, bias, x_v, rstd);
                x_v = svmul_f32_m(pg, x_v, w_v);
                x_v = svadd_f32_m(pg, x_v, b_v);
                svst1_f32(pg, x_j, x_v);
                svbfloat16_t x_bf16_v = svcvt_bf16_f32_z(pg, x_v);
                svbfloat16_t unzip_1 = svuzp1_bf16(x_bf16_v, svbfloat16_t());
                svst1_bf16(pt16_w, x_bf16_i, unzip_1);
                x_j += SVL_fp32;
                w_j += SVL_fp32;
                b_j += SVL_fp32;
                x_bf16_i += SVL_fp32;
            }
        }

        SME_ON();
        svbool_t p32all = svptrue_b32();
        svbool_t p16all = svptrue_b16();
        int step = SVL_fp32*SVL_fp32;
        for(int i = start_idx; i < end_idx; i += SVL_fp32)
        {
            float* x_i = x+i*N;
            bfloat16_t* x_r_i = x_r + i * N;
            for(int j = 0; j < N; j += 4*SVL_fp32)
            {
                float* x0 = x_i+j;
                float* x1 = x0+SVL_fp32;
                float* x2 = x1+SVL_fp32;
                float* x3 = x2+SVL_fp32;
                for(int trow = 0; trow < SVL_fp32; trow++)
                {
                    svld1_hor_za32(0, trow, p32all, x0+trow*N);
                    svld1_hor_za32(1, trow, p32all, x1+trow*N);
                    svld1_hor_za32(2, trow, p32all, x2+trow*N);
                    svld1_hor_za32(3, trow, p32all, x3+trow*N);
                }

                bfloat16_t* x0_r = x_r_i+j*SVL_fp32;
                bfloat16_t* x1_r = x0_r+step;
                bfloat16_t* x2_r = x1_r+step;
                bfloat16_t* x3_r = x2_r+step;
                for(int tcol = 0; tcol < SVL_fp32; tcol +=2 )
                {
                    svfloat32_t data_0,data_1,data_2,data_3;
                    svbfloat16_t data_bf16_0,data_bf16_1,data_bf16_2,data_bf16_3,unzip_1,unzip_2,z1,z2;
                    data_0 = svread_ver_za32_f32_m(svfloat32_t(),p32all,0,tcol);
                    data_1 = svread_ver_za32_f32_m(svfloat32_t(),p32all,0,tcol+1);
                    data_2 = svread_ver_za32_f32_m(svfloat32_t(),p32all,1,tcol);
                    data_3 = svread_ver_za32_f32_m(svfloat32_t(),p32all,1,tcol+1);

                    data_bf16_0 = svcvt_bf16_f32_z(p32all, data_0);
                    data_bf16_1 = svcvt_bf16_f32_z(p32all, data_1);
                    data_bf16_2 = svcvt_bf16_f32_z(p32all, data_2);
                    data_bf16_3 = svcvt_bf16_f32_z(p32all, data_3);

                    unzip_1 = svuzp1_bf16(data_bf16_0, data_bf16_2);
                    unzip_2 = svuzp1_bf16(data_bf16_1, data_bf16_3);
                    z1 = svzip1_bf16(unzip_1,unzip_2);
                    z2 = svzip2_bf16(unzip_1,unzip_2);
                    svst1_bf16(p16all, x0_r, z1);
                    svst1_bf16(p16all, x1_r, z2);

                    data_0 = svread_ver_za32_f32_m(svfloat32_t(),p32all,2,tcol);
                    data_1 = svread_ver_za32_f32_m(svfloat32_t(),p32all,2,tcol+1);
                    data_2 = svread_ver_za32_f32_m(svfloat32_t(),p32all,3,tcol);
                    data_3 = svread_ver_za32_f32_m(svfloat32_t(),p32all,3,tcol+1);

                    data_bf16_0 = svcvt_bf16_f32_z(p32all, data_0);
                    data_bf16_1 = svcvt_bf16_f32_z(p32all, data_1);
                    data_bf16_2 = svcvt_bf16_f32_z(p32all, data_2);
                    data_bf16_3 = svcvt_bf16_f32_z(p32all, data_3);

                    unzip_1 = svuzp1_bf16(data_bf16_0, data_bf16_2);
                    unzip_2 = svuzp1_bf16(data_bf16_1, data_bf16_3);
                    z1 = svzip1_bf16(unzip_1,unzip_2);
                    z2 = svzip2_bf16(unzip_1,unzip_2);
                    svst1_bf16(p16all, x2_r, z1);
                    svst1_bf16(p16all, x3_r, z2);

                    x0_r += 2*SVL_fp32;
                    x1_r += 2*SVL_fp32;
                    x2_r += 2*SVL_fp32;
                    x3_r += 2*SVL_fp32;
                }
            }
        }
        SME_OFF();  
    });
}

void gridself_attention_out(bfloat16_t* weights, bfloat16_t* weighted_avg, bfloat16_t* gate,  bfloat16_t* out, int batch, int seq_len, int nchannels){
    kutacc::parallel_for(0, batch, 1, [&](int64_t start, int64_t end) {
        for (int64_t bi = start; bi < end; bi++) {
            int64_t vl = svcntw();
            svbool_t pg = svptrue_b32();
        
            for (int qi = 0 ; qi < seq_len; qi ++){
                // 获取当前weighted_avg和gate的数据指针
                auto wavg_data = weighted_avg+ bi * seq_len * nchannels + qi * nchannels;
                auto gate_data = gate + bi * seq_len * nchannels + qi * nchannels;
                
                // 使用循环处理所有通道，每次处理4个向量长度
                for (int64_t ci = 0; ci < nchannels; ci += vl * 4) {
                    // 加载weighted_avg数据
                    svfloat32_t w0 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci);
                    svfloat32_t w1 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl);
                    svfloat32_t w2 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl * 2);
                    svfloat32_t w3 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl * 3);
                    
                    // 加载gate数据
                    svfloat32_t g0 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci);
                    svfloat32_t g1 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl);
                    svfloat32_t g2 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl * 2);
                    svfloat32_t g3 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl * 3);
                    
                    // 计算weighted_avg *= sigmoid(gate)
                    w0 = svmul_x(pg, w0, kutacc::Sigmoid::call(pg, g0));
                    w1 = svmul_x(pg, w1, kutacc::Sigmoid::call(pg, g1));
                    w2 = svmul_x(pg, w2, kutacc::Sigmoid::call(pg, g2));
                    w3 = svmul_x(pg, w3, kutacc::Sigmoid::call(pg, g3));
                    
                    // 存回weighted_avg
                    kutacc::svst1<bfloat16_t, float>(pg, wavg_data + ci, w0);
                    kutacc::svst1<bfloat16_t, float>(pg, wavg_data + ci + vl, w1);
                    kutacc::svst1<bfloat16_t, float>(pg, wavg_data + ci + vl * 2, w2);
                    kutacc::svst1<bfloat16_t, float>(pg, wavg_data + ci + vl * 3, w3);
                }
            }
        } 
    });
    auto [tm, tn] = kutacc::compute_tm_tn(batch * seq_len, nchannels);
    kutacc::MatrixTilingBlock tiling(tm, tn, nchannels);
    std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[batch * seq_len * nchannels]);
    std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[nchannels * nchannels]);
    std::unique_ptr<__bf16[]> tmpc(new __bf16[batch * seq_len * nchannels]);
    kutacc::bf16_gemm_pack(batch * seq_len, nchannels, std::get<0>(tiling), std::get<2>(tiling), (__bf16 *)weighted_avg, pack_a.get());
    kutacc::bf16_gemm_pack(nchannels, nchannels, std::get<1>(tiling), std::get<2>(tiling), (__bf16 *)weights, pack_b.get());
    kutacc::bf16_packed_gemm(batch * seq_len, nchannels, nchannels, tiling, 
        pack_a.get(),
        pack_b.get(),
        (__bf16 *)out,
        tmpc.get());

}

void gridself_attention_bias_linear(bfloat16_t* input, bfloat16_t* bias_weight, bfloat16_t* bias, int batch, int seq_len, int nchannels, int nheads){
    auto [tm, tn] = kutacc::compute_tm_tn(batch * seq_len, nheads);
    kutacc::MatrixTilingBlock tiling(tm, tn, nchannels);
    std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[batch * seq_len * nchannels]);
    std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[nchannels * nheads]);
    std::unique_ptr<__bf16[]> tmpc(new __bf16[batch * seq_len * nheads]);
    kutacc::bf16_gemm_pack(batch * seq_len, nchannels, std::get<0>(tiling), std::get<2>(tiling), (__bf16 *)input, pack_a.get());
    kutacc::bf16_gemm_pack(nheads, nchannels, std::get<1>(tiling), std::get<2>(tiling), (__bf16 *)bias_weight, pack_b.get());
    kutacc::bf16_packed_gemm(batch * seq_len, nheads, nchannels, tiling, 
        pack_a.get(),
        pack_b.get(),
        (__bf16 *)bias,
        tmpc.get());
}

void gridself_attention_linear_transpose_trans(bfloat16_t* x, bfloat16_t* pack_x, bfloat16_t* q_proj, bfloat16_t* k_proj, bfloat16_t* v_proj, bfloat16_t* g_proj, \
bfloat16_t* pbias_proj, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* g, bfloat16_t* pair_bias, int batch, int seq_len, int nheads, int head_size, bool trans)
{
    int M = batch * seq_len , nchannels = nheads * head_size, N = nheads * head_size, K = nheads * head_size;
    gridself_attention_bias_linear(x, pbias_proj, pair_bias, batch, seq_len, nchannels, nheads);
    const int row_block_size = ROW_BLOCK_SIZE;
    const int col_block_size = COL_BLOCK_SIZE;
    int max_threads = kutacc::get_thread_num();
    int avg_threads = max_threads / 4;
    int task_num = (M * N) / (row_block_size * col_block_size);
    int avg_task = task_num / avg_threads;
    int busy_thread = task_num - avg_task * avg_threads;
    int row_task_num = N / col_block_size; 
	kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
        int tid = kutacc::get_thread_id();
        int tid_local = tid % avg_threads;
        int task_num, start_idx;
        if(tid_local < busy_thread)
        {
            task_num = avg_task+1;
            start_idx = tid_local*task_num;
        }
        else
        {
            task_num = avg_task;
            start_idx = busy_thread*(task_num+1)+(tid_local-busy_thread)*task_num;
        }
        for(int task_idx = start_idx; task_idx < start_idx + task_num; task_idx++ )
        {
            int row_index = (task_idx / row_task_num) * row_block_size;
            int col_index = (task_idx % row_task_num) * col_block_size;
            bfloat16_t *block_part_x_base = pack_x + row_index * K ;
            if(tid < avg_threads){
                kutacc::bgemm_4vl_1vl_bf16_transpose_trans_impl(block_part_x_base, q_proj + col_index * K, q, \
                row_block_size, col_block_size, K, row_index, col_index, batch, seq_len, nheads, head_size, trans);
            }
            else if(tid < 2 * avg_threads){
                kutacc::bgemm_4vl_1vl_bf16_transpose_trans_impl(block_part_x_base, k_proj + col_index * K, k, \
                row_block_size, col_block_size, K, row_index, col_index, batch, seq_len, nheads, head_size, trans);
            }
            else if(tid < 3 * avg_threads){
                kutacc::bgemm_4vl_1vl_bf16_transpose_trans_impl(block_part_x_base, v_proj + col_index * K, v, \
                row_block_size, col_block_size, K, row_index, col_index, batch, seq_len, nheads, head_size, trans);
            }
            else{
                kutacc::bgemm_4vl_1vl_bf16_trans_impl(block_part_x_base, g_proj + col_index * K, g, \
                row_block_size, col_block_size, K, N, row_index, col_index, batch, seq_len, trans);
            }
        }
    });
}


void grid_pack_pair_logits(bfloat16_t* pair_logits_ori, float* pair_logits_r, int nblock_q, int nblock_kv, int H, int Nq, int Nkv)
{
    int task_num = H*nblock_q;
    int max_threads = kutacc::get_thread_num();
    int avg_task = task_num / max_threads;
    int busy_thread = task_num-avg_task*max_threads; 
    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
        SME_ON(); 
        int tid = kutacc::get_thread_id();
        int task_num_i, start_idx, end_idx;
        if(tid < busy_thread)
        {
            task_num_i = avg_task+1;
            start_idx = tid*task_num_i;
            end_idx = start_idx+task_num_i;
        }
        else
        {
            task_num_i = avg_task;
            start_idx = busy_thread*(task_num_i+1)+(tid-busy_thread)*task_num_i;
            end_idx = start_idx+task_num_i;
        }
        svbfloat16_t zero_vec = svdup_bf16(0.0);
        svbool_t p16all = svptrue_b16();
        svbool_t p32all = svptrue_b32();
        for(int idx = start_idx; idx < end_idx; ++idx)
        {
            int qidx = idx%nblock_q;
            int hidx = idx/nblock_q;
            int block_size_q = (qidx<nblock_q-1)?block_q:(Nq-block_q*qidx);
            for(int kvidx = 0; kvidx < nblock_kv; ++kvidx)
            {
                int block_size_kv = (kvidx<nblock_kv-1)?block_kv_grid:(Nkv-block_kv_grid*kvidx);
                bfloat16_t* pl_h_i = pair_logits_ori+hidx*Nq*Nkv+qidx*block_q*Nkv+kvidx*block_kv_grid;
                float* pl_r_h_i = pair_logits_r+hidx*Nq*Nkv+qidx*block_q*Nkv+kvidx*block_kv_grid*block_size_q;
                for(int row = 0; row < block_size_q; row += SVL_fp32)
                {
                    for(int col = 0; col < block_size_kv; col += 2*SVL_bf16)
                    {
                        bfloat16_t* p0 = pl_h_i+row*Nkv+col;
                        bfloat16_t* p1 = p0+SVL_bf16;
                        for(int trow = 0; trow < SVL_fp32; trow++)
                        {
                            svbfloat16_t p_d0 = svld1_bf16(p16all, p0+trow*Nkv);
                            svbfloat16_t p_d1 = svld1_bf16(p16all, p1+trow*Nkv);
                            svfloat32_t p_d00 = svreinterpret_f32_bf16(svzip1_bf16(zero_vec, p_d0));
                            svfloat32_t p_d01 = svreinterpret_f32_bf16(svzip2_bf16(zero_vec, p_d0));
                            svfloat32_t p_d10 = svreinterpret_f32_bf16(svzip1_bf16(zero_vec, p_d1));
                            svfloat32_t p_d11 = svreinterpret_f32_bf16(svzip2_bf16(zero_vec, p_d1));
                            svwrite_hor_za32_f32_m(0, trow, p32all, p_d00); 
                            svwrite_hor_za32_f32_m(1, trow, p32all, p_d01); 
                            svwrite_hor_za32_f32_m(2, trow, p32all, p_d10); 
                            svwrite_hor_za32_f32_m(3, trow, p32all, p_d11); 
                        }

                        float* p_r0 = pl_r_h_i+row*block_size_kv+col*SVL_fp32;
                        float* p_r1 = p_r0+SVL_fp32*SVL_fp32;
                        float* p_r2 = p_r1+SVL_fp32*SVL_fp32;
                        float* p_r3 = p_r2+SVL_fp32*SVL_fp32;
                        for(int tcol = 0; tcol < SVL_fp32; ++tcol)
                        {
                            svst1_ver_za32(0, tcol, p32all, p_r0+tcol*SVL_fp32); 
                            svst1_ver_za32(1, tcol, p32all, p_r1+tcol*SVL_fp32); 
                            svst1_ver_za32(2, tcol, p32all, p_r2+tcol*SVL_fp32); 
                            svst1_ver_za32(3, tcol, p32all, p_r3+tcol*SVL_fp32);
                        }
                    }
                }
            }
        }
        SME_OFF(); 
    });
}

void grid_flash_attn_mask_bias_bf16_hd16(int B, int H, int N, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, float* mask, float* bias_r, float scale, float m_min)
{
    int nblock_q = (N%block_q)?(N/block_q+1):(N/block_q);
    int nblock_kv = (N%block_kv_grid)?(N/block_kv_grid+1):(N/block_kv_grid);
    int task_num = B*H*nblock_q;
    int max_threads = kutacc::get_thread_num();
    int avg_task = task_num / max_threads;
    int busy_thread = task_num-avg_task*max_threads; 

    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
        SME_ON(); 
        int tid = kutacc::get_thread_id();
        int task_num_i, start_idx, end_idx;
        if(tid < busy_thread)
        {
            task_num_i = avg_task+1;
            start_idx = tid*task_num_i;
            end_idx = start_idx+task_num_i;
        }
        else
        {
            task_num_i = avg_task;
            start_idx = busy_thread*(task_num_i+1)+(tid-busy_thread)*task_num_i;
            end_idx = start_idx+task_num_i;
        }

        alignas(64) float l[block_q];
        alignas(64) float m_last[block_q];
        alignas(64) float m[block_q];              
        alignas(64) float S[block_q*block_kv_grid];
        alignas(64) float exp_diff[block_q];
        alignas(64) float o_block[block_q*D];
        alignas(64) bfloat16_t P[block_q*block_kv_grid];
        alignas(64) bfloat16_t k_block[block_kv_grid*D];
        alignas(64) bfloat16_t v_block[block_kv_grid*D];
        alignas(64) bfloat16_t q_block[block_q*D];
        svbool_t ptrue = svptrue_b16();
        svfloat32_t scal_vec = svdup_f32(scale);
        svfloat32_t zero_vec_f32 = svdup_f32(0.0f);

            for(int idx = start_idx; idx < end_idx; ++idx)
            {
                int qidx = idx%nblock_q;
                int hidx = (idx/nblock_q)%H;
                int bidx = idx/nblock_q/H;
                bfloat16_t* q_h_i = q+bidx*H*N*D+hidx*N*D+qidx*block_q*D;
                bfloat16_t* o_h_i = out+bidx*H*N*D+hidx*N*D+qidx*block_q*D;

                int block_size_q = (qidx<nblock_q-1)?block_q:(N-block_q*qidx);
                memset(l, 0, sizeof(l));
                std::fill(m_last, m_last+block_size_q, m_min);
                std::fill(m, m+block_size_q, m_min);                
                memset(o_block, 0, sizeof(o_block)); 
                kutacc::pack_qk_hd16(q_h_i, q_block, block_size_q);
                for(int kvidx = 0; kvidx < nblock_kv; ++kvidx)
                {
                    int block_size_kv = (kvidx<nblock_kv-1)?block_kv_grid:(N-block_kv_grid*kvidx);

                    bfloat16_t* k_h_i = k+bidx*H*N*D+hidx*N*D+kvidx*block_kv_grid*D;
                    bfloat16_t* v_h_i = v+bidx*H*N*D+hidx*N*D+kvidx*block_kv_grid*D;
                    float* mask_b_i = mask+bidx*N+kvidx*block_kv_grid;
                    float* bias_r_h_i = bias_r+hidx*N*N+qidx*block_q*N+kvidx*block_kv_grid*block_size_q;
                    kutacc::pack_qk_hd16(k_h_i, k_block, block_size_kv);
                    //compute S,m, block_size_q和block_size_kv都需要是SVL_fp32的倍数。
                for(int row = 0; row < block_size_q; row += SVL_fp32)
                {
                    int q_pos = row*D;   
                    svfloat32_t sve_row_max1 = svld1_f32(ptrue, &m[row]); 
                    svfloat32_t sve_row_max2 = sve_row_max1;                    
                    for(int col = 0; col < block_size_kv; col += 4*SVL_fp32)
                    {
                        svzero_za();
                        int kp0 = col*D;
                        int kp1 = kp0+SVL_fp32*D;
                        int kp2 = kp1+SVL_fp32*D;
                        int kp3 = kp2+SVL_fp32*D;
                        //use mopa to compute qk^t
                        for(int d = 0; d < D; d += 2)
                        {
                            svbfloat16_t q0 = svld1_bf16(ptrue, &q_block[q_pos+d*SVL_fp32]);
                            svbfloat16_t k0 = svld1_bf16(ptrue, &k_block[kp0+d*SVL_fp32]);
                            svmopa_za32_bf16_m(0, ptrue, ptrue, q0, k0);
                            svbfloat16_t k1 = svld1_bf16(ptrue, &k_block[kp1+d*SVL_fp32]);
                            svmopa_za32_bf16_m(1, ptrue, ptrue, q0, k1);    
                            svbfloat16_t k2 = svld1_bf16(ptrue, &k_block[kp2+d*SVL_fp32]);                                
                            svmopa_za32_bf16_m(2, ptrue, ptrue, q0, k2);  
                            svbfloat16_t k3 = svld1_bf16(ptrue, &k_block[kp3+d*SVL_fp32]);  
                            svmopa_za32_bf16_m(3, ptrue, ptrue, q0, k3);                       
                        }
                        //multiply compute qk^t*scale+mask+bias, compute m
                        float* m0 = mask_b_i+col;
                        float* m1 = m0+SVL_fp32;
                        float* m2 = m1+SVL_fp32;  
                        float* m3 = m2+SVL_fp32;  
                        
                        float* s0 = S+row*block_size_kv+col*SVL_fp32;
                        float* s1 = s0+SVL_fp32*SVL_fp32;
                        float* s2 = s1+SVL_fp32*SVL_fp32;
                        float* s3 = s2+SVL_fp32*SVL_fp32;

                        for(int trow = 0; trow < SVL_fp32; trow++)
                        {
                            svfloat32_t s_d0 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 0, trow);
                            svfloat32_t s_d1 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 1, trow);
                            svfloat32_t s_d2 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 2, trow);
                            svfloat32_t s_d3 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 3, trow);

                            svfloat32_t m_d0 = svdup_f32(m0[trow]);
                            svfloat32_t m_d1 = svdup_f32(m1[trow]);
                            svfloat32_t m_d2 = svdup_f32(m2[trow]);
                            svfloat32_t m_d3 = svdup_f32(m3[trow]);

                            s_d0 = svmul_f32_m(ptrue, s_d0, scal_vec);
                            s_d1 = svmul_f32_m(ptrue, s_d1, scal_vec);
                            s_d2 = svmul_f32_m(ptrue, s_d2, scal_vec);
                            s_d3 = svmul_f32_m(ptrue, s_d3, scal_vec);

                            s_d0 = svadd_f32_m(ptrue, s_d0, m_d0);
                            s_d1 = svadd_f32_m(ptrue, s_d1, m_d1);
                            s_d2 = svadd_f32_m(ptrue, s_d2, m_d2);
                            s_d3 = svadd_f32_m(ptrue, s_d3, m_d3);

                            svst1_f32(ptrue, s0+trow*SVL_fp32, s_d0);
                            svst1_f32(ptrue, s1+trow*SVL_fp32, s_d1);
                            svst1_f32(ptrue, s2+trow*SVL_fp32, s_d2);
                            svst1_f32(ptrue, s3+trow*SVL_fp32, s_d3);

                            s_d0 = svmax_f32_m(ptrue, s_d0, s_d1);
                            s_d2 = svmax_f32_m(ptrue, s_d2, s_d3);
                            sve_row_max1 = svmax_f32_m(ptrue, sve_row_max1, s_d0);
                            sve_row_max2 = svmax_f32_m(ptrue, sve_row_max2, s_d2);	
                        }     
                    }
                    sve_row_max1 = svmax_f32_m(ptrue, sve_row_max1, sve_row_max2);
                    svst1_f32(ptrue, &m[row], sve_row_max1); 
                }
                //compute P,O
                for(int row = 0; row < block_size_q; row += SVL_fp32)
                {
                    float* s0 = S+row*block_size_kv;
                    bfloat16_t* p0 = P+row*block_size_kv;
                    float* b0 = bias_r_h_i+row*block_size_kv;
                    float* b1 = b0+SVL_fp32;

                    svfloat32_t row_max = svld1_f32(ptrue, &m[row]);
                    svfloat32_t row_max_prev = svld1_f32(ptrue, &m_last[row]);
                    row_max_prev = svsub_f32_m(ptrue, row_max_prev, row_max);
                    row_max_prev = kutacc::fast_exp(ptrue, row_max_prev);
                    svst1_f32(ptrue, &exp_diff[row], row_max_prev);
                    svst1_f32(ptrue, &m_last[row], row_max);
                    svfloat32_t row_sum = zero_vec_f32;
                    //compute and pack P matrix
                    for(int col = 0; col < block_size_kv; col += 2)
                    {
                        svfloat32_t b_d0 = svld1_f32(ptrue, b0+col*SVL_fp32);
                        svfloat32_t b_d1 = svld1_f32(ptrue, b1+col*SVL_fp32);
                        svfloat32_t s_d0 = svld1_f32(ptrue, s0+col*SVL_fp32);
                        svfloat32_t s_d1 = svld1_f32(ptrue, s0+(col+1)*SVL_fp32);
                        b_d0 = svsub_f32_m(ptrue, b_d0, row_max);
                        b_d1 = svsub_f32_m(ptrue, b_d1, row_max);
                        s_d0 = svadd_f32_m(ptrue, s_d0, b_d0);
                        s_d1 = svadd_f32_m(ptrue, s_d1, b_d1);
                        s_d0 = kutacc::fast_exp(ptrue, s_d0);
                        s_d1 = kutacc::fast_exp(ptrue, s_d1);                             
                        row_sum = svadd_f32_m(ptrue, row_sum, svadd_f32_m(ptrue, s_d0, s_d1));
                        svbfloat16_t p_d0 = svcvt_bf16_f32_x(ptrue, s_d0);
                        p_d0 = svcvtnt_bf16_f32_x(p_d0, ptrue, s_d1);
                        svst1_bf16(ptrue, p0+col*SVL_fp32, p_d0); 
                    }
                        //update l
                        svfloat32_t l_d0 = svld1_f32(ptrue, &l[row]);
                        l_d0 =svadd_f32_m(ptrue, svmul_f32_m(ptrue, l_d0, row_max_prev), row_sum);
                        svst1_f32(ptrue, &l[row], l_d0);
                    }
                    //pack v_block with bf16 accuracy
                    bfloat16_t* v0 = v_block;
                    for(int row = 0; row < block_size_kv; row += 2)
                    {
                        svbfloat16_t v_d0 = svld1_bf16(ptrue, v_h_i+row*D);
                        svbfloat16_t v_d1 = svld1_bf16(ptrue, v_h_i+row*D+SVL_fp32);
                        v_d0 = svzip1_bf16(v_d0, v_d1);
                        svst1_bf16(ptrue, v0, v_d0);
                        v0 += SVL_bf16;
                    }

                    //compute O
                    for(int row = 0; row < block_size_q; row += 4*SVL_fp32)
                    {
                        float* o0 = o_block+row*D;
                        float* o1 = o0+SVL_fp32*D;
                        float* o2 = o1+SVL_fp32*D;
                        float* o3 = o2+SVL_fp32*D;
                        int o_offset = 0;
                        //multiply exp_diff with O^(j-1) and fill O^(j-1) to the Za tile
                        for(int trow = 0; trow < SVL_fp32; trow++)
                        {
                            svfloat32_t o_d0 = svld1_f32(ptrue, o0+o_offset);
                            o_d0 = svmul_n_f32_m(ptrue, o_d0, exp_diff[row+trow]);
                            svfloat32_t o_d1 = svld1_f32(ptrue, o1+o_offset);
                            o_d1 = svmul_n_f32_m(ptrue, o_d1, exp_diff[row+trow+SVL_fp32]);
                            svfloat32_t o_d2 = svld1_f32(ptrue, o2+o_offset);
                            o_d2 = svmul_n_f32_m(ptrue, o_d2, exp_diff[row+trow+2*SVL_fp32]); 
                            svfloat32_t o_d3 = svld1_f32(ptrue, o3+o_offset);
                            o_d3 = svmul_n_f32_m(ptrue, o_d3, exp_diff[row+trow+3*SVL_fp32]); 

                            svwrite_hor_za32_f32_m(0, trow, ptrue, o_d0);
                            svwrite_hor_za32_f32_m(1, trow, ptrue, o_d1); 
                            svwrite_hor_za32_f32_m(2, trow, ptrue, o_d2);       
                            svwrite_hor_za32_f32_m(3, trow, ptrue, o_d3); 

                            o_offset += SVL_fp32;                                  
                        }

                        bfloat16_t* p0 = P+row*block_size_kv;
                        bfloat16_t* p1 = p0 +SVL_fp32*block_size_kv;
                        bfloat16_t* p2 = p1 +SVL_fp32*block_size_kv;
                        bfloat16_t* p3 = p2 +SVL_fp32*block_size_kv;

                        bfloat16_t* v0 = v_block;
                        //exp_diff*O^{j-1}+PV
                        for(int col = 0; col < block_size_kv; col += 2)
                        {
                            svbfloat16_t v_d0 = svld1_bf16(ptrue, v0);

                            svbfloat16_t p_d00 = svld1_bf16(ptrue, p0);
                            svmopa_za32_bf16_m(0, ptrue, ptrue, p_d00, v_d0);

                            svbfloat16_t p_d10 = svld1_bf16(ptrue, p1);
                            svmopa_za32_bf16_m(1, ptrue, ptrue, p_d10, v_d0);

                            svbfloat16_t p_d20 = svld1_bf16(ptrue, p2);
                            svmopa_za32_bf16_m(2, ptrue, ptrue, p_d20, v_d0);  

                            svbfloat16_t p_d30 = svld1_bf16(ptrue, p3);
                            svmopa_za32_bf16_m(3, ptrue, ptrue, p_d30, v_d0);  

                            p0 += SVL_bf16;
                            p1 += SVL_bf16;
                            p2 += SVL_bf16;
                            p3 += SVL_bf16;
                            v0 += SVL_bf16;                            
                        }
                        o_offset = 0;
                        //write O^j to HBM
                        for(int trow = 0; trow < SVL_fp32; ++trow)
                        {
                            svst1_hor_za32(0, trow, ptrue, o0+o_offset);
                            svst1_hor_za32(1, trow, ptrue, o1+o_offset);
                            svst1_hor_za32(2, trow, ptrue, o2+o_offset);
                            svst1_hor_za32(3, trow, ptrue, o3+o_offset);
                            o_offset += SVL_fp32;
                        }
                    }
                }
                //write O/l of size B_r*D to HBM
                int tot_len = block_size_q*D;
                float* o0_fp32 = o_block;
                bfloat16_t* o0_bf16 = o_h_i;
                for(int idx = 0; idx < tot_len; idx += SVL_bf16)
                {
                    float div0 = 1/l[idx/D];
                    float div1 = 1/l[(idx+SVL_fp32)/D];
                    svfloat32_t o_d0 = svld1_f32(ptrue, o0_fp32);
                    svfloat32_t o_d1 = svld1_f32(ptrue, o0_fp32+SVL_fp32);

                    o_d0 = svmul_n_f32_m(ptrue, o_d0, div0);
                    o_d1 = svmul_n_f32_m(ptrue, o_d1, div1);
                    svbfloat16_t o_dh0 = svcvt_bf16_f32_x(ptrue, o_d0);
                    svbfloat16_t o_dh1 = svcvt_bf16_f32_x(ptrue, o_d1);
                    o_dh0 = svuzp1_bf16(o_dh0,o_dh1);
                    svst1_bf16(ptrue, o0_bf16, o_dh0);
                    o0_fp32 += SVL_bf16;
                    o0_bf16 += SVL_bf16;
                }
            }
    SME_OFF();  
    });
}

void grid_flash_attn_mask_bias_bf16_hd32(int B, int H, int N, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, float* mask, float* bias_r, float scale, float m_min)
{
    int nblock_q = (N%block_q)?(N/block_q+1):(N/block_q);
    int nblock_kv = (N%block_kv_grid)?(N/block_kv_grid+1):(N/block_kv_grid);
    int task_num = B*H*nblock_q;
    int max_threads = kutacc::get_thread_num();
    int avg_task = task_num / max_threads;
    int busy_thread = task_num-avg_task*max_threads; 

    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
        SME_ON(); 
        int tid = kutacc::get_thread_id();
        int task_num_i, start_idx, end_idx;
        if(tid < busy_thread)
        {
            task_num_i = avg_task+1;
            start_idx = tid*task_num_i;
            end_idx = start_idx+task_num_i;
        }
        else
        {
            task_num_i = avg_task;
            start_idx = busy_thread*(task_num_i+1)+(tid-busy_thread)*task_num_i;
            end_idx = start_idx+task_num_i;
        }

        alignas(64) float l[block_q];
        alignas(64) float m_last[block_q];
        alignas(64) float m[block_q];              
        alignas(64) float S[block_q*block_kv_grid];
        alignas(64) float exp_diff[block_q];
        alignas(64) float o_block[block_q*D];
        // alignas(64) float bias_block[block_q*N];
        alignas(64) bfloat16_t P[block_q*block_kv_grid];
        alignas(64) bfloat16_t k_block[block_kv_grid*D];
        alignas(64) bfloat16_t v_block[block_kv_grid*D];
        alignas(64) bfloat16_t q_block[block_q*D];
        svbool_t ptrue = svptrue_b16();
        svfloat32_t scal_vec = svdup_f32(scale);
        svfloat32_t zero_vec_f32 = svdup_f32(0.0f);

        for(int idx = start_idx; idx < end_idx; ++idx)
        {
            int qidx = idx%nblock_q;
            int hidx = (idx/nblock_q)%H;
            int bidx = idx/nblock_q/H;

            bfloat16_t* q_h_i = q+bidx*H*N*D+hidx*N*D+qidx*block_q*D;
            bfloat16_t* o_h_i = out+bidx*H*N*D+hidx*N*D+qidx*block_q*D;
            float* bias_h = bias_r+hidx*N*N+qidx*block_q*N;

            int block_size_q = (qidx<nblock_q-1)?block_q:(N-block_q*qidx);

            memset(l, 0, sizeof(l));
            std::fill(m_last, m_last+block_size_q, m_min);
            std::fill(m, m+block_size_q, m_min);                
            memset(o_block, 0, sizeof(o_block)); 
            kutacc::pack_qk_hd32(q_h_i, q_block, block_size_q);
            for(int kvidx = 0; kvidx < nblock_kv; ++kvidx)
            {
                int block_size_kv = (kvidx<nblock_kv-1)?block_kv_grid:(N-block_kv_grid*kvidx);

                bfloat16_t* k_h_i = k+bidx*H*N*D+hidx*N*D+kvidx*block_kv_grid*D;
                bfloat16_t* v_h_i = v+bidx*H*N*D+hidx*N*D+kvidx*block_kv_grid*D;
                float* mask_b_i = mask+bidx*N+kvidx*block_kv_grid;
                float* bias_r_h_i = bias_r+hidx*N*N+qidx*block_q*N+kvidx*block_kv_grid*block_size_q;
                kutacc::pack_qk_hd32(k_h_i, k_block, block_size_kv);
                //compute S,m, block_size_q和block_size_kv都需要是SVL_fp32的倍数。
                for(int row = 0; row < block_size_q; row += SVL_fp32)
                {
                    int q_pos = row*D;   
                    svfloat32_t sve_row_max1 = svld1_f32(ptrue, &m[row]); 
                    svfloat32_t sve_row_max2 = sve_row_max1;                  
                    for(int col = 0; col < block_size_kv; col += 4*SVL_fp32)
                    {
                        svzero_za();
                        int kp0 = col*D;
                        int kp1 = kp0+SVL_fp32*D;
                        int kp2 = kp1+SVL_fp32*D;
                        int kp3 = kp2+SVL_fp32*D;

                        //use mopa to compute qk^t
                        for(int d = 0; d < D; d += 2)
                        {
                            svbfloat16_t q0 = svld1_bf16(ptrue, &q_block[q_pos+d*SVL_fp32]);
                            svbfloat16_t k0 = svld1_bf16(ptrue, &k_block[kp0+d*SVL_fp32]);
                            svmopa_za32_bf16_m(0, ptrue, ptrue, q0, k0);
                            svbfloat16_t k1 = svld1_bf16(ptrue, &k_block[kp1+d*SVL_fp32]);
                            svmopa_za32_bf16_m(1, ptrue, ptrue, q0, k1);    
                            svbfloat16_t k2 = svld1_bf16(ptrue, &k_block[kp2+d*SVL_fp32]);                                
                            svmopa_za32_bf16_m(2, ptrue, ptrue, q0, k2);  
                            svbfloat16_t k3 = svld1_bf16(ptrue, &k_block[kp3+d*SVL_fp32]);  
                            svmopa_za32_bf16_m(3, ptrue, ptrue, q0, k3); 
                        }
                        //multiply compute qk^t*scale+mask+bias, compute m

                        float* m0 = mask_b_i+col;
                        float* m1 = m0+SVL_fp32;
                        float* m2 = m1+SVL_fp32;  
                        float* m3 = m2+SVL_fp32;  
                        
                        float* s0 = S+row*block_size_kv+col*SVL_fp32;
                        float* s1 = s0+SVL_fp32*SVL_fp32;
                        float* s2 = s1+SVL_fp32*SVL_fp32;
                        float* s3 = s2+SVL_fp32*SVL_fp32;

                        for(int trow = 0; trow < SVL_fp32; trow++)
                        {

                            svfloat32_t m_d0 = svdup_f32(m0[trow]);
                            svfloat32_t m_d1 = svdup_f32(m1[trow]);
                            svfloat32_t m_d2 = svdup_f32(m2[trow]);
                            svfloat32_t m_d3 = svdup_f32(m3[trow]);

                            svfloat32_t s_d0 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 0, trow);
                            svfloat32_t s_d1 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 1, trow);
                            s_d0 = svmul_f32_m(ptrue, s_d0, scal_vec);
                            s_d1 = svmul_f32_m(ptrue, s_d1, scal_vec);
                            svfloat32_t s_d2 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 2, trow);
                            svfloat32_t s_d3 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 3, trow);
                            s_d2 = svmul_f32_m(ptrue, s_d2, scal_vec);
                            s_d3 = svmul_f32_m(ptrue, s_d3, scal_vec);

                            s_d0 = svadd_f32_m(ptrue, m_d0, s_d0);
                            s_d1 = svadd_f32_m(ptrue, m_d1, s_d1);
                            s_d2 = svadd_f32_m(ptrue, m_d2, s_d2);
                            s_d3 = svadd_f32_m(ptrue, m_d3, s_d3);

                            svst1_f32(ptrue, s0+trow*SVL_fp32, s_d0);
                            svst1_f32(ptrue, s1+trow*SVL_fp32, s_d1);
                            svst1_f32(ptrue, s2+trow*SVL_fp32, s_d2);
                            svst1_f32(ptrue, s3+trow*SVL_fp32, s_d3);

                            svfloat32_t s_d01 = svmax_f32_m(ptrue, s_d0, s_d1);
                            svfloat32_t s_d23 = svmax_f32_m(ptrue, s_d2, s_d3);
                            sve_row_max1 = svmax_f32_m(ptrue, sve_row_max1, s_d01);
                            sve_row_max2 = svmax_f32_m(ptrue, sve_row_max2, s_d23);
                        }     
                    }
                    sve_row_max1 = svmax_f32_m(ptrue, sve_row_max1, sve_row_max2);
                    svst1_f32(ptrue, &m[row], sve_row_max1); 
                }
                //compute P,O
                for(int row = 0; row < block_size_q; row += SVL_fp32)
                {
                    float* s0 = S+row*block_size_kv;
                    bfloat16_t* p0 = P+row*block_size_kv;
                    float* b0 = bias_r_h_i+row*block_size_kv;
                    float* b1 = b0+SVL_fp32;

                    svfloat32_t row_max = svld1_f32(ptrue, &m[row]);
                    svfloat32_t row_max_prev = svld1_f32(ptrue, &m_last[row]);
                    row_max_prev = svsub_f32_m(ptrue, row_max_prev, row_max);
                    row_max_prev = kutacc::fast_exp(ptrue, row_max_prev);
                    svst1_f32(ptrue, &exp_diff[row], row_max_prev);
                    svst1_f32(ptrue, &m_last[row], row_max);
                    svfloat32_t row_sum = zero_vec_f32;
                    //compute and pack P matrix
                    for(int col = 0; col < block_size_kv; col += 2)
                    {
                        svfloat32_t b_d0 = svld1_f32(ptrue, b0+col*SVL_fp32);
                        svfloat32_t b_d1 = svld1_f32(ptrue, b1+col*SVL_fp32);
                        svfloat32_t s_d0 = svld1_f32(ptrue, s0+col*SVL_fp32);
                        svfloat32_t s_d1 = svld1_f32(ptrue, s0+(col+1)*SVL_fp32);
                        b_d0 = svsub_f32_m(ptrue, b_d0, row_max);
                        b_d1 = svsub_f32_m(ptrue, b_d1, row_max);
                        s_d0 = svadd_f32_m(ptrue, s_d0, b_d0);
                        s_d1 = svadd_f32_m(ptrue, s_d1, b_d1);
                        s_d0 = kutacc::fast_exp(ptrue, s_d0);
                        s_d1 = kutacc::fast_exp(ptrue, s_d1);                             
                        row_sum = svadd_f32_m(ptrue, row_sum, svadd_f32_m(ptrue, s_d0, s_d1));
                        svbfloat16_t p_d0 = svcvt_bf16_f32_x(ptrue, s_d0);
                        p_d0 = svcvtnt_bf16_f32_x(p_d0, ptrue, s_d1);
                        svst1_bf16(ptrue, p0+col*SVL_fp32, p_d0); 
                    }
                    //update l
                    svfloat32_t l_d0 = svld1_f32(ptrue, &l[row]);
                    l_d0 =svadd_f32_m(ptrue, svmul_f32_m(ptrue, l_d0, row_max_prev), row_sum);
                    svst1_f32(ptrue, &l[row], l_d0);
                    
                }
                //pack v_block with bf16 accuracy
                bfloat16_t* v0 = v_block;
                for(int row = 0; row < block_size_kv; row += 2)
                {
                    svbfloat16_t v_d0 = svld1_bf16(ptrue, v_h_i+row*D);
                    svbfloat16_t v_d1 = svld1_bf16(ptrue, v_h_i+row*D+SVL_bf16);
                    svbfloat16_t v_dz0 = svzip1_bf16(v_d0, v_d1);
                    svbfloat16_t v_dz1 = svzip2_bf16(v_d0, v_d1);
                    svst1_bf16(ptrue, v0, v_dz0);
                    svst1_bf16(ptrue, v0+SVL_bf16, v_dz1);
                    v0 += 2*SVL_bf16;
                }
                //std::cout << "startO" << std::endl;
                //compute O
                for(int row = 0; row < block_size_q; row += 2*SVL_fp32)
                {
                    float* o0 = o_block+row*D;
                    float* o1 = o0+SVL_fp32;
                    float* o2 = o0+SVL_fp32*D;
                    float* o3 = o2+SVL_fp32;
                    int o_offset = 0;
                    //multiply exp_diff with O^(j-1) and fill O^(j-1) to the Za tile
                    for(int trow = 0; trow < SVL_fp32; trow++)
                    {
                        svfloat32_t o_d0 = svld1_f32(ptrue, o0+o_offset);
                        o_d0 = svmul_n_f32_m(ptrue, o_d0, exp_diff[row+trow]);
                        svfloat32_t o_d1 = svld1_f32(ptrue, o1+o_offset);
                        o_d1 = svmul_n_f32_m(ptrue, o_d1, exp_diff[row+trow]);
                        svfloat32_t o_d2 = svld1_f32(ptrue, o2+o_offset);
                        o_d2 = svmul_n_f32_m(ptrue, o_d2, exp_diff[row+trow+SVL_fp32]); 
                        svfloat32_t o_d3 = svld1_f32(ptrue, o3+o_offset);
                        o_d3 = svmul_n_f32_m(ptrue, o_d3, exp_diff[row+trow+SVL_fp32]); 

                        svwrite_hor_za32_f32_m(0, trow, ptrue, o_d0);
                        svwrite_hor_za32_f32_m(1, trow, ptrue, o_d1); 
                        svwrite_hor_za32_f32_m(2, trow, ptrue, o_d2);       
                        svwrite_hor_za32_f32_m(3, trow, ptrue, o_d3); 

                        o_offset += D;                                  
                    }

                    bfloat16_t* p0 = P+row*block_size_kv;
                    bfloat16_t* p1 = p0 +SVL_fp32*block_size_kv;

                    bfloat16_t* v0 = v_block;
                    bfloat16_t* v1 = v_block+SVL_bf16;
                    //exp_diff*O^{j-1}+PV
                    for(int col = 0; col < block_size_kv; col += 2)
                    {
                        svbfloat16_t p_d00 = svld1_bf16(ptrue, p0+col*SVL_fp32);
                        svbfloat16_t v_d0 = svld1_bf16(ptrue, v0+col*SVL_bf16);
                        svbfloat16_t p_d10 = svld1_bf16(ptrue, p1+col*SVL_fp32);
                        svbfloat16_t v_d1 = svld1_bf16(ptrue, v1+col*SVL_bf16);

                        svmopa_za32_bf16_m(0, ptrue, ptrue, p_d00, v_d0);
                        svmopa_za32_bf16_m(1, ptrue, ptrue, p_d00, v_d1);
                        svmopa_za32_bf16_m(2, ptrue, ptrue, p_d10, v_d0);
                        svmopa_za32_bf16_m(3, ptrue, ptrue, p_d10, v_d1);                                                         
                    }
                    o_offset = 0;
                    //write O^j to HBM
                    for(int trow = 0; trow < SVL_fp32; ++trow)
                    {
                        svst1_hor_za32(0, trow, ptrue, o0+o_offset);
                        svst1_hor_za32(1, trow, ptrue, o1+o_offset);
                        svst1_hor_za32(2, trow, ptrue, o2+o_offset);
                        svst1_hor_za32(3, trow, ptrue, o3+o_offset);
                        o_offset += D;
                    }
                }                        
            }
            //write O/l of size B_r*D to HBM
            int tot_len = block_size_q*D;
            float* o0_fp32 = o_block;
            bfloat16_t* o0_bf16 = o_h_i;
            for(int idx = 0; idx < tot_len; idx += SVL_bf16)
            {
                float div0 = 1/l[idx/D];
                float div1 = 1/l[(idx+SVL_fp32)/D];
                svfloat32_t o_d0 = svld1_f32(ptrue, o0_fp32);
                svfloat32_t o_d1 = svld1_f32(ptrue, o0_fp32+SVL_fp32);

                o_d0 = svmul_n_f32_m(ptrue, o_d0, div0);
                o_d1 = svmul_n_f32_m(ptrue, o_d1, div1);
                svbfloat16_t o_dh0 = svcvt_bf16_f32_x(ptrue, o_d0);
                svbfloat16_t o_dh1 = svcvt_bf16_f32_x(ptrue, o_d1);
                o_dh0 = svuzp1_bf16(o_dh0,o_dh1);
                svst1_bf16(ptrue, o0_bf16, o_dh0);
                o0_fp32 += SVL_bf16;
                o0_bf16 += SVL_bf16;
            }
        }
    SME_OFF(); 
    });
}
} //namespace kutacc