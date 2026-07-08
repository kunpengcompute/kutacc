#include "../linear.h"
#include "activation/sigmoid.h"
#include "math/fast_exp.h"

namespace kutacc{
void cross_attention_out_pack(bfloat16_t* weighted_avg, bfloat16_t* gate, bfloat16_t* pack_wavg, int ndim, int batch, int seq_len, int n_res, int nchannels)
{
    if(ndim == 4){
        kutacc::parallel_for(0, batch, 1, [&](int64_t start, int64_t end) {
            SME_ON();
            int vl = SVL_fp32;
            int vl_bf16 = SVL_bf16;
            svbool_t pg = svptrue_b32();
            svbool_t pg16 = svptrue_b16();
            for (int64_t bi = start; bi < end; bi++)
            {
                for (int64_t qi = 0; qi < seq_len; qi++)
                {
                    for (int ri = 0 ; ri < n_res ; ri += vl_bf16) 
                    {
                        for (int ci = 0; ci < nchannels; ci += vl * 4) 
                        {
                            auto wavg_data = weighted_avg+ bi * seq_len * n_res * nchannels + qi * n_res * nchannels + ri * nchannels;
                            auto gate_data = gate + bi * seq_len * n_res * nchannels + qi * n_res * nchannels + ri * nchannels;
                            auto pack_wavg_data = pack_wavg + bi * seq_len * n_res * nchannels + qi * n_res * nchannels;
                            svzero_za();
                            for(int row_idx = 0; row_idx < vl_bf16; row_idx ++)
                            {
                                svfloat32_t w0 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci);
                                svfloat32_t w1 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl);
                                svfloat32_t w2 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl * 2);
                                svfloat32_t w3 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl * 3);
                                
                                svfloat32_t g0 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci);
                                svfloat32_t g1 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl);
                                svfloat32_t g2 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl * 2);
                                svfloat32_t g3 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl * 3);
                                
                                w0 = svmul_x(pg, w0, kutacc::Sigmoid::call(pg, g0));
                                w1 = svmul_x(pg, w1, kutacc::Sigmoid::call(pg, g1));
                                w2 = svmul_x(pg, w2, kutacc::Sigmoid::call(pg, g2));
                                w3 = svmul_x(pg, w3, kutacc::Sigmoid::call(pg, g3));
                                svbfloat16_t w0_bf16 = svcvt_bf16_f32_x(pg,w0);
                                svbfloat16_t w1_bf16 = svcvt_bf16_f32_x(pg,w1);
                                svbfloat16_t w2_bf16 = svcvt_bf16_f32_x(pg,w2);
                                svbfloat16_t w3_bf16 = svcvt_bf16_f32_x(pg,w3);

                                svbfloat16_t unz1 = svuzp1_bf16(w0_bf16,w1_bf16);
                                svbfloat16_t unz2 = svuzp1_bf16(w2_bf16,w3_bf16);

                                svwrite_hor_za16_bf16_m(0, row_idx, pg16, unz1);
                                svwrite_hor_za16_bf16_m(1, row_idx, pg16, unz2);
                                wavg_data += nchannels;
                                gate_data += nchannels;
                            }
                            for(int col_idx = 0; col_idx < vl_bf16; col_idx +=2)
                            {
                                svbfloat16_t unz1,unz2,z1,z2;
                                unz1 = svread_ver_za16_bf16_m(unz1, pg16, 0, col_idx);
                                unz2 = svread_ver_za16_bf16_m(unz2, pg16, 0, col_idx+1); 

                                z1 = svzip1_bf16(unz1, unz2); 
                                z2 = svzip2_bf16(unz1, unz2); 
                                svst1_bf16(pg16,pack_wavg_data + ri * nchannels + (col_idx+ ci)*vl, z1); 
                                svst1_bf16(pg16,pack_wavg_data + (ri + vl) * nchannels + (col_idx+ ci)*vl, z2);

                                unz1 = svread_ver_za16_bf16_m(unz1, pg16, 1, col_idx);  
                                unz2 = svread_ver_za16_bf16_m(unz2, pg16, 1, col_idx+1); 

                                z1 = svzip1_bf16(unz1,unz2);
                                z2 = svzip2_bf16(unz1,unz2);
                                svst1_bf16(pg16,pack_wavg_data + ri * nchannels + (col_idx+ ci + vl_bf16)*vl , z1); 
                                svst1_bf16(pg16,pack_wavg_data + (ri + vl) * nchannels + (col_idx+ ci + vl_bf16)*vl, z2);
                            }
                        }
                        
                    }
                }
                SME_OFF();
            }
        });
    }
    else{
        kutacc::parallel_for(0, batch, 1, [&](int64_t start, int64_t end) {
            SME_ON();
            int vl = SVL_fp32;
            int vl_bf16 = SVL_bf16;
            svbool_t pg = svptrue_b32();
            svbool_t pg16 = svptrue_b16();
            for (int64_t bi = start; bi < end; bi++){
                for (int qi = 0 ; qi < seq_len; qi += vl_bf16) 
                {
                    for (int ci = 0; ci < nchannels; ci += vl * 4) 
                    {
                        auto wavg_data = weighted_avg + bi * seq_len * nchannels + qi * nchannels;
                        auto gate_data = gate + bi * seq_len * nchannels + qi  * nchannels;
                        auto pack_wavg_data = pack_wavg + bi * seq_len * nchannels;
                        svzero_za();
                        for(int row_idx = 0; row_idx < vl_bf16; row_idx ++)
                        {
                            svfloat32_t w0 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci);
                            svfloat32_t w1 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl);
                            svfloat32_t w2 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl * 2);
                            svfloat32_t w3 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl * 3);
                            
                            svfloat32_t g0 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci);
                            svfloat32_t g1 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl);
                            svfloat32_t g2 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl * 2);
                            svfloat32_t g3 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl * 3);
                            
                            w0 = svmul_x(pg, w0, kutacc::Sigmoid::call(pg, g0));
                            w1 = svmul_x(pg, w1, kutacc::Sigmoid::call(pg, g1));
                            w2 = svmul_x(pg, w2, kutacc::Sigmoid::call(pg, g2));
                            w3 = svmul_x(pg, w3, kutacc::Sigmoid::call(pg, g3));
                            svbfloat16_t w0_bf16 = svcvt_bf16_f32_x(pg,w0);
                            svbfloat16_t w1_bf16 = svcvt_bf16_f32_x(pg,w1);
                            svbfloat16_t w2_bf16 = svcvt_bf16_f32_x(pg,w2);
                            svbfloat16_t w3_bf16 = svcvt_bf16_f32_x(pg,w3);

                            svbfloat16_t unz1 = svuzp1_bf16(w0_bf16,w1_bf16);
                            svbfloat16_t unz2 = svuzp1_bf16(w2_bf16,w3_bf16);

                            svwrite_hor_za16_bf16_m(0, row_idx, pg16, unz1);
                            svwrite_hor_za16_bf16_m(1, row_idx, pg16, unz2);
                            wavg_data += nchannels;
                            gate_data += nchannels;
                        }
                        for(int col_idx = 0; col_idx < vl_bf16; col_idx +=2)
                        {
                            svbfloat16_t unz1,unz2,z1,z2;
                            unz1 = svread_ver_za16_bf16_m(unz1, pg16, 0, col_idx);
                            unz2 = svread_ver_za16_bf16_m(unz2, pg16, 0, col_idx+1); 

                            z1 = svzip1_bf16(unz1, unz2); 
                            z2 = svzip2_bf16(unz1, unz2); 
                            svst1_bf16(pg16,pack_wavg_data + qi * nchannels + (col_idx+ ci)*vl, z1); 
                            svst1_bf16(pg16,pack_wavg_data + (qi + vl) * nchannels + (col_idx+ ci)*vl, z2);

                            unz1 = svread_ver_za16_bf16_m(unz1, pg16, 1, col_idx);  
                            unz2 = svread_ver_za16_bf16_m(unz2, pg16, 1, col_idx+1); 

                            z1 = svzip1_bf16(unz1,unz2);
                            z2 = svzip2_bf16(unz1,unz2);
                            svst1_bf16(pg16,pack_wavg_data + qi * nchannels + (col_idx+ ci + vl_bf16)*vl , z1); 
                            svst1_bf16(pg16,pack_wavg_data + (qi + vl) * nchannels + (col_idx+ ci + vl_bf16)*vl, z2);
                        }
                    }
                    
                }
            }
            SME_OFF();
        });
    }
}

void cross_attention_linear_transpose(bfloat16_t* x_q, bfloat16_t* x_k, bfloat16_t* q_proj, float* q_proj_bias, bfloat16_t* k_proj, bfloat16_t* v_proj, bfloat16_t* g_proj,\
     bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* g, int batch_q, int seq_len_q, int batch_k, int seq_len_k, int nheads, int head_size)
{
    int M_q = batch_q * seq_len_q, M_k = batch_k * seq_len_k , N = nheads * head_size, K = nheads * head_size;

    const int row_block_size = ROW_BLOCK_SIZE;
    const int col_block_size = COL_BLOCK_SIZE;
    int max_threads = kutacc::get_thread_num();
    int avg_threads = max_threads / 2;
    int task_num_q = (M_q * N) / (row_block_size * col_block_size);
    int task_num_k = (M_k * N) / (row_block_size * col_block_size);
    int avg_task = (task_num_q + task_num_k) / avg_threads;
    int busy_thread = (task_num_q + task_num_k) - avg_task * avg_threads;
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
        if(tid < avg_threads)
        {
            for(int task_idx = start_idx; task_idx < start_idx + task_num; task_idx++ )
            {
                if(task_idx < task_num_q)
                {
                    int row_index = (task_idx / row_task_num) * row_block_size;
                    int col_index = (task_idx % row_task_num) * col_block_size;
                    bfloat16_t *block_part_x_q_base = x_q + row_index * K ;
                    kutacc::bgemm_bias_4vl_1vl_bf16_transpose_impl(block_part_x_q_base, q_proj + col_index * K, q_proj_bias + col_index, q, \
                    row_block_size, col_block_size, K, row_index, col_index, seq_len_q, nheads, head_size);
                }
                else
                {
                    int task_idx_local = task_idx - task_num_q;
                    int row_index = (task_idx_local / row_task_num) * row_block_size;
                    int col_index = (task_idx_local % row_task_num) * col_block_size;
                    bfloat16_t *block_part_x_k_base = x_k + row_index * K ;
                    kutacc::bgemm_4vl_1vl_bf16_transpose_impl(block_part_x_k_base, k_proj + col_index * K, k, row_block_size, col_block_size, \
                    K, row_index, col_index, seq_len_k, nheads, head_size);
                }
            }
        }
        else
        {
            for(int task_idx = start_idx; task_idx < start_idx + task_num; task_idx++ )
            {
                if(task_idx < task_num_q)
                {
                    int row_index = (task_idx / row_task_num) * row_block_size;
                    int col_index = (task_idx % row_task_num) * col_block_size;
                    bfloat16_t *block_part_x_q_base = x_q + row_index * K ;
                    kutacc::bgemm_4vl_1vl_bf16_impl(block_part_x_q_base, g_proj + col_index * K, g + row_index * N + col_index, row_block_size, col_block_size, K, N);
                }
                else
                {
                    int task_idx_local = task_idx - task_num_q;
                    int row_index = (task_idx_local / row_task_num) * row_block_size;
                    int col_index = (task_idx_local % row_task_num) * col_block_size;
                    bfloat16_t *block_part_x_k_base = x_k + row_index * K ;
                    kutacc::bgemm_4vl_1vl_bf16_transpose_impl(block_part_x_k_base, v_proj + col_index * K, v, row_block_size, col_block_size, \
                    K, row_index, col_index, seq_len_k, nheads, head_size);
                }
            }
        }
    });
}

template <typename T>
void cross_pack_pair_logits(T pair_logits_ori, float* pair_logits_r, int nblock_q, int nblock_kv, int N_a, int H, int Nq, int Nkv)
{
    #pragma omp parallel for collapse(3)
    for(int naidx = 0; naidx < N_a; ++naidx)
    {
        for(int hidx = 0; hidx < H; ++hidx)
        {
            for(int qidx = 0; qidx < nblock_q; ++qidx)
            {
                for(int kvidx = 0; kvidx < nblock_kv; ++kvidx)
                {
                    int block_size_q = (qidx<nblock_q-1)?block_q:(Nq-block_q*qidx);
                    int block_size_kv = (kvidx<nblock_kv-1)?block_kv:(Nkv-block_kv*kvidx);
                    T pl_h_i = pair_logits_ori+naidx*H*Nq*Nkv+hidx*Nq*Nkv+qidx*block_q*Nkv+kvidx*block_kv;
                    float* pl_r_h_i = pair_logits_r+naidx*H*Nq*Nkv+hidx*Nq*Nkv+qidx*block_q*Nkv+kvidx*block_kv*block_size_q;
                    
                    for(int i = 0; i < block_size_q; ++i)
                    {
                        for(int j = 0; j < block_size_kv; ++j)
                        {
                            pl_r_h_i[i*block_size_kv+j] = pl_h_i[i*Nkv+j];
                        }
                    }
                }
            }
        }
    }
}

void cross_flash_attn_mask_bias_bf16_hd32(int B, int N_a, int H, int Nq, int Nkv, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, float* bias, float* pair_logits_r, float scale, float m_min)
{
    int nblock_q = (Nq%block_q)?(Nq/block_q+1):(Nq/block_q);
    int nblock_kv = (Nkv%block_kv)?(Nkv/block_kv+1):(Nkv/block_kv);

    kutacc::parallel_for(0, B * H * nblock_q , 1, [&](int64_t start, int64_t end){
        for(int task_idx = start; task_idx < end; task_idx ++)
        {
            int bidx = task_idx / (H * nblock_q);
            int hidx = (task_idx % (H * nblock_q)) / nblock_q;
            int qidx = task_idx % nblock_q;
            SME_ON(); 
            bfloat16_t* q_h_i = q+bidx*H*Nq*D+hidx*Nq*D+qidx*block_q*D;
            bfloat16_t* o_h_i = out+bidx*H*Nq*D+hidx*Nq*D+qidx*block_q*D;

            int block_size_q = (qidx<nblock_q-1)?block_q:(Nq-block_q*qidx);
            // memset(o_h_i, 0, block_size_q*D*sizeof(bfloat16_t));
            alignas(64) float l[block_size_q];
            memset(l, 0, sizeof(l));
            alignas(64) float m_last[block_size_q];
            std::fill(m_last, m_last+block_size_q, m_min);
            alignas(64) float m[block_size_q];
            std::fill(m, m+block_size_q, m_min);                
            alignas(64) bfloat16_t P[block_q*block_kv];
            alignas(64) float exp_diff[block_size_q];

            alignas(64) bfloat16_t k_block[block_kv*D];
            alignas(64) bfloat16_t v_block[block_kv*D];
            alignas(64) float o_block[block_size_q*D];
            memset(o_block, 0, sizeof(o_block)); 
            //fill q, D should be even, block_size_q 整除SVL_fp32
            alignas(64) bfloat16_t q_block[block_size_q*D];
            for(int row = 0; row < block_size_q; row += SVL_fp32)
            {
                for(int col = 0; col < D; col += 2)
                {
                    int in_id = row*D+col*SVL_fp32;
                    for(int trow = 0; trow < SVL_fp32; trow++)
                    {
                        q_block[in_id+trow*2] = q_h_i[(row+trow)*D+col];
                        q_block[in_id+trow*2+1] = q_h_i[(row+trow)*D+col+1];
                    }
                }
            }
            svbool_t ptrue = svptrue_b16();
            svfloat32_t scal_vec = svdup_f32(scale);
            svfloat32_t zero_vec_f32 = svdup_f32(0.0f);
            for(int kvidx = 0; kvidx < nblock_kv; ++kvidx)
            {
                int block_size_kv = (kvidx<nblock_kv-1)?block_kv:(Nkv-block_kv*kvidx);
                bfloat16_t* k_h_i = k+bidx*H*Nkv*D+hidx*Nkv*D+kvidx*block_kv*D;
                bfloat16_t* v_h_i = v+bidx*H*Nkv*D+hidx*Nkv*D+kvidx*block_kv*D;
                float* bias_i = bias+(bidx%N_a)*Nq*Nkv+qidx*block_q*Nkv+kvidx*block_kv*block_size_q;
                float* pl_r_h_i = pair_logits_r+(bidx%N_a)*H*Nq*Nkv+hidx*Nq*Nkv+qidx*block_q*Nkv+kvidx*block_kv*block_size_q;
                //fill k, block_size_kv should be a multiple of 4*SVL_fp32                    
                for(int row = 0; row < block_size_kv; row += 4*SVL_fp32)
                {
                    for(int col = 0; col < D; col += 2)
                    {
                        int in_id = row*D+col*4*SVL_fp32;
                        for(int trow = 0; trow < 4*SVL_fp32; trow++)
                        {
                            k_block[in_id+2*trow] = k_h_i[(row+trow)*D+col];
                            k_block[in_id+2*trow+1] = k_h_i[(row+trow)*D+col+1];
                        }
                    }
                }   
                //compute S,m
                for(int row = 0; row < block_size_q; row += SVL_fp32)
                {
                    
                    for(int col = 0; col < block_size_kv; col += 4*SVL_fp32)
                    {
                        svzero_za();
                        int q_pos = row*D;
                        int k_pos = col/(4*SVL_fp32)*(4*SVL_fp32*D);
                        //use mopa to compute qk^t
                        for(int d = 0; d < D; d += 2)
                        {
                            svbfloat16_t q0 = svld1_bf16(ptrue, &q_block[q_pos+d*SVL_fp32]);
                            svbfloat16_t k0 = svld1_bf16(ptrue, &k_block[d*4*SVL_fp32+k_pos]);
                            svmopa_za32_bf16_m(0, ptrue, ptrue, q0, k0);
                            svbfloat16_t k1 = svld1_bf16(ptrue, &k_block[d*4*SVL_fp32+k_pos+2*SVL_fp32]);
                            svmopa_za32_bf16_m(1, ptrue, ptrue, q0, k1);    
                            svbfloat16_t k2 = svld1_bf16(ptrue, &k_block[d*4*SVL_fp32+k_pos+4*SVL_fp32]);
                            svmopa_za32_bf16_m(2, ptrue, ptrue, q0, k2);    
                            svbfloat16_t k3 = svld1_bf16(ptrue, &k_block[d*4*SVL_fp32+k_pos+6*SVL_fp32]);
                            svmopa_za32_bf16_m(3, ptrue, ptrue, q0, k3);                         
                        }
                        //multiply compute qk^t*scale+mask+bias, compute m
                        float* b0 = pl_r_h_i+row*block_size_kv+col;
                        float* b1 = b0+SVL_fp32;
                        float* b2 = b1+SVL_fp32;
                        float* b3 = b2+SVL_fp32;    

                        float* m0 = bias_i+row*block_size_kv+col;
                        float* m1 = m0+SVL_fp32;
                        float* m2 = m1+SVL_fp32;  
                        float* m3 = m2+SVL_fp32;  

                        for(int trow = 0; trow < SVL_fp32; trow++)
                        {
                            svfloat32_t s_d0 = svread_hor_za32_f32_m(zero_vec_f32, ptrue, 0, trow);
                            s_d0 = svmul_f32_m(ptrue, s_d0, scal_vec);
                            svfloat32_t b_d0 = svld1_f32(ptrue, b0);
                            svfloat32_t m_d0 = svld1_f32(ptrue, m0);
                            s_d0 = svadd_f32_m(ptrue, s_d0, svadd_f32_m(ptrue, m_d0, b_d0));
                            svwrite_hor_za32_f32_m(0, trow, ptrue, s_d0);                                

                            svfloat32_t s_d1 = svread_hor_za32_f32_m(zero_vec_f32, ptrue, 1, trow);
                            s_d1 = svmul_f32_m(ptrue, s_d1, scal_vec);
                            svfloat32_t b_d1 = svld1_f32(ptrue, b1);
                            svfloat32_t m_d1 = svld1_f32(ptrue, m1);
                            s_d1 = svadd_f32_m(ptrue, s_d1, svadd_f32_m(ptrue, m_d1, b_d1));
                            svwrite_hor_za32_f32_m(1, trow, ptrue, s_d1);

                            svfloat32_t s_d2 = svread_hor_za32_f32_m(zero_vec_f32, ptrue, 2, trow);
                            s_d2 = svmul_f32_m(ptrue, s_d2, scal_vec);
                            svfloat32_t b_d2 = svld1_f32(ptrue, b2);
                            svfloat32_t m_d2 = svld1_f32(ptrue, m2);
                            s_d2 = svadd_f32_m(ptrue, s_d2, svadd_f32_m(ptrue, m_d2, b_d2));
                            svwrite_hor_za32_f32_m(2, trow, ptrue, s_d2);

                            svfloat32_t s_d3 = svread_hor_za32_f32_m(zero_vec_f32, ptrue, 3, trow);
                            s_d3 = svmul_f32_m(ptrue, s_d3, scal_vec);
                            svfloat32_t b_d3 = svld1_f32(ptrue, b3);
                            svfloat32_t m_d3 = svld1_f32(ptrue, m3);
                            s_d3 = svadd_f32_m(ptrue, s_d3, svadd_f32_m(ptrue, m_d3, b_d3));
                            svwrite_hor_za32_f32_m(3, trow, ptrue, s_d3);

                            float val0 = svmaxv_f32(ptrue, s_d0);
                            float val1 = svmaxv_f32(ptrue, s_d1);
                            float val2 = svmaxv_f32(ptrue, s_d2);
                            float val3 = svmaxv_f32(ptrue, s_d3);
                            float max_val = std::max({val0, val1, val2, val3});
                            m[row+trow] = std::max(m[row+trow], max_val);

                            b0 += block_size_kv;
                            b1 += block_size_kv;
                            b2 += block_size_kv;
                            b3 += block_size_kv;
                            m0 += block_size_kv;
                            m1 += block_size_kv;
                            m2 += block_size_kv;
                            m3 += block_size_kv;                                
                        }
                    }
                    //P matrix
                    bfloat16_t* p0 = P+row*block_size_kv;
                    bfloat16_t* p1 = p0+SVL_fp32*SVL_fp32;
                    bfloat16_t* p2 = p1+SVL_fp32*SVL_fp32;
                    bfloat16_t* p3 = p2+SVL_fp32*SVL_fp32;
                    svfloat32_t row_max = svld1_f32(ptrue, &m[row]);
                    svfloat32_t row_max_prev = svld1_f32(ptrue, &m_last[row]);
                    row_max_prev = svsub_f32_m(ptrue, row_max_prev, row_max);
                    row_max_prev = kutacc::fast_exp(ptrue, row_max_prev);
                    //row_max_prev = _ZGVsNxv_exp2f(svmul_n_f32_m(ptrue, row_max_prev, log2e));   
                    svst1_f32(ptrue, &exp_diff[row], row_max_prev);
                    svst1_f32(ptrue, &m_last[row], row_max);
                    svfloat32_t row_sum = zero_vec_f32;  
                    //compute and pack P matrix
                    svfloat32_t s_d00, s_d01, s_d10, s_d11, s_d20, s_d21, s_d30, s_d31;
                    svbfloat16_t p_d0, p_d1, p_d2, p_d3;
                    for(int col = 0; col < SVL_fp32; col += 2)
                    {

                        s_d00 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 0, col);
                        s_d01 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 0, col+1);
                        s_d00 = svsub_f32_m(ptrue, s_d00, row_max);
                        s_d01 = svsub_f32_m(ptrue, s_d01, row_max);  
                        s_d00 = kutacc::fast_exp(ptrue, s_d00);
                        s_d01 = kutacc::fast_exp(ptrue, s_d01);                                                      

                        s_d10 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 1, col);
                        s_d11 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 1, col+1);
                        s_d10 = svsub_f32_m(ptrue, s_d10, row_max);
                        s_d11 = svsub_f32_m(ptrue, s_d11, row_max);
                        s_d10 = kutacc::fast_exp(ptrue, s_d10);
                        s_d11 = kutacc::fast_exp(ptrue, s_d11);                            

                        s_d20 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 2, col);
                        s_d21 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 2, col+1);
                        s_d20 = svsub_f32_m(ptrue, s_d20, row_max);
                        s_d21 = svsub_f32_m(ptrue, s_d21, row_max);
                        s_d20 = kutacc::fast_exp(ptrue, s_d20);
                        s_d21 = kutacc::fast_exp(ptrue, s_d21);                            

                        s_d30 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 3, col);
                        s_d31 = svread_ver_za32_f32_m(zero_vec_f32, ptrue, 3, col+1);
                        s_d30 = svsub_f32_m(ptrue, s_d30, row_max);
                        s_d31 = svsub_f32_m(ptrue, s_d31, row_max); 
                        s_d30 = kutacc::fast_exp(ptrue, s_d30);
                        s_d31 = kutacc::fast_exp(ptrue, s_d31);                           


                        row_sum = svadd_f32_m(ptrue, row_sum, svadd_f32_m(ptrue, s_d00, s_d01));
                        row_sum = svadd_f32_m(ptrue, row_sum, svadd_f32_m(ptrue, s_d10, s_d11));
                        row_sum = svadd_f32_m(ptrue, row_sum, svadd_f32_m(ptrue, s_d20, s_d21));
                        row_sum = svadd_f32_m(ptrue, row_sum, svadd_f32_m(ptrue, s_d30, s_d31));

                        p_d0 = svcvt_bf16_f32_x(ptrue, s_d00);
                        p_d1 = svcvt_bf16_f32_x(ptrue, s_d10);
                        p_d2 = svcvt_bf16_f32_x(ptrue, s_d20);
                        p_d3 = svcvt_bf16_f32_x(ptrue, s_d30);

                        p_d0 = svcvtnt_bf16_f32_x(p_d0, ptrue, s_d01);
                        p_d1 = svcvtnt_bf16_f32_x(p_d1, ptrue, s_d11);
                        p_d2 = svcvtnt_bf16_f32_x(p_d2, ptrue, s_d21);
                        p_d3 = svcvtnt_bf16_f32_x(p_d3, ptrue, s_d31);

                        svst1_bf16(ptrue, p0, p_d0); 
                        svst1_bf16(ptrue, p1, p_d1); 
                        svst1_bf16(ptrue, p2, p_d2); 
                        svst1_bf16(ptrue, p3, p_d3); 

                        p0 += SVL_bf16;
                        p1 += SVL_bf16;
                        p2 += SVL_bf16;
                        p3 += SVL_bf16;
                    }
                    //update l
                    svfloat32_t l_d0 = svld1_f32(ptrue, &l[row]);
                    l_d0 =svadd_f32_m(ptrue, svmul_f32_m(ptrue, l_d0, row_max_prev), row_sum);
                    svst1_f32(ptrue, &l[row], l_d0);
                }
                //compute P,O
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
                        svbfloat16_t v_d0 = svld1_bf16(ptrue, v0);
                        svbfloat16_t v_d1 = svld1_bf16(ptrue, v1);

                        svbfloat16_t p_d00 = svld1_bf16(ptrue, p0);
                        svmopa_za32_bf16_m(0, ptrue, ptrue, p_d00, v_d0);
                        svmopa_za32_bf16_m(1, ptrue, ptrue, p_d00, v_d1);

                        svbfloat16_t p_d10 = svld1_bf16(ptrue, p1);
                        svmopa_za32_bf16_m(2, ptrue, ptrue, p_d10, v_d0);
                        svmopa_za32_bf16_m(3, ptrue, ptrue, p_d10, v_d1);

                        p0 += SVL_bf16;
                        p1 += SVL_bf16;
                        v0 += 2*SVL_bf16;   
                        v1 += 2*SVL_bf16;                                                          
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
        SME_OFF();  
        }
    });
}

template void cross_pack_pair_logits<bfloat16_t*>(bfloat16_t* , float* , int, int, int, int, int, int);
template void cross_pack_pair_logits<float*>(float* , float* , int, int, int, int, int, int);

}//namespace kutacc