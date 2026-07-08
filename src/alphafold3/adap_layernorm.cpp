#include "kutacc.h"
#include "common.h"
#include "math/fast_exp.h"
namespace kutacc{

void layernorm_noparams(float* x, int B, int M, int N, float eps)
{
    kutacc::parallel_for(0, B*M, 1, [&](int64_t start, int64_t end)
    {
        for(int task_idx = start; task_idx < end; task_idx ++){
            int bidx = task_idx / M;
            int i = task_idx % M;

            svbool_t p32all = svptrue_b32();
            svfloat32_t sum_v = svdup_f32(0);
            svfloat32_t sum2_v = svdup_f32(0.0f);
            float* x_i = x+bidx*M*N+i*N;
            float* x_j = x_i;
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
                x_v = svmla_x(pg, bias, x_v, rstd);
                svst1_f32(pg, x_j, x_v);
                x_j += SVL_fp32;
            }
        }
    });
}

void layernorm_pack_bf16(float* x, bfloat16_t* x_r, float* ln_weight, float* ln_bias, int M, int N, float eps)
{
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
                x_v = svmla_x(pg, bias, x_v, rstd);
                x_v = svmul_f32_m(pg, x_v, w_v);
                if(ln_bias != nullptr){
                    svfloat32_t b_v = svld1_f32(pg, b_j);
                    x_v = svadd_f32_m(pg, x_v, b_v);
                    b_j += SVL_fp32;
                }
                svst1_f32(pg, x_j, x_v);
                x_j += SVL_fp32;
                w_j += SVL_fp32;
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
                for(uint32_t trow = 0; trow < SVL_fp32; trow++)
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
                for(uint32_t tcol = 0; tcol < SVL_fp32; tcol +=2 )
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

void doublelinear_sigmoid_bf16(bfloat16_t* s_c_block, float* o1_block, float* o2_block, bfloat16_t* s_w_block, float* s_b_block, bfloat16_t* c_b2_block, int M_block, int D_block, int D1, int D)
{
    SME_ON();
    int task_num_left = M_block/SVL_fp32;
    int task_num_right = D_block/SVL_fp32;
    svbool_t p32all = svptrue_b32();
    svbool_t p16all = svptrue_b16();
    svfloat32_t zero_vec_f32 = svdup_f32(0.0f);
    svfloat32_t one_vec_f32 = svdup_f32(1.0f);
    for(int tl_idx = 0; tl_idx < task_num_left; tl_idx += 2)
    {
        bfloat16_t* sc1 = s_c_block+tl_idx*SVL_fp32*D1;
        bfloat16_t* sc2 = sc1+SVL_fp32*D1;
        for(int tr_idx = 0; tr_idx < task_num_right; tr_idx++) // use bf16 mopa
        {
            svzero_za();
            bfloat16_t* sw = s_w_block+tr_idx*SVL_fp32*D1;
            bfloat16_t* cb2 = c_b2_block+tr_idx*SVL_fp32*D1;
            float* sb = s_b_block+tr_idx*SVL_fp32;
            float* o1_1 = o1_block+tl_idx*D*SVL_fp32+tr_idx*SVL_fp32;
            float* o1_2 = o1_block+(tl_idx+1)*D*SVL_fp32+tr_idx*SVL_fp32;
            float* o2_1 = o2_block+tl_idx*D*SVL_fp32+tr_idx*SVL_fp32;
            float* o2_2 = o2_block+(tl_idx+1)*D*SVL_fp32+tr_idx*SVL_fp32;
            svfloat32_t sb_v = svld1_f32(p32all, sb);
            for(int d1_idx = 0; d1_idx < D1; d1_idx += 2)
            {
                svbfloat16_t sc1_d = svld1_bf16(p16all, sc1+d1_idx*SVL_fp32);
                svbfloat16_t sc2_d = svld1_bf16(p16all, sc2+d1_idx*SVL_fp32);
                svbfloat16_t sw_d = svld1_bf16(p16all, sw+d1_idx*SVL_fp32);
                svbfloat16_t cb2_d = svld1_bf16(p16all, cb2+d1_idx*SVL_fp32);

                svmopa_za32_bf16_m(0, p16all, p16all, sc1_d, sw_d);
                svmopa_za32_bf16_m(1, p16all, p16all, sc2_d, sw_d);
                svmopa_za32_bf16_m(2, p16all, p16all, sc1_d, cb2_d);
                svmopa_za32_bf16_m(3, p16all, p16all, sc2_d, cb2_d);
            }
            #pragma unroll(4)
            for(int trow = 0; trow < SVL_fp32; ++trow)
            {
                svst1_hor_za32(2, trow, p32all, o2_1+trow*D);
                svst1_hor_za32(3, trow, p32all, o2_2+trow*D);

                svfloat32_t scale_1 = svread_hor_za32_f32_m(zero_vec_f32, p32all, 0, static_cast<uint32_t>(trow));
                svfloat32_t scale_2 = svread_hor_za32_f32_m(zero_vec_f32, p32all, 1, static_cast<uint32_t>(trow)); 

                scale_1 = svadd_f32_m(p32all, scale_1, sb_v);
                scale_2 = svadd_f32_m(p32all, scale_2, sb_v);
                scale_1 = svneg_f32_z(p32all, scale_1);
                scale_2 = svneg_f32_z(p32all, scale_2);
                scale_1 = kutacc::fast_exp(p32all, scale_1);
                scale_2 = kutacc::fast_exp(p32all, scale_2);
                scale_1 = svadd_n_f32_z(p32all,scale_1,1.0f);
                scale_2 = svadd_n_f32_z(p32all,scale_2,1.0f);

                scale_1 = svdiv_f32_m(p32all, one_vec_f32, scale_1);
                scale_2 = svdiv_f32_m(p32all, one_vec_f32, scale_2);

                svst1_f32(p32all, o1_1+trow*D, scale_1);
                svst1_f32(p32all, o1_2+trow*D, scale_2);
            }
        }
    }
    SME_OFF(); 
}

void adpln_merge_bf16(float* x, bfloat16_t* s_c, bfloat16_t* s_w, float* s_b, bfloat16_t* c_b2, int B, int M, int D, int D1, bfloat16_t* pack_x, bool dopack)
{
    int M_block = (M%128)?64:128;
    int D_block = 128;
    int task_m = (M%M_block)?(M/M_block+1):(M/M_block);
    int task_d = (D%D_block)?(D/D_block+1):(D/D_block);
    alignas(64) float o1[M*D];
    alignas(64) float o2[M*D];
    kutacc::parallel_for(0, task_m * task_d, 1, [&](int64_t start, int64_t end){
        for(int task_idx = start; task_idx < end; task_idx ++)
        {
            int tmidx = task_idx / task_d;
            int tdidx = task_idx % task_d;
            bfloat16_t* s_c_block = s_c+tmidx*M_block*D1;
            float* o1_block = o1+tmidx*M_block*D+tdidx*D_block;
            float* o2_block = o2+tmidx*M_block*D+tdidx*D_block;
            bfloat16_t* s_w_block = s_w+tdidx*D_block*D1;
            float* s_b_block = s_b+tdidx*D_block;
            bfloat16_t* c_b2_block = c_b2+tdidx*D_block*D1;
            doublelinear_sigmoid_bf16(s_c_block, o1_block, o2_block, s_w_block, s_b_block, c_b2_block, M_block, D_block, D1, D);            
        }
    });
    if(dopack){
        int row_step = SVL_fp32;
        int col_step = 4 * SVL_fp32;
        int task_num = (M * D) / (row_step * col_step);
        int row_task_num = D / col_step;
        int max_threads = kutacc::get_thread_num();
        int avg_task = task_num / max_threads;
        int busy_thread = task_num-avg_task*max_threads;

        for(int bidx = 0; bidx < B; ++bidx)
        {
            kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
            {
                int tid = kutacc::get_thread_id();
                int task_num, start_idx;
                if(tid < busy_thread)
                {
                    task_num = avg_task+1;
                    start_idx = tid*task_num;
                }
                else
                {
                    task_num = avg_task;
                    start_idx = busy_thread*(task_num+1)+(tid-busy_thread)*task_num;
                }

                svbool_t p32all = svptrue_b32();
                svbool_t p16all = svptrue_b16();
                SME_ON();
                for(int i = 0; i < task_num; ++i)
                {
                    int row_start_idx = ((start_idx + i) / row_task_num) * row_step;
                    int col_start_idx = ((start_idx + i) % row_task_num) * col_step;
                    float* x_b_tid = x + bidx * M * D + row_start_idx * D + col_start_idx;
                    float* o1_b_tid = o1 + row_start_idx * D + col_start_idx;
                    float* o2_b_tid = o2 + row_start_idx * D + col_start_idx;
                    svzero_za();
                    for(int row_idx = 0; row_idx < SVL_fp32; row_idx++){
                        svfloat32_t o1v_0 , o1v_1, o1v_2, o1v_3, o2v_0, o2v_1, o2v_2, o2v_3, xv_0, xv_1, xv_2, xv_3;
                        o1v_0 = svld1_f32(p32all, o1_b_tid), o2v_0 = svld1_f32(p32all, o2_b_tid), xv_0 = svld1_f32(p32all, x_b_tid);
                        o1v_1 = svld1_f32(p32all, o1_b_tid + SVL_fp32), o2v_1 = svld1_f32(p32all, o2_b_tid + SVL_fp32), xv_1 = svld1_f32(p32all, x_b_tid + SVL_fp32);
                        o1v_2 = svld1_f32(p32all, o1_b_tid + 2 * SVL_fp32), o2v_2 = svld1_f32(p32all, o2_b_tid + 2 * SVL_fp32), xv_2 = svld1_f32(p32all, x_b_tid + 2 * SVL_fp32);
                        o1v_3 = svld1_f32(p32all, o1_b_tid + 3 * SVL_fp32), o2v_3 = svld1_f32(p32all, o2_b_tid + 3 * SVL_fp32), xv_3 = svld1_f32(p32all, x_b_tid + 3 * SVL_fp32);
                        xv_0 = svadd_f32_m(p32all, svmul_f32_m(p32all, xv_0, o1v_0), o2v_0);
                        xv_1 = svadd_f32_m(p32all, svmul_f32_m(p32all, xv_1, o1v_1), o2v_1);
                        xv_2 = svadd_f32_m(p32all, svmul_f32_m(p32all, xv_2, o1v_2), o2v_2);
                        xv_3 = svadd_f32_m(p32all, svmul_f32_m(p32all, xv_3, o1v_3), o2v_3);
                        svwrite_hor_za32_f32_m(0, row_idx, p32all, xv_0);
                        svwrite_hor_za32_f32_m(1, row_idx, p32all, xv_1);
                        svwrite_hor_za32_f32_m(2, row_idx, p32all, xv_2);
                        svwrite_hor_za32_f32_m(3, row_idx, p32all, xv_3);
                        x_b_tid += D;
                        o1_b_tid += D;
                        o2_b_tid += D;
                    }
                    int step = SVL_fp32 * SVL_fp32;
                    bfloat16_t * pack_x_base_0 = pack_x + bidx * M * D + row_start_idx * D + col_start_idx * SVL_fp32;
                    bfloat16_t * pack_x_base_1 = pack_x_base_0 + step;
                    bfloat16_t * pack_x_base_2 = pack_x_base_1 + step;
                    bfloat16_t * pack_x_base_3 = pack_x_base_2 + step;

                    for(uint32_t col_idx = 0; col_idx < SVL_fp32; col_idx+=2){
                        svfloat32_t data_0,data_1,data_2,data_3;
                        svbfloat16_t data_bf16_0,data_bf16_1,data_bf16_2,data_bf16_3,unzip_1,unzip_2,z1,z2;
                        data_0 = svread_ver_za32_f32_m(svfloat32_t(),p32all,0,col_idx);
                        data_1 = svread_ver_za32_f32_m(svfloat32_t(),p32all,0,col_idx+1);
                        data_2 = svread_ver_za32_f32_m(svfloat32_t(),p32all,1,col_idx);
                        data_3 = svread_ver_za32_f32_m(svfloat32_t(),p32all,1,col_idx+1);

                        data_bf16_0 = svcvt_bf16_f32_z(p32all, data_0);
                        data_bf16_1 = svcvt_bf16_f32_z(p32all, data_1);
                        data_bf16_2 = svcvt_bf16_f32_z(p32all, data_2);
                        data_bf16_3 = svcvt_bf16_f32_z(p32all, data_3);

                        unzip_1 = svuzp1_bf16(data_bf16_0, data_bf16_2);
                        unzip_2 = svuzp1_bf16(data_bf16_1, data_bf16_3);
                        z1 = svzip1_bf16(unzip_1,unzip_2);
                        z2 = svzip2_bf16(unzip_1,unzip_2);
                        svst1_bf16(p16all, pack_x_base_0, z1);
                        svst1_bf16(p16all, pack_x_base_1, z2);

                        data_0 = svread_ver_za32_f32_m(svfloat32_t(),p32all,2,col_idx);
                        data_1 = svread_ver_za32_f32_m(svfloat32_t(),p32all,2,col_idx+1);
                        data_2 = svread_ver_za32_f32_m(svfloat32_t(),p32all,3,col_idx);
                        data_3 = svread_ver_za32_f32_m(svfloat32_t(),p32all,3,col_idx+1);

                        data_bf16_0 = svcvt_bf16_f32_z(p32all, data_0);
                        data_bf16_1 = svcvt_bf16_f32_z(p32all, data_1);
                        data_bf16_2 = svcvt_bf16_f32_z(p32all, data_2);
                        data_bf16_3 = svcvt_bf16_f32_z(p32all, data_3);

                        unzip_1 = svuzp1_bf16(data_bf16_0, data_bf16_2);
                        unzip_2 = svuzp1_bf16(data_bf16_1, data_bf16_3);
                        z1 = svzip1_bf16(unzip_1,unzip_2);
                        z2 = svzip2_bf16(unzip_1,unzip_2);
                        svst1_bf16(p16all, pack_x_base_2, z1);
                        svst1_bf16(p16all, pack_x_base_3, z2);

                        pack_x_base_0 += 2*SVL_fp32;
                        pack_x_base_1 += 2*SVL_fp32;
                        pack_x_base_2 += 2*SVL_fp32;
                        pack_x_base_3 += 2*SVL_fp32;
                    }
                }
                SME_OFF();
            });
        }
    }else{
        int task_num = M;
        int max_threads = kutacc::get_thread_num();
        int avg_task = task_num / max_threads;
        int busy_thread = task_num-avg_task*max_threads;  
        for(int bidx = 0; bidx < B; ++bidx)
        {
            kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
            {
                int tid = kutacc::get_thread_id();
                int task_num, start_idx;
                if(tid < busy_thread)
                {
                    task_num = avg_task+1;
                    start_idx = tid*task_num;
                }
                else
                {
                    task_num = avg_task;
                    start_idx = busy_thread*(task_num+1)+(tid-busy_thread)*task_num;
                }

                float* x_b_tid = x+bidx*M*D+start_idx*D;
                float* o1_b_tid = o1+start_idx*D;
                float* o2_b_tid = o2+start_idx*D;
                svbool_t p32all = svptrue_b32();
                for(int i = 0; i < task_num; ++i)
                {
                    #pragma unroll(8)
                    for(int d = 0; d < D; d += SVL_fp32)
                    {
                        svfloat32_t o1v = svld1_f32(p32all, o1_b_tid);
                        svfloat32_t o2v = svld1_f32(p32all, o2_b_tid);
                        svfloat32_t xv = svld1_f32(p32all, x_b_tid);
                        xv = svadd_f32_m(p32all, svmul_f32_m(p32all, xv, o1v), o2v);
                        svst1_f32(p32all, x_b_tid, xv);
                        x_b_tid += SVL_fp32;
                        o1_b_tid += SVL_fp32;
                        o2_b_tid += SVL_fp32;
                    }
                }
            });
        }

    }

}


} //namespace kutacc