#include "linear.h"

namespace kutacc{
void bgemm_4vl_1vl_bf16_transpose_trans_impl(bfloat16_t *mat1, bfloat16_t *mat2, bfloat16_t * results,int M, int N, int K,\
    int row_idx, int col_idx, int batch, int seq_len, int nheads, int head_size, bool trans)
{
    svbool_t pt16 = svptrue_b16();
    svbool_t pt16_w = svwhilelt_b16(0,16);
    svbool_t pt32 = svptrue_b32();
    const int row_step = 4 * SVL_fp32;
    const int row_single_step = SVL_fp32;
    const int col_step = SVL_fp32;
    SME_ON();
    int B_step, H_step, N_step;
    if(trans)
    {
        // BNHD -> NHBD
        B_step = head_size;
        H_step = batch * head_size;
        N_step = nheads * batch * head_size;
    }
    else
    {
        // BNHD -> BHND
        B_step = seq_len * nheads * head_size;
        H_step = seq_len * head_size;
        N_step = head_size;
    }
    for(int row_index = 0; row_index < M; row_index += row_step)
    {
        for(int col_index = 0; col_index < N; col_index += col_step)
        {
            bfloat16_t* mat1_base0 = mat1 + row_index * K;
            bfloat16_t* mat1_base1 = mat1 + (row_index + row_single_step ) * K;
            bfloat16_t* mat1_base2 = mat1 + (row_index + 2 * row_single_step ) * K;
            bfloat16_t* mat1_base3 = mat1 + (row_index + 3 * row_single_step ) * K;
            bfloat16_t* mat2_base0 = mat2 + col_index * K ;
            svzero_za();
            for(int k_idx = 0; k_idx < K; k_idx += 2)
            {
                svbfloat16_t rowB0 = svld1_bf16(pt16,mat2_base0);

                svbfloat16_t colA0 = svld1_bf16(pt16,mat1_base0); 
                svmopa_za32_bf16_m(0,pt16,pt16,colA0,rowB0);
                svbfloat16_t colA1 = svld1_bf16(pt16,mat1_base1); 
                svmopa_za32_bf16_m(1,pt16,pt16,colA1,rowB0);
                svbfloat16_t colA2 = svld1_bf16(pt16,mat1_base2); 
                svmopa_za32_bf16_m(2,pt16,pt16,colA2,rowB0);
                svbfloat16_t colA3 = svld1_bf16(pt16,mat1_base3); 
                svmopa_za32_bf16_m(3,pt16,pt16,colA3,rowB0);
                mat2_base0 += SVL_bf16;
                mat1_base0 += SVL_bf16;
                mat1_base1 += SVL_bf16;
                mat1_base2 += SVL_bf16;
                mat1_base3 += SVL_bf16;
            }
            int col_index_global = col_idx + col_index;
            int H_idx = col_index_global / head_size;
            int D_idx = col_index_global % head_size;
            int col_offset = H_idx * H_step + D_idx;
            #pragma unroll(8)
            for (int inner_row_idx=0; inner_row_idx < SVL_fp32; inner_row_idx++) 
            {
                svfloat32_t data_0 = svread_hor_za32_f32_m(svfloat32_t(),pt32,0,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_0 = svcvt_bf16_f32_z(pt32, data_0);
                data_bf16_0 = svuzp1_bf16(data_bf16_0, svdup_bf16(0.0));
                int row_index_global = row_idx + row_index + inner_row_idx;
                int B_idx = row_index_global / seq_len;
                int N_idx = row_index_global % seq_len;
                svst1_bf16(pt16_w, results + B_idx * B_step + N_idx * N_step + col_offset, data_bf16_0);

                svfloat32_t data_1 = svread_hor_za32_f32_m(svfloat32_t(),pt32,1,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_1 = svcvt_bf16_f32_z(pt32, data_1);
                data_bf16_1 = svuzp1_bf16(data_bf16_1, svdup_bf16(0.0));
                int row_index_global_1 = row_index_global + SVL_fp32;
                int B_idx_1 = row_index_global_1 / seq_len;
                int N_idx_1 = row_index_global_1 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_1 * B_step + N_idx_1 * N_step + col_offset, data_bf16_1);

                svfloat32_t data_2 = svread_hor_za32_f32_m(svfloat32_t(),pt32,2,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_2 = svcvt_bf16_f32_z(pt32, data_2);
                data_bf16_2 = svuzp1_bf16(data_bf16_2, svdup_bf16(0.0));
                int row_index_global_2 = row_index_global_1 + SVL_fp32;
                int B_idx_2 = row_index_global_2 / seq_len;
                int N_idx_2 = row_index_global_2 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_2 * B_step + N_idx_2 * N_step + col_offset, data_bf16_2);

                svfloat32_t data_3 = svread_hor_za32_f32_m(svfloat32_t(),pt32,3,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_3 = svcvt_bf16_f32_z(pt32, data_3);
                data_bf16_3 = svuzp1_bf16(data_bf16_3, svdup_bf16(0.0));
                int row_index_global_3 = row_index_global_2 + SVL_fp32;
                int B_idx_3 = row_index_global_3 / seq_len;
                int N_idx_3 = row_index_global_3 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_3 * B_step + N_idx_3 * N_step + col_offset, data_bf16_3);

            }
        }
    }
    SME_OFF();
}

void bgemm_4vl_1vl_bf16_trans_impl(bfloat16_t *mat1, bfloat16_t *mat2, bfloat16_t * results,int M, int N, int K, int c_stride, \
    int row_idx, int col_idx, int batch, int seq_len, bool trans)
{
    svbool_t pt16 = svptrue_b16();
    svbool_t pt16_w = svwhilelt_b16(0,16);
    svbool_t pt32 = svptrue_b32();
    const int row_step = 4 * SVL_fp32;
    const int row_single_step = SVL_fp32;
    const int col_step = SVL_fp32;
    SME_ON();
    int B_step, N_step;
    if(trans)
    {
        // BNHD -> NBHD
        B_step = c_stride;
        N_step = batch * c_stride;
    }
    else
    {
        // BNHD
        B_step = seq_len * c_stride;
        N_step = c_stride;
    }
    for(int row_index = 0; row_index < M; row_index += row_step)
    {
        for(int col_index = 0; col_index < N; col_index += col_step)
        {
            bfloat16_t* mat1_base0 = mat1 + row_index * K;
            bfloat16_t* mat1_base1 = mat1 + (row_index + row_single_step ) * K;
            bfloat16_t* mat1_base2 = mat1 + (row_index + 2 * row_single_step ) * K;
            bfloat16_t* mat1_base3 = mat1 + (row_index + 3 * row_single_step ) * K;
            bfloat16_t* mat2_base0 = mat2 + col_index * K ;

            svzero_za();
            for(int k_idx = 0; k_idx < SVL_fp32 * K; k_idx += SVL_bf16)
            {
                svbfloat16_t rowB0 = svld1_bf16(pt16,mat2_base0);

                svbfloat16_t colA0 = svld1_bf16(pt16,mat1_base0); 
                svmopa_za32_bf16_m(0,pt16,pt16,colA0,rowB0);
                svbfloat16_t colA1 = svld1_bf16(pt16,mat1_base1); 
                svmopa_za32_bf16_m(1,pt16,pt16,colA1,rowB0);
                svbfloat16_t colA2 = svld1_bf16(pt16,mat1_base2); 
                svmopa_za32_bf16_m(2,pt16,pt16,colA2,rowB0);
                svbfloat16_t colA3 = svld1_bf16(pt16,mat1_base3); 
                svmopa_za32_bf16_m(3,pt16,pt16,colA3,rowB0);
                mat2_base0 += SVL_bf16;
                mat1_base0 += SVL_bf16;
                mat1_base1 += SVL_bf16;
                mat1_base2 += SVL_bf16;
                mat1_base3 += SVL_bf16;
            }
            int col_offset = col_idx + col_index;
            #pragma unroll(8)
            for (int inner_row_idx=0; inner_row_idx < SVL_fp32; inner_row_idx++) 
            {
                svfloat32_t data_0 = svread_hor_za32_f32_m(svfloat32_t(),pt32,0,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_0 = svcvt_bf16_f32_z(pt32, data_0);
                data_bf16_0 = svuzp1_bf16(data_bf16_0, svdup_bf16(0.0));
                int row_index_global = row_idx + row_index + inner_row_idx;
                int B_idx = row_index_global / seq_len;
                int N_idx = row_index_global % seq_len;
                svst1_bf16(pt16_w, results + B_idx * B_step + N_idx * N_step + col_offset, data_bf16_0);

                svfloat32_t data_1 = svread_hor_za32_f32_m(svfloat32_t(),pt32,1,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_1 = svcvt_bf16_f32_z(pt32, data_1);
                data_bf16_1 = svuzp1_bf16(data_bf16_1, svdup_bf16(0.0));
                int row_index_global_1 = row_index_global + SVL_fp32;
                int B_idx_1 = row_index_global_1 / seq_len;
                int N_idx_1 = row_index_global_1 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_1 * B_step + N_idx_1 * N_step + col_offset, data_bf16_1);

                svfloat32_t data_2 = svread_hor_za32_f32_m(svfloat32_t(),pt32,2,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_2 = svcvt_bf16_f32_z(pt32, data_2);
                data_bf16_2 = svuzp1_bf16(data_bf16_2, svdup_bf16(0.0));
                int row_index_global_2 = row_index_global_1 + SVL_fp32;
                int B_idx_2 = row_index_global_2 / seq_len;
                int N_idx_2 = row_index_global_2 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_2 * B_step + N_idx_2 * N_step + col_offset, data_bf16_2);

                svfloat32_t data_3 = svread_hor_za32_f32_m(svfloat32_t(),pt32,3,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_3 = svcvt_bf16_f32_z(pt32, data_3);
                data_bf16_3 = svuzp1_bf16(data_bf16_3, svdup_bf16(0.0));
                int row_index_global_3 = row_index_global_2 + SVL_fp32;
                int B_idx_3 = row_index_global_3 / seq_len;
                int N_idx_3 = row_index_global_3 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_3 * B_step + N_idx_3 * N_step + col_offset, data_bf16_3);

            }
        }
    }
    SME_OFF();
}

void bgemm_4vl_1vl_bf16_transpose_impl(bfloat16_t *mat1, bfloat16_t *mat2, bfloat16_t * results,int M, int N, int K,\
    int row_idx, int col_idx, int seq_len, int nheads, int head_size)
{
    svbool_t pt16 = svptrue_b16();
    svbool_t pt16_w = svwhilelt_b16(0,16);
    svbool_t pt32 = svptrue_b32();
    const int row_step = 4 * SVL_fp32;
    const int row_single_step = SVL_fp32;
    const int col_step = SVL_fp32;
    SME_ON();
    const int B_step = seq_len * nheads * head_size;
    const int H_step = seq_len * head_size;
    const int N_step = head_size;
    for(int row_index = 0; row_index < M; row_index += row_step)
    {
        for(int col_index = 0; col_index < N; col_index += col_step)
        {
            bfloat16_t* mat1_base0 = mat1 + row_index * K;
            bfloat16_t* mat1_base1 = mat1 + (row_index + row_single_step ) * K;
            bfloat16_t* mat1_base2 = mat1 + (row_index + 2 * row_single_step ) * K;
            bfloat16_t* mat1_base3 = mat1 + (row_index + 3 * row_single_step ) * K;
            bfloat16_t* mat2_base0 = mat2 + col_index * K ;
            svzero_za();
            for(int k_idx = 0; k_idx < K; k_idx += 2)
            {
                svbfloat16_t rowB0 = svld1_bf16(pt16,mat2_base0);

                svbfloat16_t colA0 = svld1_bf16(pt16,mat1_base0); 
                svmopa_za32_bf16_m(0,pt16,pt16,colA0,rowB0);
                svbfloat16_t colA1 = svld1_bf16(pt16,mat1_base1); 
                svmopa_za32_bf16_m(1,pt16,pt16,colA1,rowB0);
                svbfloat16_t colA2 = svld1_bf16(pt16,mat1_base2); 
                svmopa_za32_bf16_m(2,pt16,pt16,colA2,rowB0);
                svbfloat16_t colA3 = svld1_bf16(pt16,mat1_base3); 
                svmopa_za32_bf16_m(3,pt16,pt16,colA3,rowB0);
                mat2_base0 += SVL_bf16;
                mat1_base0 += SVL_bf16;
                mat1_base1 += SVL_bf16;
                mat1_base2 += SVL_bf16;
                mat1_base3 += SVL_bf16;
            }
            int col_index_global = col_idx + col_index;
            int H_idx = col_index_global / head_size;
            int D_idx = col_index_global % head_size;
            int col_offset = H_idx * H_step + D_idx;
            #pragma unroll(8)
            for (int inner_row_idx=0; inner_row_idx < SVL_fp32; inner_row_idx++) 
            {
                svfloat32_t data_0 = svread_hor_za32_f32_m(svfloat32_t(),pt32,0,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_0 = svcvt_bf16_f32_z(pt32, data_0);
                data_bf16_0 = svuzp1_bf16(data_bf16_0, svdup_bf16(0.0));
                int row_index_global = row_idx + row_index + inner_row_idx;
                int B_idx = row_index_global / seq_len;
                int N_idx = row_index_global % seq_len;
                svst1_bf16(pt16_w, results + B_idx * B_step + N_idx * N_step + col_offset, data_bf16_0);

                svfloat32_t data_1 = svread_hor_za32_f32_m(svfloat32_t(),pt32,1,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_1 = svcvt_bf16_f32_z(pt32, data_1);
                data_bf16_1 = svuzp1_bf16(data_bf16_1, svdup_bf16(0.0));
                int row_index_global_1 = row_index_global + SVL_fp32;
                int B_idx_1 = row_index_global_1 / seq_len;
                int N_idx_1 = row_index_global_1 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_1 * B_step + N_idx_1 * N_step + col_offset, data_bf16_1);

                svfloat32_t data_2 = svread_hor_za32_f32_m(svfloat32_t(),pt32,2,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_2 = svcvt_bf16_f32_z(pt32, data_2);
                data_bf16_2 = svuzp1_bf16(data_bf16_2, svdup_bf16(0.0));
                int row_index_global_2 = row_index_global_1 + SVL_fp32;
                int B_idx_2 = row_index_global_2 / seq_len;
                int N_idx_2 = row_index_global_2 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_2 * B_step + N_idx_2 * N_step + col_offset, data_bf16_2);

                svfloat32_t data_3 = svread_hor_za32_f32_m(svfloat32_t(),pt32,3,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_3 = svcvt_bf16_f32_z(pt32, data_3);
                data_bf16_3 = svuzp1_bf16(data_bf16_3, svdup_bf16(0.0));
                int row_index_global_3 = row_index_global_2 + SVL_fp32;
                int B_idx_3 = row_index_global_3 / seq_len;
                int N_idx_3 = row_index_global_3 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_3 * B_step + N_idx_3 * N_step + col_offset, data_bf16_3);

            }
        }
    }
    SME_OFF();
}

void bgemm_bias_4vl_1vl_bf16_transpose_impl(bfloat16_t *mat1, bfloat16_t *mat2, float32_t *bias, bfloat16_t * results,int M, int N, int K, \
    int row_idx, int col_idx, int seq_len, int nheads, int head_size)
{
    svbool_t pt16 = svptrue_b16();
    svbool_t pt16_w = svwhilelt_b16(0,16);
    svbool_t pt32 = svptrue_b32();
    const int row_step = 4 * SVL_fp32;
    const int row_single_step = SVL_fp32;
    const int col_step = SVL_fp32;
    SME_ON();
    const int B_step = seq_len * nheads * head_size;
    const int H_step = seq_len * head_size;
    const int N_step = head_size;
    for(int row_index = 0; row_index < M; row_index += row_step)
    {
        for(int col_index = 0; col_index < N; col_index += col_step)
        {
            bfloat16_t* mat1_base0 = mat1 + row_index * K;
            bfloat16_t* mat1_base1 = mat1 + (row_index + row_single_step ) * K;
            bfloat16_t* mat1_base2 = mat1 + (row_index + 2 * row_single_step ) * K;
            bfloat16_t* mat1_base3 = mat1 + (row_index + 3 * row_single_step ) * K;
            bfloat16_t* mat2_base0 = mat2 + col_index * K ;
            svfloat32_t bias_v = svld1_f32(pt32, bias + col_index);
            svzero_za();
            for(int k_idx = 0; k_idx < K; k_idx += 2)
            {
                svbfloat16_t rowB0 = svld1_bf16(pt16,mat2_base0);

                svbfloat16_t colA0 = svld1_bf16(pt16,mat1_base0); 
                svmopa_za32_bf16_m(0,pt16,pt16,colA0,rowB0);
                svbfloat16_t colA1 = svld1_bf16(pt16,mat1_base1); 
                svmopa_za32_bf16_m(1,pt16,pt16,colA1,rowB0);
                svbfloat16_t colA2 = svld1_bf16(pt16,mat1_base2); 
                svmopa_za32_bf16_m(2,pt16,pt16,colA2,rowB0);
                svbfloat16_t colA3 = svld1_bf16(pt16,mat1_base3); 
                svmopa_za32_bf16_m(3,pt16,pt16,colA3,rowB0);
                mat2_base0 += SVL_bf16;
                mat1_base0 += SVL_bf16;
                mat1_base1 += SVL_bf16;
                mat1_base2 += SVL_bf16;
                mat1_base3 += SVL_bf16;
            }
            int col_index_global = col_idx + col_index;
            int H_idx = col_index_global / head_size;
            int D_idx = col_index_global % head_size;
            int col_offset = H_idx * H_step + D_idx;
            
            #pragma unroll(8)
            for (int inner_row_idx=0; inner_row_idx < SVL_fp32; inner_row_idx++) 
            {
                svfloat32_t data_0 = svread_hor_za32_f32_m(svfloat32_t(),pt32,0,static_cast<uint32_t>(inner_row_idx));
                data_0 = svadd_f32_x(pt32,data_0,bias_v);
                svbfloat16_t data_bf16_0 = svcvt_bf16_f32_z(pt32, data_0);
                data_bf16_0 = svuzp1_bf16(data_bf16_0, svdup_bf16(0.0));
                int row_index_global = row_idx + row_index + inner_row_idx;
                int B_idx = row_index_global / seq_len;
                int N_idx = row_index_global % seq_len;
                svst1_bf16(pt16_w, results + B_idx * B_step + N_idx * N_step + col_offset, data_bf16_0);

                svfloat32_t data_1 = svread_hor_za32_f32_m(svfloat32_t(),pt32,1,static_cast<uint32_t>(inner_row_idx));
                data_1 = svadd_f32_x(pt32,data_1,bias_v);
                svbfloat16_t data_bf16_1 = svcvt_bf16_f32_z(pt32, data_1);
                data_bf16_1 = svuzp1_bf16(data_bf16_1, svdup_bf16(0.0));
                int row_index_global_1 = row_index_global + SVL_fp32;
                int B_idx_1 = row_index_global_1 / seq_len;
                int N_idx_1 = row_index_global_1 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_1 * B_step + N_idx_1 * N_step + col_offset, data_bf16_1);

                svfloat32_t data_2 = svread_hor_za32_f32_m(svfloat32_t(),pt32,2,static_cast<uint32_t>(inner_row_idx));
                data_2 = svadd_f32_x(pt32,data_2,bias_v);
                svbfloat16_t data_bf16_2 = svcvt_bf16_f32_z(pt32, data_2);
                data_bf16_2 = svuzp1_bf16(data_bf16_2, svdup_bf16(0.0));
                int row_index_global_2 = row_index_global_1 + SVL_fp32;
                int B_idx_2 = row_index_global_2 / seq_len;
                int N_idx_2 = row_index_global_2 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_2 * B_step + N_idx_2 * N_step + col_offset, data_bf16_2);

                svfloat32_t data_3 = svread_hor_za32_f32_m(svfloat32_t(),pt32,3,static_cast<uint32_t>(inner_row_idx));
                data_3 = svadd_f32_x(pt32,data_3,bias_v);
                svbfloat16_t data_bf16_3 = svcvt_bf16_f32_z(pt32, data_3);
                data_bf16_3 = svuzp1_bf16(data_bf16_3, svdup_bf16(0.0));
                int row_index_global_3 = row_index_global_2 + SVL_fp32;
                int B_idx_3 = row_index_global_3 / seq_len;
                int N_idx_3 = row_index_global_3 % seq_len;
                svst1_bf16(pt16_w, results + B_idx_3 * B_step + N_idx_3 * N_step + col_offset, data_bf16_3);

            }
        }
    }
    SME_OFF();
}

void bgemm_bias_4vl_1vl_bf16_transpose_24_impl(bfloat16_t *mat1, bfloat16_t *mat2, float32_t *bias, bfloat16_t * results,int M, int N, int K,\
    int row_idx, int col_idx, int seq_len, int nheads, int head_size)
{ // version for head_size = 24
    svbool_t pt16 = svptrue_b16();
    svbool_t pt16_w = svwhilelt_b16(0,16);
    svbool_t pt16_8 = svwhilelt_b16(0,8);
    svbool_t pt16_8_16 = svcmpgt_n_u16(pt16_w,svindex_u16(0, 1), 7); // 8-16
    svbool_t pt32 = svptrue_b32();
    const int row_step = 4 * SVL_fp32;
    const int row_single_step = SVL_fp32;
    const int col_step = SVL_fp32;
    SME_ON();
    const int B_step = seq_len * nheads * head_size;
    const int H_step = seq_len * head_size;
    const int N_step = head_size;
    for(int row_index = 0; row_index < M; row_index += row_step)
    {
        for(int col_index = 0; col_index < N; col_index += col_step)
        {
            bfloat16_t* mat1_base0 = mat1 + row_index * K;
            bfloat16_t* mat1_base1 = mat1 + (row_index + row_single_step ) * K;
            bfloat16_t* mat1_base2 = mat1 + (row_index + 2 * row_single_step ) * K;
            bfloat16_t* mat1_base3 = mat1 + (row_index + 3 * row_single_step ) * K;
            bfloat16_t* mat2_base0 = mat2 + col_index * K ;
            svfloat32_t bias_v = svld1_f32(pt32, bias + col_index);
            svzero_za();
            for(int k_idx = 0; k_idx < K; k_idx += 2)
            {
                svbfloat16_t rowB0 = svld1_bf16(pt16,mat2_base0);

                svbfloat16_t colA0 = svld1_bf16(pt16,mat1_base0); 
                svmopa_za32_bf16_m(0,pt16,pt16,colA0,rowB0);
                svbfloat16_t colA1 = svld1_bf16(pt16,mat1_base1); 
                svmopa_za32_bf16_m(1,pt16,pt16,colA1,rowB0);
                svbfloat16_t colA2 = svld1_bf16(pt16,mat1_base2); 
                svmopa_za32_bf16_m(2,pt16,pt16,colA2,rowB0);
                svbfloat16_t colA3 = svld1_bf16(pt16,mat1_base3); 
                svmopa_za32_bf16_m(3,pt16,pt16,colA3,rowB0);
                mat2_base0 += SVL_bf16;
                mat1_base0 += SVL_bf16;
                mat1_base1 += SVL_bf16;
                mat1_base2 += SVL_bf16;
                mat1_base3 += SVL_bf16;
            }

            int col_index_global = col_idx + col_index;
            int H_idx = col_index_global / head_size;
            int D_idx = col_index_global % head_size; // D_idx = 0 / 8 / 16
            if (D_idx == 0 || D_idx == 8){
                int col_offset = H_idx * H_step + D_idx;
                #pragma unroll(8)
                for (int inner_row_idx=0; inner_row_idx < SVL_fp32; inner_row_idx++) 
                {
                    svfloat32_t data_0 = svread_hor_za32_f32_m(svfloat32_t(),pt32,0,static_cast<uint32_t>(inner_row_idx));
                    data_0 = svadd_f32_x(pt32,data_0,bias_v);
                    svbfloat16_t data_bf16_0 = svcvt_bf16_f32_z(pt32, data_0);
                    data_bf16_0 = svuzp1_bf16(data_bf16_0, svdup_bf16(0.0));
                    int row_index_global = row_idx + row_index + inner_row_idx;
                    int B_idx = row_index_global / seq_len;
                    int N_idx = row_index_global % seq_len;
                    svst1_bf16(pt16_w, results + B_idx * B_step + N_idx * N_step + col_offset, data_bf16_0);

                    svfloat32_t data_1 = svread_hor_za32_f32_m(svfloat32_t(),pt32,1,static_cast<uint32_t>(inner_row_idx));
                    data_1 = svadd_f32_x(pt32,data_1,bias_v);
                    svbfloat16_t data_bf16_1 = svcvt_bf16_f32_z(pt32, data_1);
                    data_bf16_1 = svuzp1_bf16(data_bf16_1, svdup_bf16(0.0));
                    int row_index_global_1 = row_index_global + SVL_fp32;
                    int B_idx_1 = row_index_global_1 / seq_len;
                    int N_idx_1 = row_index_global_1 % seq_len;
                    svst1_bf16(pt16_w, results + B_idx_1 * B_step + N_idx_1 * N_step + col_offset, data_bf16_1);

                    svfloat32_t data_2 = svread_hor_za32_f32_m(svfloat32_t(),pt32,2,static_cast<uint32_t>(inner_row_idx));
                    data_2 = svadd_f32_x(pt32,data_2,bias_v);
                    svbfloat16_t data_bf16_2 = svcvt_bf16_f32_z(pt32, data_2);
                    data_bf16_2 = svuzp1_bf16(data_bf16_2, svdup_bf16(0.0));
                    int row_index_global_2 = row_index_global_1 + SVL_fp32;
                    int B_idx_2 = row_index_global_2 / seq_len;
                    int N_idx_2 = row_index_global_2 % seq_len;
                    svst1_bf16(pt16_w, results + B_idx_2 * B_step + N_idx_2 * N_step + col_offset, data_bf16_2);

                    svfloat32_t data_3 = svread_hor_za32_f32_m(svfloat32_t(),pt32,3,static_cast<uint32_t>(inner_row_idx));
                    data_3 = svadd_f32_x(pt32,data_3,bias_v);
                    svbfloat16_t data_bf16_3 = svcvt_bf16_f32_z(pt32, data_3);
                    data_bf16_3 = svuzp1_bf16(data_bf16_3, svdup_bf16(0.0));
                    int row_index_global_3 = row_index_global_2 + SVL_fp32;
                    int B_idx_3 = row_index_global_3 / seq_len;
                    int N_idx_3 = row_index_global_3 % seq_len;
                    svst1_bf16(pt16_w, results + B_idx_3 * B_step + N_idx_3 * N_step + col_offset, data_bf16_3);

                }
            }
            else{
                int col_offset_0 = H_idx * H_step + D_idx; // D_idx = 16
                int col_offset_1 = (H_idx + 1) * H_step - 8 ;
                #pragma unroll(8)
                for (int inner_row_idx=0; inner_row_idx < SVL_fp32; inner_row_idx++) 
                {
                    svfloat32_t data_0 = svread_hor_za32_f32_m(svfloat32_t(),pt32,0,static_cast<uint32_t>(inner_row_idx));
                    data_0 = svadd_f32_x(pt32,data_0,bias_v);
                    svbfloat16_t data_bf16_0 = svcvt_bf16_f32_z(pt32, data_0);
                    data_bf16_0 = svuzp1_bf16(data_bf16_0, svdup_bf16(0.0));
                    int row_index_global = row_idx + row_index + inner_row_idx;
                    int B_idx = row_index_global / seq_len;
                    int N_idx = row_index_global % seq_len;
                    svst1_bf16(pt16_8, results + B_idx * B_step + N_idx * N_step + col_offset_0, data_bf16_0);
                    svst1_bf16(pt16_8_16, results + B_idx * B_step + N_idx * N_step + col_offset_1, data_bf16_0);

                    svfloat32_t data_1 = svread_hor_za32_f32_m(svfloat32_t(),pt32,1,static_cast<uint32_t>(inner_row_idx));
                    data_1 = svadd_f32_x(pt32,data_1,bias_v);
                    svbfloat16_t data_bf16_1 = svcvt_bf16_f32_z(pt32, data_1);
                    data_bf16_1 = svuzp1_bf16(data_bf16_1, svdup_bf16(0.0));
                    int row_index_global_1 = row_index_global + SVL_fp32;
                    int B_idx_1 = row_index_global_1 / seq_len;
                    int N_idx_1 = row_index_global_1 % seq_len;
                    svst1_bf16(pt16_8, results + B_idx_1 * B_step + N_idx_1 * N_step + col_offset_0, data_bf16_1);
                    svst1_bf16(pt16_8_16, results + B_idx_1 * B_step + N_idx_1 * N_step + col_offset_1, data_bf16_1);

                    svfloat32_t data_2 = svread_hor_za32_f32_m(svfloat32_t(),pt32,2,static_cast<uint32_t>(inner_row_idx));
                    data_2 = svadd_f32_x(pt32,data_2,bias_v);
                    svbfloat16_t data_bf16_2 = svcvt_bf16_f32_z(pt32, data_2);
                    data_bf16_2 = svuzp1_bf16(data_bf16_2, svdup_bf16(0.0));
                    int row_index_global_2 = row_index_global_1 + SVL_fp32;
                    int B_idx_2 = row_index_global_2 / seq_len;
                    int N_idx_2 = row_index_global_2 % seq_len;
                    svst1_bf16(pt16_8, results + B_idx_2 * B_step + N_idx_2 * N_step + col_offset_0, data_bf16_2);
                    svst1_bf16(pt16_8_16, results + B_idx_2 * B_step + N_idx_2 * N_step + col_offset_1, data_bf16_2);

                    svfloat32_t data_3 = svread_hor_za32_f32_m(svfloat32_t(),pt32,3,static_cast<uint32_t>(inner_row_idx));
                    data_3 = svadd_f32_x(pt32,data_3,bias_v);
                    svbfloat16_t data_bf16_3 = svcvt_bf16_f32_z(pt32, data_3);
                    data_bf16_3 = svuzp1_bf16(data_bf16_3, svdup_bf16(0.0));
                    int row_index_global_3 = row_index_global_2 + SVL_fp32;
                    int B_idx_3 = row_index_global_3 / seq_len;
                    int N_idx_3 = row_index_global_3 % seq_len;
                    svst1_bf16(pt16_8, results + B_idx_3 * B_step + N_idx_3 * N_step + col_offset_0, data_bf16_3);
                    svst1_bf16(pt16_8_16, results + B_idx_3 * B_step + N_idx_3 * N_step + col_offset_1, data_bf16_3);
                }
            }
        }
    }
    SME_OFF();
}
void bgemm_4vl_1vl_bf16_transpose_24_impl(bfloat16_t *mat1, bfloat16_t *mat2, bfloat16_t * results,int M, int N, int K,\
    int row_idx, int col_idx, int seq_len, int nheads, int head_size)
{ // version for head_size = 24
    svbool_t pt16 = svptrue_b16();
    svbool_t pt16_w = svwhilelt_b16(0,16);
    svbool_t pt16_8 = svwhilelt_b16(0,8);
    svbool_t pt16_8_16 = svcmpgt_n_u16(pt16_w,svindex_u16(0, 1), 7); // 8-16
    svbool_t pt32 = svptrue_b32();
    const int row_step = 4 * SVL_fp32;
    const int row_single_step = SVL_fp32;
    const int col_step = SVL_fp32;
    SME_ON();
    const int B_step = seq_len * nheads * head_size;
    const int H_step = seq_len * head_size;
    const int N_step = head_size;
    for(int row_index = 0; row_index < M; row_index += row_step)
    {
        for(int col_index = 0; col_index < N; col_index += col_step)
        {
            bfloat16_t* mat1_base0 = mat1 + row_index * K;
            bfloat16_t* mat1_base1 = mat1 + (row_index + row_single_step ) * K;
            bfloat16_t* mat1_base2 = mat1 + (row_index + 2 * row_single_step ) * K;
            bfloat16_t* mat1_base3 = mat1 + (row_index + 3 * row_single_step ) * K;
            bfloat16_t* mat2_base0 = mat2 + col_index * K ;
            svzero_za();
            for(int k_idx = 0; k_idx < K; k_idx += 2)
            {
                svbfloat16_t rowB0 = svld1_bf16(pt16,mat2_base0);

                svbfloat16_t colA0 = svld1_bf16(pt16,mat1_base0); 
                svmopa_za32_bf16_m(0,pt16,pt16,colA0,rowB0);
                svbfloat16_t colA1 = svld1_bf16(pt16,mat1_base1); 
                svmopa_za32_bf16_m(1,pt16,pt16,colA1,rowB0);
                svbfloat16_t colA2 = svld1_bf16(pt16,mat1_base2); 
                svmopa_za32_bf16_m(2,pt16,pt16,colA2,rowB0);
                svbfloat16_t colA3 = svld1_bf16(pt16,mat1_base3); 
                svmopa_za32_bf16_m(3,pt16,pt16,colA3,rowB0);
                mat2_base0 += SVL_bf16;
                mat1_base0 += SVL_bf16;
                mat1_base1 += SVL_bf16;
                mat1_base2 += SVL_bf16;
                mat1_base3 += SVL_bf16;
            }

            int col_index_global = col_idx + col_index;
            int H_idx = col_index_global / head_size;
            int D_idx = col_index_global % head_size; // D_idx = 0 / 8 / 16
            if (D_idx == 0 || D_idx == 8){
                int col_offset = H_idx * H_step + D_idx;
                #pragma unroll(8)
                for (int inner_row_idx=0; inner_row_idx < SVL_fp32; inner_row_idx++) 
                {
                    svfloat32_t data_0 = svread_hor_za32_f32_m(svfloat32_t(),pt32,0,static_cast<uint32_t>(inner_row_idx));
                    svbfloat16_t data_bf16_0 = svcvt_bf16_f32_z(pt32, data_0);
                    data_bf16_0 = svuzp1_bf16(data_bf16_0, svdup_bf16(0.0));
                    int row_index_global = row_idx + row_index + inner_row_idx;
                    int B_idx = row_index_global / seq_len;
                    int N_idx = row_index_global % seq_len;
                    svst1_bf16(pt16_w, results + B_idx * B_step + N_idx * N_step + col_offset, data_bf16_0);

                    svfloat32_t data_1 = svread_hor_za32_f32_m(svfloat32_t(),pt32,1,static_cast<uint32_t>(inner_row_idx));
                    svbfloat16_t data_bf16_1 = svcvt_bf16_f32_z(pt32, data_1);
                    data_bf16_1 = svuzp1_bf16(data_bf16_1, svdup_bf16(0.0));
                    int row_index_global_1 = row_index_global + SVL_fp32;
                    int B_idx_1 = row_index_global_1 / seq_len;
                    int N_idx_1 = row_index_global_1 % seq_len;
                    svst1_bf16(pt16_w, results + B_idx_1 * B_step + N_idx_1 * N_step + col_offset, data_bf16_1);

                    svfloat32_t data_2 = svread_hor_za32_f32_m(svfloat32_t(),pt32,2,static_cast<uint32_t>(inner_row_idx));
                    svbfloat16_t data_bf16_2 = svcvt_bf16_f32_z(pt32, data_2);
                    data_bf16_2 = svuzp1_bf16(data_bf16_2, svdup_bf16(0.0));
                    int row_index_global_2 = row_index_global_1 + SVL_fp32;
                    int B_idx_2 = row_index_global_2 / seq_len;
                    int N_idx_2 = row_index_global_2 % seq_len;
                    svst1_bf16(pt16_w, results + B_idx_2 * B_step + N_idx_2 * N_step + col_offset, data_bf16_2);

                    svfloat32_t data_3 = svread_hor_za32_f32_m(svfloat32_t(),pt32,3,static_cast<uint32_t>(inner_row_idx));
                    svbfloat16_t data_bf16_3 = svcvt_bf16_f32_z(pt32, data_3);
                    data_bf16_3 = svuzp1_bf16(data_bf16_3, svdup_bf16(0.0));
                    int row_index_global_3 = row_index_global_2 + SVL_fp32;
                    int B_idx_3 = row_index_global_3 / seq_len;
                    int N_idx_3 = row_index_global_3 % seq_len;
                    svst1_bf16(pt16_w, results + B_idx_3 * B_step + N_idx_3 * N_step + col_offset, data_bf16_3);

                }
            }
            else{
                int col_offset_0 = H_idx * H_step + D_idx; // D_idx = 16
                int col_offset_1 = (H_idx + 1) * H_step - 8 ;
                #pragma unroll(8)
                for (int inner_row_idx=0; inner_row_idx < SVL_fp32; inner_row_idx++) 
                {
                    svfloat32_t data_0 = svread_hor_za32_f32_m(svfloat32_t(),pt32,0,static_cast<uint32_t>(inner_row_idx));
                    svbfloat16_t data_bf16_0 = svcvt_bf16_f32_z(pt32, data_0);
                    data_bf16_0 = svuzp1_bf16(data_bf16_0, svdup_bf16(0.0));
                    int row_index_global = row_idx + row_index + inner_row_idx;
                    int B_idx = row_index_global / seq_len;
                    int N_idx = row_index_global % seq_len;
                    svst1_bf16(pt16_8, results + B_idx * B_step + N_idx * N_step + col_offset_0, data_bf16_0);
                    svst1_bf16(pt16_8_16, results + B_idx * B_step + N_idx * N_step + col_offset_1, data_bf16_0);

                    svfloat32_t data_1 = svread_hor_za32_f32_m(svfloat32_t(),pt32,1,static_cast<uint32_t>(inner_row_idx));
                    svbfloat16_t data_bf16_1 = svcvt_bf16_f32_z(pt32, data_1);
                    data_bf16_1 = svuzp1_bf16(data_bf16_1, svdup_bf16(0.0));
                    int row_index_global_1 = row_index_global + SVL_fp32;
                    int B_idx_1 = row_index_global_1 / seq_len;
                    int N_idx_1 = row_index_global_1 % seq_len;
                    svst1_bf16(pt16_8, results + B_idx_1 * B_step + N_idx_1 * N_step + col_offset_0, data_bf16_1);
                    svst1_bf16(pt16_8_16, results + B_idx_1 * B_step + N_idx_1 * N_step + col_offset_1, data_bf16_1);

                    svfloat32_t data_2 = svread_hor_za32_f32_m(svfloat32_t(),pt32,2,static_cast<uint32_t>(inner_row_idx));
                    svbfloat16_t data_bf16_2 = svcvt_bf16_f32_z(pt32, data_2);
                    data_bf16_2 = svuzp1_bf16(data_bf16_2, svdup_bf16(0.0));
                    int row_index_global_2 = row_index_global_1 + SVL_fp32;
                    int B_idx_2 = row_index_global_2 / seq_len;
                    int N_idx_2 = row_index_global_2 % seq_len;
                    svst1_bf16(pt16_8, results + B_idx_2 * B_step + N_idx_2 * N_step + col_offset_0, data_bf16_2);
                    svst1_bf16(pt16_8_16, results + B_idx_2 * B_step + N_idx_2 * N_step + col_offset_1, data_bf16_2);

                    svfloat32_t data_3 = svread_hor_za32_f32_m(svfloat32_t(),pt32,3,static_cast<uint32_t>(inner_row_idx));
                    svbfloat16_t data_bf16_3 = svcvt_bf16_f32_z(pt32, data_3);
                    data_bf16_3 = svuzp1_bf16(data_bf16_3, svdup_bf16(0.0));
                    int row_index_global_3 = row_index_global_2 + SVL_fp32;
                    int B_idx_3 = row_index_global_3 / seq_len;
                    int N_idx_3 = row_index_global_3 % seq_len;
                    svst1_bf16(pt16_8, results + B_idx_3 * B_step + N_idx_3 * N_step + col_offset_0, data_bf16_3);
                    svst1_bf16(pt16_8_16, results + B_idx_3 * B_step + N_idx_3 * N_step + col_offset_1, data_bf16_3);
                }
            }
        }
    }
    SME_OFF();
}

void bgemm_4vl_1vl_bf16_impl(bfloat16_t *mat1, bfloat16_t *mat2, bfloat16_t * results,int M, int N, int K, int c_stride)
{
    svbool_t pt16 = svptrue_b16();
    svbool_t pt16_w = svwhilelt_b16(0,16);
    svbool_t pt32 = svptrue_b32();
    const int row_step = 4 * SVL_fp32;
    const int row_single_step = SVL_fp32;
    const int col_step = SVL_fp32;
    SME_ON();
    for(int row_index = 0; row_index < M; row_index += row_step)
    {
        for(int col_index = 0; col_index < N; col_index += col_step)
        {
            bfloat16_t* mat1_base0 = mat1 + row_index * K;
            bfloat16_t* mat1_base1 = mat1 + (row_index + row_single_step ) * K;
            bfloat16_t* mat1_base2 = mat1 + (row_index + 2 * row_single_step ) * K;
            bfloat16_t* mat1_base3 = mat1 + (row_index + 3 * row_single_step ) * K;
            bfloat16_t* mat2_base0 = mat2 + col_index * K ;

            svzero_za();
            for(int k_idx = 0; k_idx < SVL_fp32 * K; k_idx += SVL_bf16)
            {
                svbfloat16_t rowB0 = svld1_bf16(pt16,mat2_base0);

                svbfloat16_t colA0 = svld1_bf16(pt16,mat1_base0); 
                svmopa_za32_bf16_m(0,pt16,pt16,colA0,rowB0);
                svbfloat16_t colA1 = svld1_bf16(pt16,mat1_base1); 
                svmopa_za32_bf16_m(1,pt16,pt16,colA1,rowB0);
                svbfloat16_t colA2 = svld1_bf16(pt16,mat1_base2); 
                svmopa_za32_bf16_m(2,pt16,pt16,colA2,rowB0);
                svbfloat16_t colA3 = svld1_bf16(pt16,mat1_base3); 
                svmopa_za32_bf16_m(3,pt16,pt16,colA3,rowB0);
                mat2_base0 += SVL_bf16;
                mat1_base0 += SVL_bf16;
                mat1_base1 += SVL_bf16;
                mat1_base2 += SVL_bf16;
                mat1_base3 += SVL_bf16;
            }
            #pragma unroll(8)
            for (int inner_row_idx=0; inner_row_idx < SVL_fp32; inner_row_idx++) 
            {
                svfloat32_t data_0 = svread_hor_za32_f32_m(svfloat32_t(),pt32,0,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_0 = svcvt_bf16_f32_z(pt32, data_0);
                data_bf16_0 = svuzp1_bf16(data_bf16_0, svdup_bf16(0.0));
                svst1_bf16(pt16_w, results + (row_index + inner_row_idx) * c_stride + col_index, data_bf16_0);

                svfloat32_t data_1 = svread_hor_za32_f32_m(svfloat32_t(),pt32,1,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_1 = svcvt_bf16_f32_z(pt32, data_1);
                data_bf16_1 = svuzp1_bf16(data_bf16_1, svdup_bf16(0.0));
                svst1_bf16(pt16_w, results + (row_index + inner_row_idx + SVL_fp32) * c_stride + col_index, data_bf16_1);

                svfloat32_t data_2 = svread_hor_za32_f32_m(svfloat32_t(),pt32,2,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_2 = svcvt_bf16_f32_z(pt32, data_2);
                data_bf16_2 = svuzp1_bf16(data_bf16_2, svdup_bf16(0.0));
                svst1_bf16(pt16_w, results + (row_index + inner_row_idx + 2 * SVL_fp32) * c_stride + col_index, data_bf16_2);

                svfloat32_t data_3 = svread_hor_za32_f32_m(svfloat32_t(),pt32,3,static_cast<uint32_t>(inner_row_idx));
                svbfloat16_t data_bf16_3 = svcvt_bf16_f32_z(pt32, data_3);
                data_bf16_3 = svuzp1_bf16(data_bf16_3, svdup_bf16(0.0));
                svst1_bf16(pt16_w, results + (row_index + inner_row_idx + 3 * SVL_fp32) * c_stride + col_index, data_bf16_3);

            }
        }
    }
    SME_OFF();
}


void bf16_gemm_pack_singlethread(int64_t r, int64_t c, int64_t split_r, int64_t split_c, bfloat16_t *input_ptr,
    bfloat16_t *output_ptr)
{
    gemm_pack_thread_task<bfloat16_t, false>(0, r, c, split_r, split_c, input_ptr, output_ptr, nullptr, 0, 1);
}

void bf16_packed_gemm_singlethread(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, bfloat16_t *act_ptr, bfloat16_t *weight_ptr,
    bfloat16_t *output_ptr, bfloat16_t *tmpc, float *bias, bool row_bias)
{
    int64_t tile_m = std::get<0>(t);
    int64_t tile_n = std::get<1>(t);
    int64_t tile_k = std::get<2>(t);
    int64_t blocks_in_m = m / tile_m;
    int64_t blocks_in_n = n / tile_n;
    int64_t blocks_in_k = k / tile_k;
    for(int64_t idx_m = 0; idx_m < blocks_in_m; idx_m++ ){
        for(int64_t idx_n = 0; idx_n < blocks_in_n; idx_n++ ){
            for(int64_t idx_k = 0; idx_k < blocks_in_k; idx_k++ ){
                bfloat16_t *a = act_ptr + idx_m * tile_m * k + idx_k * tile_m * tile_k;
                bfloat16_t *b = weight_ptr + idx_n * tile_n * k + idx_k * tile_n * tile_k;
                bfloat16_t *c = output_ptr + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
                float *sub_bias = nullptr;
                if (bias != nullptr) {
                    if (row_bias) {
                        sub_bias = bias + idx_n * tile_n;
                    } else {
                        sub_bias = bias + idx_m * tile_m;
                    }
                }
                bf16_packed_gemm_alpha1_beta0(tile_m, tile_n, n, tile_k, a, b, c, sub_bias, row_bias);
            }
        }
    }
}

}//namespace kutacc