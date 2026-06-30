#include "kutacc.h"
#include "common.h"
#include "../core/matmul/linear_pack.h"
#include "../core/matmul/linear_gemm.h"

namespace kutacc{
void bgemm_4vl_1vl_bf16_transpose_impl(bfloat16_t *mat1, bfloat16_t *mat2, bfloat16_t * results,int M, int N, int K,
    int row_idx, int col_idx, int seq_len, int nheads, int head_size);
void bgemm_bias_4vl_1vl_bf16_transpose_impl(bfloat16_t *mat1, bfloat16_t *mat2, float *bias, bfloat16_t * results,int M, int N, int K,
    int row_idx, int col_idx, int seq_len, int nheads, int head_size);
void bgemm_4vl_1vl_bf16_impl(bfloat16_t *mat1, bfloat16_t *mat2, bfloat16_t * results,int M, int N, int K, int c_stride);
void bgemm_4vl_1vl_bf16_trans_impl(bfloat16_t *mat1, bfloat16_t *mat2, bfloat16_t * results,int M, int N, int K, int c_stride,
    int row_idx, int col_idx, int batch, int seq_len, bool trans);
void bgemm_4vl_1vl_bf16_transpose_trans_impl(bfloat16_t *mat1, bfloat16_t *mat2, bfloat16_t * results,int M, int N, int K,
    int row_idx, int col_idx, int batch, int seq_len, int nheads, int head_size, bool trans);
void bgemm_bias_4vl_1vl_bf16_transpose_24_impl(bfloat16_t *mat1, bfloat16_t *mat2, float *bias, bfloat16_t * results,int M, int N, int K,
    int row_idx, int col_idx, int seq_len, int nheads, int head_size);
void bgemm_4vl_1vl_bf16_transpose_24_impl(bfloat16_t *mat1, bfloat16_t *mat2, bfloat16_t * results,int M, int N, int K,
    int row_idx, int col_idx, int seq_len, int nheads, int head_size);

inline std::pair<int64_t, int64_t> compute_tm_tn(int64_t m, int64_t n) {
    if (m == 0 || n == 0) {
        return {1, 1};
    }

    int64_t x = m / n;
    int64_t y = n / m;
    int64_t tm, tn;
    if( m % 1152 == 0 ){ // 36 * 32
        tm = m / 36;
        tn = n;
    } else if ( m % 192 == 0 && n % 192 == 0){ // 6 * 32
        tm = m / 6;
        tn = n / 6;
    }
    if (x > 20 && m % 32 == 0) {
        tm = m / 32;
        tn = n;
    } else if (x > 5 && m % 16 == 0) {
        tm = m / 16;
        tn = n / 2;
    } else if (x >= 1) {
        tm = m / 8;
        tn = n / 4;
    } else if (y > 20 && n % 32 == 0) {
        tm = m;
        tn = n / 32;
    } else if (y > 5 && n % 16 == 0) {
        tm = m / 2;
        tn = n / 16;
    } else {
        tm = m / 4;
        tn = n / 8;
    }

    if (tm < 1) tm = 1;
    if (tn < 1) tn = 1;

    return {tm, tn};
}

void bf16_gemm_pack_singlethread(int64_t r, int64_t c, int64_t split_r, int64_t split_c,
    bfloat16_t *input_ptr, bfloat16_t *output_ptr);
void bf16_packed_gemm_singlethread(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t,
    bfloat16_t *act_ptr, bfloat16_t *weight_ptr,
    bfloat16_t *output_ptr, bfloat16_t *tmpc, float *bias, bool row_bias);

}//namespace kutacc