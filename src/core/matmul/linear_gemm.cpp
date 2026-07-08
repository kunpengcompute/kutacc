/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "linear_gemm.h"
#include "utils/check.h"

namespace kutacc {
int64_t get_s8_s8_gemm_bf16_dq_tmpc_size(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t)
{
    return m * n * 2 * k / std::get<2>(t);
}

__attribute__((always_inline)) void s8_packed_gemm_bf16_dq_kernel_filter(int64_t m, int64_t n, int64_t k,
    int64_t tile_m, int64_t tile_n, int64_t tile_k, int8_t *a, int8_t *b, bfloat16_t *c, float *rscales, float *cscales)
{
    if (tile_k >= 448 && tile_k <= 2560 && tile_k % 16 == 0) {
        s8_packed_gemm_bf16_dq_alpha1_beta0_m_lt_128_k_2048(tile_m, tile_n, n, tile_k,
            a, b, c, rscales, cscales, 128, tile_n, tile_k);
    } else {
        s8_packed_gemm_bf16_dq_alpha1_beta0(tile_m, tile_n, n, tile_k, a, b, c, rscales, cscales);
    }
}

void s8_packed_gemm_bf16_dq_dynamic_tile_n_thread_task(int64_t thread_id, int64_t m, int64_t n, int64_t k,
    MatrixTilingBlock t, int8_t *act_ptr, int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr,
    bfloat16_t *output_ptr, bfloat16_t *tmpc, int64_t num_threads)
{
    int64_t tile_m = std::get<0>(t);
    KUTACC_CHECK(tile_m == m, " tile_m = ", tile_m, " m = ", m, "Dynamic tile_n need tile_m == m");
    int64_t tile_n = std::get<1>(t);
    int64_t tile_k = std::get<2>(t);

    int64_t step = 0;
    while (step < m) {
        int64_t cur_m;
        int64_t cur_tile_m;
        int64_t cur_tile_n;
        get_cur_m_stride(m - step, n, tile_n, cur_m, cur_tile_m, cur_tile_n);
        cur_tile_n = (cur_tile_n + 15) / 16 * 16;
        int64_t blocks_in_m = cur_m / cur_tile_m;
        int64_t blocks_in_n = (n + cur_tile_n - 1) / cur_tile_n;
        int64_t blocks_in_k = k / tile_k;

        auto [idx_m, idx_n, idx_k] = get_idxs(thread_id, blocks_in_m, blocks_in_n, blocks_in_k);
        bool c_reused = (blocks_in_k <= 4);
        int8_t *a = (act_ptr + step * tile_k) + idx_m * cur_tile_m * tile_k + idx_k * m * tile_k;
        int8_t *b = weight_ptr + idx_n * cur_tile_n * tile_k + idx_k * n * tile_k;
        bfloat16_t *c = (output_ptr + step * n) + n * m * idx_k + idx_m * cur_tile_m * n + idx_n * cur_tile_n;
        bfloat16_t *nowc = (tmpc + step * n) + n * m * idx_k + idx_m * cur_tile_m * n + idx_n * cur_tile_n;
        bfloat16_t *target_c = (c_reused && idx_k == 0) ? c : nowc;
        float *cscales = (act_scale_ptr + step) + idx_m * cur_tile_m;
        float *rscales = weight_scale_ptr + idx_n * cur_tile_n;
        bfloat16_t alpha = 1;
        bfloat16_t beta = 0;

        s8_packed_gemm_bf16_dq_kernel_filter(cur_m, n, k, cur_tile_m, std::min(cur_tile_n, n - idx_n * cur_tile_n),
            tile_k, a, b, target_c, rscales, cscales);

        step += cur_m;
    }

    int64_t reduce_start = (m * n * thread_id / num_threads + 31) / 32 * 32;
    int64_t reduce_end = std::min(m * n, (m * n * (thread_id + 1) / num_threads + 31) / 32 * 32);
    reduce_filter(m, n, k, tile_k, output_ptr, tmpc, reduce_start, reduce_end);
}

void s8_packed_gemm_bf16_dq_dynamic_n_slice_thread_task(
    int64_t thread_id, int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int64_t n_begin, int64_t n_length,
    int8_t *act_ptr, int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr,
    bfloat16_t *output_ptr, bfloat16_t *tmpc,
    int64_t num_threads
)
{
    int64_t tile_m = std::get<0>(t);
    int64_t tile_n = std::get<1>(t);
    int64_t tile_k = std::get<2>(t);
    int64_t blocks_in_m = m / tile_m;
    int64_t blocks_in_n = (n_length + tile_n - 1) / tile_n;
    int64_t blocks_in_k = k / tile_k;

    auto [idx_m, idx_n, idx_k] = get_idxs(thread_id, blocks_in_m, blocks_in_n, blocks_in_k);
    bool c_reused = (blocks_in_k <= 4);
    int8_t *a = act_ptr + idx_m * tile_m * tile_k + idx_k * m * tile_k;
    int8_t *b = (weight_ptr + n_begin * tile_k) + idx_n * tile_n * tile_k + idx_k * n * tile_k;
    bfloat16_t *c = (output_ptr + n_begin) + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
    bfloat16_t *nowc = (tmpc + n_begin) + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
    bfloat16_t *target_c = (c_reused && idx_k == 0) ? c : nowc;
    float *cscales = act_scale_ptr + idx_m * tile_m;
    float *rscales = (weight_scale_ptr + n_begin) + idx_n * tile_n;

    s8_packed_gemm_bf16_dq_kernel_filter(
        m, n, k, tile_m, std::min(n_length - idx_n * tile_n, tile_n), tile_k, a, b, target_c, rscales, cscales);

    int64_t reduce_start = (m * n_length  * thread_id / num_threads + 31) / 32 * 32;
    int64_t reduce_end = std::min(m * n_length, (m * n_length * (thread_id + 1) / num_threads + 31) / 32 * 32);
    reduce_stride_filter(m, n, k, tile_k, output_ptr, tmpc, reduce_start, reduce_end, n_begin, n_length);
}

void s8_packed_gemm_bf16_dq_thread_task(int64_t thread_id, int64_t m, int64_t n, int64_t k, MatrixTilingBlock t,
    int8_t *act_ptr, int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr,
    bfloat16_t *tmpc, int64_t num_threads)
{
    int64_t tile_m = std::get<0>(t);
    int64_t tile_n = std::get<1>(t);
    int64_t tile_k = std::get<2>(t);
    int64_t blocks_in_m = m / tile_m;
    int64_t blocks_in_n = (n + tile_n - 1) / tile_n;
    int64_t blocks_in_k = k / tile_k;

    auto [idx_m, idx_n, idx_k] = get_idxs(thread_id, blocks_in_m, blocks_in_n, blocks_in_k);
    int8_t *a = act_ptr + idx_m * tile_m * k + idx_k * tile_m * tile_k;
    int8_t *b = weight_ptr + idx_n * tile_n * k + idx_k * tile_n * tile_k;

    bfloat16_t *c = output_ptr + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
    bfloat16_t *nowc = tmpc + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
    bool c_reused = (blocks_in_k <= 4);
    bfloat16_t *target_c = (c_reused && idx_k == 0) ? c : nowc;
    float *cscales = act_scale_ptr + idx_m * tile_m;
    float *rscales = weight_scale_ptr + idx_n * tile_n;
    bfloat16_t alpha = 1;
    bfloat16_t beta = 0;

    s8_packed_gemm_bf16_dq_kernel_filter(
        m, n, k, tile_m, std::min(n - idx_n * tile_n, tile_n), std::get<2>(t), a, b, target_c, rscales, cscales);
    
    int64_t reduce_start = (m * n * thread_id / num_threads + 31) / 32 * 32;
    int64_t reduce_end = std::min(m * n, (m * n * (thread_id + 1) / num_threads + 31) / 32 * 32);
    reduce_filter(m, n, k, tile_k, output_ptr, tmpc, reduce_start, reduce_end);
}

void s8_s8_packed_gemm_bf16_dq(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr,
    int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr, bfloat16_t *tmpc)
{
    int64_t num_threads = kutacc::get_thread_num();
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        int64_t thread_id = kutacc::get_thread_id();
        s8_packed_gemm_bf16_dq_thread_task(thread_id, m, n, k, t, act_ptr, weight_ptr, act_scale_ptr,
            weight_scale_ptr, output_ptr, tmpc, num_threads);
    });
}

void s8_packed_gemm_bf16_dq_dynamic_tile_n(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr,
    int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr, bfloat16_t *tmpc)
{
    int64_t num_threads = kutacc::get_thread_num();
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        int64_t thread_id = kutacc::get_thread_id();
        s8_packed_gemm_bf16_dq_dynamic_tile_n_thread_task(thread_id, m, n, k, t, act_ptr, weight_ptr, act_scale_ptr,
            weight_scale_ptr, output_ptr, tmpc, num_threads);
    });
}

void s8_s8_gemm_bf16_dq(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr, int8_t *input_ptr,
    int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr, bfloat16_t *tmpc)
{
    int64_t tile_m = std::get<0>(t);
    int64_t tile_n = std::get<1>(t);
    int64_t tile_k = std::get<2>(t);
    int64_t num_threads = kutacc::get_thread_num();

    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        int64_t thread_id = kutacc::get_thread_id();
        gemm_pack_thread_task<int8_t, false>(
            thread_id, m, k, tile_m, tile_k, act_ptr, input_ptr, nullptr, k, num_threads);
        kutacc::parallel_barrier();
        s8_packed_gemm_bf16_dq_thread_task(thread_id, m, n, k, t, input_ptr, weight_ptr, act_scale_ptr,
            weight_scale_ptr, output_ptr, tmpc, num_threads);
    });
}
void bf16_packed_gemm(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, bfloat16_t *act_ptr, bfloat16_t *weight_ptr,
    bfloat16_t *output_ptr, bfloat16_t *tmpc, float *bias, bool row_bias)
{
    int64_t tile_m = std::get<0>(t);
    int64_t tile_n = std::get<1>(t);
    int64_t tile_k = std::get<2>(t);
    int64_t blocks_in_m = m / tile_m;
    int64_t blocks_in_n = n / tile_n;
    int64_t blocks_in_k = k / tile_k;
    int64_t num_threads = kutacc::get_thread_num();
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        int64_t thread_id = kutacc::get_thread_id();
        auto [idx_m, idx_n, idx_k] = get_idxs(thread_id, blocks_in_m, blocks_in_n, blocks_in_k);
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
        if (blocks_in_k == 1) {
            bf16_packed_gemm_alpha1_beta0(tile_m, tile_n, n, tile_k, a, b, c, sub_bias, row_bias);
        } else {
            bfloat16_t *nowc = tmpc + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
            bf16_packed_gemm_alpha1_beta0(tile_m, tile_n, n, tile_k, a, b, nowc, bias, row_bias);
            kutacc::parallel_barrier();
            int64_t step = (m * n + num_threads - 1) / num_threads;
            int64_t start = step * thread_id;
            int64_t end = std::min(start + step, m * n);
            if (start < end) {
                reduce_for_kblocks(m, n, blocks_in_k, output_ptr, tmpc, start, end);
            }
        }
    });
}

void batch_bf16_s8_packed_gemm_bf16(int64_t bs, int64_t m, int64_t n, int64_t k, int64_t stride_bs, int64_t stride_m,
    bfloat16_t* act, int8_t* weight, bfloat16_t* out, float* rscale, float* cscale)
{
    int64_t num_threads = kutacc::get_thread_num();
    int64_t m_threads = std::max(num_threads / bs, int64_t{1});
    int64_t block_m = (m + m_threads - 1) / m_threads;
    kutacc::parallel_for(0, bs * m_threads, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            int64_t bi = i / m_threads;
            int64_t m_id = i % m_threads;
            int64_t start_m = m_id * block_m;
            int64_t end_m = std::min((m_id + 1) * block_m, m);

            float* rscale_data = rscale ? rscale + bi * n : nullptr;
            float* cscale_data = cscale ? cscale + bi * k : nullptr;
            auto weight_data = weight + bi * n * k;
            bfloat16_t* act_data = act + bi * m * k + start_m * k;
            bfloat16_t* out_data = out + bi * stride_bs + start_m * stride_m;
            int lda = k;
            int ldb = k;
            int ldc = stride_m;
            kutacc::batch_bf16_s8_packed_gemm_bf16_alpha1_beta0(end_m - start_m, n, k, act_data, lda, weight_data,
                ldb, out_data, ldc, rscale_data, cscale_data);
        }
    });
}
}
