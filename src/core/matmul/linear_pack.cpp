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
#include "linear_pack.h"

namespace kutacc {

template <typename scalar_t, bool with_idx>
void gemm_pack_thread_task(int64_t thread_id, int64_t m, int64_t n, int64_t tm, int64_t tn, scalar_t *i_ptr,
    scalar_t *o_ptr, int *idx, int64_t ldi, int thread_nums)
{
    int64_t blocks_m = m / tm;
    int64_t blocks_n = n / tn;
    int64_t threads_blocks = blocks_m * blocks_n;
    int64_t remain_threads = thread_nums / threads_blocks;

    int64_t subblock_m = (tm + remain_threads * 16 - 1) / (remain_threads * 16) * 16;
    int64_t threads_tm = (tm + subblock_m - 1) / subblock_m;
    int64_t threads_tn = remain_threads / threads_tm * 4 / sizeof(scalar_t);
    int64_t subblock_n = (tn + threads_tn - 1) / threads_tn * 4 / sizeof(scalar_t);
    threads_tn = std::min((tn + subblock_n - 1) / subblock_n, remain_threads / threads_tm);

    int64_t threads_subblocks = threads_tm * threads_tn;
    if (n <= 512) {
        subblock_m = tm, subblock_n = tn, threads_tm = threads_tn = threads_subblocks = 1;
    }

    // 线程顺序是先内后外，先N后M，因此tile_m == m时以M * tile_n的模式pack，否则以tile_m * tile_n的模式pack
    if (thread_id < threads_blocks * threads_subblocks) {
        int x = (thread_id / threads_subblocks) / blocks_n;
        int y = (thread_id / threads_subblocks) % blocks_n;
        int subx = (thread_id % threads_subblocks) / threads_tn;
        int suby = (thread_id % threads_subblocks) % threads_tn;
        int bm = std::min(subblock_m, tm - subx * subblock_m);
        int bn = std::min(subblock_n, tn - suby * subblock_n);
        int x_off = x * tm + subx * subblock_m;
        int y_off = y * tn + suby * subblock_n;
        int o_off =
            thread_id / threads_subblocks * tm * tn + subx * subblock_m * tn + suby * subblock_n * std::min(bm, 16);
        if constexpr (!with_idx) {
            if constexpr (std::is_same_v<scalar_t, int8_t>) {
                kutacc::s8_gemm_pack_kernel(bm, bn, i_ptr + x_off * ldi + y_off, ldi, o_ptr + o_off, tn);
            } else if constexpr (std::is_same_v<scalar_t, bfloat16_t>) {
                kutacc::bf16_gemm_pack_kernel(bm, bn, i_ptr + x_off * n + y_off, n, o_ptr + o_off, tn);
            }
        } else {
            kutacc::s8_gemm_pack_with_idx_kernel(bm, bn, i_ptr + y_off, ldi, x_off, idx, o_ptr + o_off, tn);
        }
    }
}

template <typename scalar_t, bool with_idx>
void gemm_pack(int64_t m, int64_t n, int64_t tm, int64_t tn, scalar_t *i_ptr, scalar_t *o_ptr, int *idx, int64_t ldi)
{
    int64_t num_threads = kutacc::get_thread_num();
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        int64_t thread_id = kutacc::get_thread_id();
        gemm_pack_thread_task<scalar_t, with_idx>(thread_id, m, n, tm, tn, i_ptr, o_ptr, idx, ldi, num_threads);
    });
}

void s8_gemm_pack(int64_t m, int64_t n, int64_t tm, int64_t tn, int8_t *i_ptr, int8_t *o_ptr,
    int64_t ldi, bool with_idx, int *idx)
{
    if (with_idx) {
        gemm_pack<int8_t, true>(m, n, tm, tn, i_ptr, o_ptr, idx, ldi);
    } else {
        // 默认i_ptr的stride[0] = n 即ldi = n
        ldi = (ldi == 0) ? n : ldi;
        gemm_pack<int8_t, false>(m, n, tm, tn, i_ptr, o_ptr, idx, ldi);
    }
}

void s8_gemm_pack_fusedmoe(int64_t m, int64_t n, int64_t tn, int8_t *i_ptr, int8_t *o_ptr,
    int64_t ldi, bool with_idx, int *idx)
{
    s8_gemm_pack(m, n, m, tn, i_ptr, o_ptr, ldi, with_idx, idx);
}


void bf16_gemm_pack(int64_t r, int64_t c, int64_t split_r, int64_t split_c, bfloat16_t *input_ptr,
    bfloat16_t *output_ptr)
{
    gemm_pack<bfloat16_t, false>(r, c, split_r, split_c, input_ptr, output_ptr, nullptr, 0);
}

void batch_bf16_gemm_pack(int64_t bs, int64_t m, int64_t n, int64_t stride_bs, int64_t stride_m, void* src, void* dst,
    int64_t dtype)
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

            uint8_t* src_off = (uint8_t*)src + (bi * stride_bs + start_m * stride_m) * dtype;
            uint8_t* dst_off = (uint8_t*)dst + (bi * m * n + start_m * n) * dtype;
            char trans;
            char pack_matrix;
            if (dtype == 1) {
                trans = 'T';
                pack_matrix = 'A';
            } else if (dtype == 2) {
                trans = 'N';
                pack_matrix = 'B';
            }
            gemm_ex_t extra{.nThreads = 1};
            batch_bf16_gemm_pack_kernel(&pack_matrix, &trans, end_m - start_m, n, src_off, stride_m, dst_off, &extra);
        }
    });
}
}
