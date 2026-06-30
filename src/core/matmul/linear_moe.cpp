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
#include <tuple>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <optional>
#include <arm_sve.h>
#include <arm_sme.h>
#include <arm_neon.h>

#include "kernel/kernel.h"
#include "linear_gemm.h"
#include "linear_utils.h"
#include "kutacc.h"

namespace kutacc {

std::tuple<int64_t, int64_t, int64_t> get_fusedmoe_s8gemm_workspace_size(int64_t N, int64_t K, int64_t tilebuf_size,
    MatrixTilingBlock t)
{
    int64_t tmpx_size = tilebuf_size * K;
    int64_t tmpy_size = tilebuf_size * N * (K / std::get<2>(t)) / 2;
    int64_t tmpscale_size = tilebuf_size * 4;
    return std::make_tuple(tmpx_size, tmpy_size, tmpscale_size);
}

void fusedmoe_gateup_buffer_expansion(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int64_t lda,
    int64_t ldas, int8_t *acts, int8_t *weights, float *acts_scale, float *weights_scale, int *token_ids,
    int *experts_offset, bfloat16_t *output, int8_t *tmpx, float *tmpy, float *tmp_scales, MatrixTilingBlock t)
{
    // 1. Gather scale
    for (int64_t ib = 0; ib < total_bs; ib++) {
        tmp_scales[ib] = acts_scale[token_ids[ib] * ldas];
    }
    int64_t tile_n = std::get<1>(t);
    int64_t tile_k = std::get<2>(t);
    int64_t num_threads = kutacc::get_thread_num();
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        int64_t thread_id = kutacc::get_thread_id();
        // 2. Pack act with ids
        for (int64_t i = 0; i < num_experts; i++) {
            auto m = experts_offset[i + 1] - experts_offset[i];
            auto bias = experts_offset[i] - experts_offset[0];
            gemm_pack_thread_task<int8_t, true>(
                thread_id, m, K, m, tile_k, acts, tmpx + bias * K, token_ids + bias, lda, num_threads);
        }
        kutacc::parallel_barrier();
        // 3. Call IGEMMBDQ kernel
        for (int64_t i = 0; i < num_experts; i++) {
            auto m = experts_offset[i + 1] - experts_offset[i];
            auto bias = experts_offset[i] - experts_offset[0];
            s8_packed_gemm_bf16_dq_dynamic_tile_n_thread_task(
                thread_id, m, N, K, MatrixTilingBlock(m, tile_n, tile_k), tmpx + bias * K, weights + i * N * K,
                tmp_scales + bias, weights_scale + i * N, output + bias * N,
                reinterpret_cast<bfloat16_t *>(tmpy + 2 * bias * N), num_threads);
        }
    });
}

void fusedmoe_gateup_dualexpt_parallel(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int64_t lda,
    int64_t ldas, int8_t *acts, int8_t *weights, float *acts_scale, float *weights_scale, int *token_ids,
    int *experts_offset, bfloat16_t *output, int8_t *tmpx, float *tmpy, float *tmp_scales, MatrixTilingBlock t,
    int64_t n_slice)
{
    // 默认32线程专家并行中各用一半
    int64_t num_threads = kutacc::get_thread_num();
    int64_t num_threads0 = 16;
    int64_t num_threads1 = 16;
    MatrixTilingBlock default_tile(0, std::get<1>(t), std::get<2>(t));
    ExpPara exp0 = get_exp_para(0, experts_offset, K, N, num_threads0, default_tile);
    ExpPara exp1 = get_exp_para(1, experts_offset, K, N, num_threads1, default_tile);

    auto reorder_exp = reorder_exppara(exp0, exp1);
    ExpPara &bigger_exp = std::get<0>(reorder_exp);
    ExpPara &smaller_exp = std::get<1>(reorder_exp);

    bool n_sliced = (n_slice < N);
    auto retile_exp = reset_task_tile(num_threads, num_threads0, bigger_exp, N, K, default_tile, n_slice);
    ExpPara &bigger_exp_fst = std::get<0>(retile_exp);
    ExpPara &bigger_exp_scd = std::get<1>(retile_exp);

    for (int64_t ib = 0; ib < total_bs; ib++) {
        tmp_scales[ib] = acts_scale[token_ids[ib] * ldas];
    }
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        int thread_id = kutacc::get_thread_id();
        for (int64_t i = 0; i < num_experts; i++) {
            auto m = experts_offset[i + 1] - experts_offset[i];
            auto bias = experts_offset[i] - experts_offset[0];
            gemm_pack_thread_task<int8_t, true>(
                thread_id, m, K, m, std::get<2>(default_tile), acts, tmpx + bias * K, token_ids + bias,
                lda, num_threads);
        }
        kutacc::parallel_barrier();

        ExpPara ept;
        int local_thread_id;
        if (thread_id < bigger_exp_fst.num_threads + smaller_exp.num_threads) {
            if (thread_id < bigger_exp_fst.num_threads) {
                local_thread_id = thread_id;
                ept = bigger_exp_fst;
                if (n_sliced) {
                    s8_packed_gemm_bf16_dq_dynamic_n_slice_thread_task(
                        local_thread_id, ept.m, N, K, ept.t, 0, n_slice, tmpx + ept.bias * K, weights + ept.ei * N * K,
                        tmp_scales + ept.bias, weights_scale + ept.ei * N,
                        output + ept.bias * N, reinterpret_cast<bfloat16_t *>(tmpy + 2 * ept.bias * N), ept.num_threads
                    );
                } else {
                    s8_packed_gemm_bf16_dq_dynamic_tile_n_thread_task(
                        local_thread_id, ept.m, N, K, ept.t, tmpx + ept.bias * K, weights + ept.ei * N * K,
                        tmp_scales + ept.bias, weights_scale + ept.ei * N,
                        output + ept.bias * N, reinterpret_cast<bfloat16_t *>(tmpy + 2 * ept.bias * N), ept.num_threads
                    );
                }
            } else {
                local_thread_id = thread_id - num_threads0;
                ept = smaller_exp;
                s8_packed_gemm_bf16_dq_dynamic_tile_n_thread_task(
                    local_thread_id, ept.m, N, K, ept.t, tmpx + ept.bias * K, weights + ept.ei * N * K,
                    tmp_scales + ept.bias, weights_scale + ept.ei * N,
                    output + ept.bias * N, reinterpret_cast<bfloat16_t *>(tmpy + 2 * ept.bias * N), ept.num_threads
                );
            }
            if (n_sliced) {
                ept = bigger_exp_scd;
                local_thread_id = thread_id;
                s8_packed_gemm_bf16_dq_dynamic_n_slice_thread_task(
                    local_thread_id, ept.m, N, K, ept.t, n_slice, N - n_slice, tmpx + ept.bias * K,
                    weights + ept.ei * N * K, tmp_scales + ept.bias, weights_scale + ept.ei * N,
                    output + ept.bias * N, reinterpret_cast<bfloat16_t *>(tmpy + 2 * ept.bias * N), ept.num_threads);
            }
        } else {
            kutacc::parallel_barrier();
        }
    });
}

void fusedmoe_gateup_buffer_limited(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int64_t lda,
    int64_t ldas, int8_t *acts, int8_t *weights, float *acts_scale, float *weights_scale, int *token_ids,
    int *experts_offset, bfloat16_t *output, int8_t *tmpx, float *tmpy, float *tmp_scales, MatrixTilingBlock t,
    int64_t tilebuf_size)
{
    for (int64_t i = 0; i < num_experts; i++) {
        int64_t bs = experts_offset[i + 1] - experts_offset[i];
        int64_t bs_offset = experts_offset[i] - experts_offset[0];
        if (bs == 0) {
            continue;
        }
        int64_t step = 0;
        while (step < bs) {
            int64_t now_bs = std::min(tilebuf_size, bs - step);
            int *ids = token_ids + experts_offset[i] + step;
            int64_t m = now_bs;
            int64_t tile_n = std::get<1>(t);
            int64_t tile_k = std::get<2>(t);
            // 1. Gather scale
            for (int64_t ib = 0; ib < m; ib++) {
                tmp_scales[ib] = acts_scale[ids[ib] * ldas];
            }
            // 2. Pack act with ids
            s8_gemm_pack_fusedmoe(m, K, tile_k, acts, tmpx, lda, 1, ids);

            // 3. Call IGEMMBDQ kernel
            s8_packed_gemm_bf16_dq_dynamic_tile_n(m, N, K, MatrixTilingBlock(m, tile_n, tile_k), tmpx,
                weights + i * N * K, tmp_scales, weights_scale + i * N, output + (bs_offset + step) * N,
                reinterpret_cast<bfloat16_t *>(tmpy));

            step += m;
        }
    }
}

void fusedmoe_gateup(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int64_t lda, int64_t ldas,
    int8_t *acts, int8_t *weights, float *acts_scale, float *weights_scale, int *token_ids, int *experts_offset,
    bfloat16_t *output, int8_t *tmpx, float *tmpy, float *tmp_scales, MatrixTilingBlock t, int64_t tilebuf_size,
    std::optional<int64_t> n_slice)
{
    if (total_bs <= tilebuf_size) {
        if (num_experts == 2 && n_slice.has_value()) {
            fusedmoe_gateup_dualexpt_parallel(total_bs, K, N, num_experts, lda, ldas, acts, weights, acts_scale,
                weights_scale, token_ids, experts_offset, output, tmpx, tmpy, tmp_scales, t, *n_slice);
            return;
        }
        fusedmoe_gateup_buffer_expansion(total_bs, K, N, num_experts, lda, ldas, acts, weights, acts_scale,
            weights_scale, token_ids, experts_offset, output, tmpx, tmpy, tmp_scales, t);
    } else {
        fusedmoe_gateup_buffer_limited(total_bs, K, N, num_experts, lda, ldas, acts, weights, acts_scale,
            weights_scale, token_ids, experts_offset, output, tmpx, tmpy, tmp_scales, t, tilebuf_size);
    }
}

void fusedmoe_down_dualexpt_parallel(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int8_t *acts,
    int8_t *weights, float *acts_scale, float *weights_scale, int *experts_offset, bfloat16_t *output, int8_t *tmpx,
    float *tmpy, MatrixTilingBlock t, int64_t n_slice)
{
        // 默认32线程专家并行中各用一半
    int64_t num_threads = kutacc::get_thread_num();
    int64_t num_threads0 = 16;
    int64_t num_threads1 = 16;
    MatrixTilingBlock default_tile(0, std::get<1>(t), std::get<2>(t));
    ExpPara exp0 = get_exp_para(0, experts_offset, K, N, num_threads0, default_tile);
    ExpPara exp1 = get_exp_para(1, experts_offset, K, N, num_threads1, default_tile);

    auto reorder_exp = reorder_exppara(exp0, exp1);
    ExpPara &bigger_exp = std::get<0>(reorder_exp);
    ExpPara &smaller_exp = std::get<1>(reorder_exp);

    bool n_sliced = (n_slice < N);
    auto retile_exp = reset_task_tile(num_threads, num_threads0, bigger_exp, N, K, default_tile, n_slice);
    ExpPara &bigger_exp_fst = std::get<0>(retile_exp);
    ExpPara &bigger_exp_scd = std::get<1>(retile_exp);

    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        int thread_id = kutacc::get_thread_id();
        for (int64_t i = 0; i < num_experts; i++) {
            auto m = experts_offset[i + 1] - experts_offset[i];
            auto bias = experts_offset[i] - experts_offset[0];
            gemm_pack_thread_task<int8_t, false>(
                thread_id, m, K, m, std::get<2>(default_tile), acts + bias * K, tmpx + bias * K, nullptr, K,
                num_threads);
        }
        kutacc::parallel_barrier();

        ExpPara ept;
        int local_thread_id;
        if (thread_id < bigger_exp_fst.num_threads + smaller_exp.num_threads) {
            if (thread_id < bigger_exp_fst.num_threads) {
                local_thread_id = thread_id;
                ept = bigger_exp_fst;
                if (n_sliced) {
                    s8_packed_gemm_bf16_dq_dynamic_n_slice_thread_task(
                        local_thread_id, ept.m, N, K, ept.t, 0, n_slice, tmpx + ept.bias * K, weights + ept.ei * N * K,
                        acts_scale + ept.bias, weights_scale + ept.ei * N,
                        output + ept.bias * N, reinterpret_cast<bfloat16_t *>(tmpy + 2 * ept.bias * N), ept.num_threads
                    );
                } else {
                    s8_packed_gemm_bf16_dq_dynamic_tile_n_thread_task(
                        local_thread_id, ept.m, N, K, ept.t, tmpx + ept.bias * K, weights + ept.ei * N * K,
                        acts_scale + ept.bias, weights_scale + ept.ei * N,
                        output + ept.bias * N, reinterpret_cast<bfloat16_t *>(tmpy + 2 * ept.bias * N), ept.num_threads
                    );
                }
            } else {
                local_thread_id = thread_id - num_threads0;
                ept = smaller_exp;
                s8_packed_gemm_bf16_dq_dynamic_tile_n_thread_task(
                    local_thread_id, ept.m, N, K, ept.t, tmpx + ept.bias * K, weights + ept.ei * N * K,
                    acts_scale + ept.bias, weights_scale + ept.ei * N,
                    output + ept.bias * N, reinterpret_cast<bfloat16_t *>(tmpy + 2 * ept.bias * N), ept.num_threads
                );
            }
            if (n_sliced) {
                ept = bigger_exp_scd;
                local_thread_id = thread_id;
                s8_packed_gemm_bf16_dq_dynamic_n_slice_thread_task(
                    local_thread_id, ept.m, N, K, ept.t, n_slice, N - n_slice, tmpx + ept.bias * K,
                    weights + ept.ei * N * K, acts_scale + ept.bias, weights_scale + ept.ei * N,
                    output + ept.bias * N, reinterpret_cast<bfloat16_t *>(tmpy + 2 * ept.bias * N), ept.num_threads
                );
            }
        } else {
            kutacc::parallel_barrier();
        }
    });
}

void fusedmoe_down_buffer_expansion(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int8_t *acts,
    int8_t *weights, float *acts_scale, float *weights_scale, int *experts_offset, bfloat16_t *output, int8_t *tmpx,
    float *tmpy, MatrixTilingBlock t)
{
    int64_t tile_n = std::get<1>(t);
    int64_t tile_k = std::get<2>(t);
    int64_t num_threads = kutacc::get_thread_num();
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        for (int64_t i = 0; i < num_experts; i++) {
            int64_t thread_id = kutacc::get_thread_id();
            auto m = experts_offset[i + 1] - experts_offset[i];
            auto bias = experts_offset[i] - experts_offset[0];
            gemm_pack_thread_task<int8_t, false>(
                thread_id, m, K, m, tile_k, acts + bias * K, tmpx + bias * K, nullptr, K, num_threads);
        }
        kutacc::parallel_barrier();
        for (int64_t i = 0; i < num_experts; i++) {
            int64_t thread_id = kutacc::get_thread_id();
            auto m = experts_offset[i + 1] - experts_offset[i];
            auto bias = experts_offset[i] - experts_offset[0];
            s8_packed_gemm_bf16_dq_dynamic_tile_n_thread_task(
                thread_id, m, N, K, MatrixTilingBlock(m, tile_n, tile_k), tmpx + bias * K, weights + i * N * K,
                acts_scale + bias, weights_scale + i * N, output + bias * N,
                reinterpret_cast<bfloat16_t *>(tmpy + bias * N / 2), num_threads);
        }
    });
}

void fusedmoe_down_buffer_limited(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int8_t *acts,
    int8_t *weights, float *acts_scale, float *weights_scale, int *experts_offset, bfloat16_t *output,
    int8_t *tmpx, float *tmpy, MatrixTilingBlock t, int64_t tilebuf_size)
{
    for (int64_t i = 0; i < num_experts; i++) {
        int64_t bs = experts_offset[i + 1] - experts_offset[i];
        int64_t bs_offset = experts_offset[i] - experts_offset[0];
        if (bs == 0) {
            continue;
        }
        int64_t step = 0;
        while (step < bs) {
            bool sliced = false;
            int64_t now_bs = std::min(tilebuf_size, bs - step);
            int64_t m = now_bs;
            int64_t tile_n = std::get<1>(t);
            int64_t tile_k = std::get<2>(t);
            // 1. Pack act
            s8_gemm_pack_fusedmoe(m, K, tile_k, acts + (bs_offset + step) * K, tmpx);

            // 2. Call IGEMMBDQ kernel
            s8_packed_gemm_bf16_dq_dynamic_tile_n(m, N, K, MatrixTilingBlock(m, tile_n, tile_k), tmpx,
                weights + i * N * K, acts_scale + bs_offset + step, weights_scale + i * N,
                output + (bs_offset + step) * N, reinterpret_cast<bfloat16_t *>(tmpy));

            step += m;
        }
    }
}

void fusedmoe_down(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int8_t *acts, int8_t *weights,
    float *acts_scale, float *weights_scale, int *experts_offset, bfloat16_t *output, int8_t *tmpx, float *tmpy,
    MatrixTilingBlock t, int64_t tilebuf_size, std::optional<int64_t> n_slice)
{
    if (total_bs <= tilebuf_size) {
        if (num_experts == 2 && n_slice.has_value()) {
            fusedmoe_down_dualexpt_parallel(total_bs, K, N, num_experts, acts, weights, acts_scale, weights_scale,
                experts_offset, output, tmpx, tmpy, t, *n_slice);
            return;
        }
        fusedmoe_down_buffer_expansion(total_bs, K, N, num_experts, acts, weights, acts_scale, weights_scale,
            experts_offset, output, tmpx, tmpy, t);
    } else {
        fusedmoe_down_buffer_limited(total_bs, K, N, num_experts, acts, weights, acts_scale, weights_scale,
            experts_offset, output, tmpx, tmpy, t, tilebuf_size);
    }
}
}