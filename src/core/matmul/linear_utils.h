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
#pragma once
#include <tuple>
#include <unordered_map>
#include <iostream>
#include <arm_sve.h>
#include <arm_sme.h>
#include <arm_neon.h>

#include "kernel/kernel.h"

namespace kutacc {
inline int largest_power(uint32_t N)
{
    N = N | (N>>1);
    N = N | (N>>2);
    N = N | (N>>4);
    N = N | (N>>8);
    N = N | (N>>16);
    return (N + 1) >> 1;
}
 
inline void get_cur_m_stride(int64_t m, int64_t n, int64_t tile_n, int64_t &cur_m,
    int64_t &cur_tile_m, int64_t &cur_tile_n)
{
    if (m < 2 * 128) {
        cur_m = cur_tile_m = m, cur_tile_n = tile_n;
        return;
    }
    int m_parts = largest_power(uint32_t(m / 128));
    int n_parts = n / tile_n;
    int m_part_per_tile = (m_parts > n_parts) ? m_parts / n_parts : 1;
    cur_m = m_parts * 128;
    cur_tile_m = m_part_per_tile * 128;
    cur_tile_n = tile_n * std::min(n_parts, m_parts);
}

inline std::tuple<int64_t, int64_t, int64_t> get_idxs(int64_t tid, int64_t bm, int64_t bn, int64_t bk)
{
    int64_t idk = tid % bk;
    tid /= bk;
    int64_t idm = tid % bm;
    tid /= bm;
    int64_t idn = tid % bn;
    tid /= bn;
    return std::tuple(idm, idn, idk);
}


struct ExpPara {
    int64_t ei;
    int64_t m, bias;
    int64_t num_threads;
    MatrixTilingBlock t;
};

inline int64_t get_tile_n(int64_t num_threads, int64_t M, int64_t N, int64_t K, const MatrixTilingBlock& t)
{
    // tile_n 要同时首先要接近 N 的均分， 其次要 16 整除以保证切分在计算中填满 ZA 寄存器的 tile
    int64_t blocks_in_m = M / M;
    int64_t blocks_in_k = K / std::get<2>(t);
    int64_t blocks_in_n = num_threads / (blocks_in_k * blocks_in_m);
    int64_t tile_n = (N + blocks_in_n - 1) / blocks_in_n;
    tile_n = (tile_n + 15) / 16 * 16;
    return tile_n;
}

inline ExpPara get_exp_para(
    int64_t expert_index, int *experts_offset, int64_t K, int64_t N, int64_t num_threads, const MatrixTilingBlock& t)
{
    int64_t m = experts_offset[expert_index + 1] - experts_offset[expert_index];
    int64_t tile_m = m;        // tile_m == m
    int64_t tile_k = std::get<2>(t);     // tile_k 在 ntk pack方法下不可以动态调整，在 tntk下可以动态调整
    int64_t tile_n = get_tile_n(num_threads, m, N, K, t);
    return ExpPara{
        .ei = expert_index,
        .m = m,
        .bias = experts_offset[expert_index] - experts_offset[0],
        .num_threads = num_threads,
        .t = MatrixTilingBlock(tile_m, tile_n, tile_k)
    };
}

inline std::tuple<ExpPara&, ExpPara&> reorder_exppara(ExpPara &exp0, ExpPara &exp1)
{
    if (exp0.m < exp1.m) {
        return std::tie(exp1, exp0);
    } else {
        return std::tie(exp0, exp1);
    }
}

inline std::tuple<ExpPara, ExpPara> reset_task_tile(int64_t num_threads, int64_t num_threads0, ExpPara &exp,
    int64_t N, int64_t K, const MatrixTilingBlock& t, int64_t n_slice)
{
    if (n_slice == N) {
        return {ExpPara(exp), ExpPara()};
    }
    int64_t tile_n_fst = get_tile_n(num_threads0, exp.m, n_slice, K, t);
    int64_t tile_n_scd = get_tile_n(num_threads, exp.m, N - n_slice, K, t);
    ExpPara exp0 = exp;
    ExpPara exp1 = exp;
    exp0.t = MatrixTilingBlock(std::get<0>(exp0.t), tile_n_fst, std::get<2>(exp0.t));
    exp1.t = MatrixTilingBlock(std::get<0>(exp0.t), tile_n_scd, std::get<2>(exp0.t));
    exp1.num_threads = num_threads;
    return std::tie(exp0, exp1);
}

}