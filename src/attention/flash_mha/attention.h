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

#include <cstdlib>
#include <optional>
#include <arm_sve.h>
#include <unistd.h>
#include "gemm.h"
#include "pack.h"
#include "utils/bf16.h"
#include "math/fast_exp.h"
#include "tensor/tensor.h"
#include "kutacc.h"

namespace kutacc {
void varlen_attention(
    const Tensor<bfloat16_t, 3> &(q),
    const Tensor<bfloat16_t, 3> &(k),
    const Tensor<bfloat16_t, 3> &(v),
    const Tensor<bfloat16_t, 3> &(out),
    bool causal,
    double softmax_scale,
    const Tensor<int, 1> &(query_start_loc),
    const Tensor<int, 1> &(key_start_loc)
);

void flash_attention(
    const Tensor<bfloat16_t, 3> &(q),
    const Tensor<bfloat16_t, 3> &(k),
    const Tensor<bfloat16_t, 3> &(v),
    const Tensor<bfloat16_t, 3> &(out),
    const Tensor<bfloat16_t, 2> &(pack_attn_q),
    const Tensor<bfloat16_t, 3> &(pack_attn_k),
    const Tensor<bfloat16_t, 3> &(pack_attn_v),
    const Tensor<float, 2> &(attn_s),
    const Tensor<float, 3> &(attn_out_block_old),
    const Tensor<float, 3> &(attn_out_block_new),
    const Tensor<float, 2> &(attn_max_block_old),
    const Tensor<float, 2> &(attn_max_block_new),
    const Tensor<float, 2> &(attn_base_block_old),
    const Tensor<float, 2> &(attn_base_block_new),
    bool causal,
    double softmax_scale,
    const Tensor<int, 1> &(query_start_loc),
    const Tensor<int, 1> &(key_start_loc),
    int chunked_prefill_size, std::vector<int64_t>& seq_lens, std::vector<int64_t>& cur_lens, bool is_kv_packed
);
}  // namespace attn
