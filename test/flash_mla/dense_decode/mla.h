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

#include "Tensor.h"

void flash_mla_init();
void flash_mla_finalize();

void flash_mla_get_metadata(
    const Tensor &seqlens_k,
    int64_t seqlen_q,
    int64_t num_heads_q,
    int64_t head_size,
    int64_t head_size_v,
    int64_t page_block_size,
    bool is_kv_packed);

void flash_mla_with_kvcache(
    Tensor q,                                   // batch_size x num_tokens x num_heads x head_size
    const Tensor &kcache,                       // num_blocks x page_block_size x head_size
    const Tensor &vcache,                       // num_blocks x page_block_size x head_size_v
    const Tensor &block_table,                  // batch_size x max_num_blocks_per_seq
    const Tensor &seqlens_k,                    // batch_size
    const double &softmax_scale,
    bool is_causal,
    Tensor &out);                               // batch_size x num_tokens x num_heads x head_size_v

void naive_mla_with_kvcache(
    const Tensor &q,                            // batch_size x num_heads x head_size
    const Tensor &kvcache,                      // num_blocks x page_block_size x head_size
    const Tensor &block_table,                  // batch_size x max_num_blocks_per_seq
    const Tensor &seqlens_k,                    // batch_size
    const int &head_size_v,
    const double &softmax_scale,
    bool is_causal,
    Tensor &out);                               // batch_size x num_heads x head_size_v

void mla_pack_kvcache(
    const Tensor &kvcache,
    const Tensor &packed_kcache,
    const Tensor &packed_vcache);