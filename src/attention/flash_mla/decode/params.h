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

namespace kutacc {

namespace flash_mla {

struct DenseDecodeParams {
    int64_t batch_size;
    int64_t seqlen_q;
    int64_t num_heads_q;

    int64_t casual_group_size;
    bool is_casual;

    float softmax_scale;
    float softmax_scale_log2;

    const Tensor<bfloat16_t, 4> *q;                            // [batch_size, seqlen_q, num_heads_q, head_dim]

    const Tensor<bfloat16_t, 3> *kcache;                       // [num_blocks, page_block_size, head_dim]
    const std::optional<Tensor<bfloat16_t, 3>> *vcache;        // [num_blocks, page_block_size, head_dim_v]
    const Tensor<int, 2> *block_table;                         // [batch_size, max_num_blocks_per_seq]
    const Tensor<int, 1> *seqlens_kv;                          // [batch_size]

    const Tensor<bfloat16_t, 4> *o;                            // [batch_size, seqlen_q, num_heads_q, head_dim_v]
    const Tensor<float, 3> *softmax_lse;                       // [batch_size, seqlen_q, num_heads_q]

    Tensor<float, 3> softmax_lseaccum;                         // [total_num_splits, seqlen_q, num_heads_q]
    Tensor<float, 4> oaccum;                                   // [total_num_splits, seqlen_q, num_heads_q, head_dim_v]

    void *threads_buffer;

    FlashMLAMetaHandle meta;
};

struct SparseDecodeParams {
    int64_t batch_size;
    int64_t seqlen_q;
    int64_t num_heads_q;

    int64_t topk;

    int64_t extra_topk;

    float softmax_scale;
    float softmax_scale_log2;

    const Tensor<bfloat16_t, 4> *q;                            // [batch_size, seqlen_q, num_heads_q, head_dim]

    Tensor<bfloat16_t, 2> kvcache;                             // [num_blocks * page_block_size, head_dim]
    const Tensor<int, 3> *indices;                             // [batch_size, seqlen_q, topk]
    const std::optional<Tensor<int, 1>> *topk_length;          // [batch_size]

    std::optional<Tensor<bfloat16_t, 2>> extra_kvcache;        // [extra_num_blocks * extra_page_block_size, head_dim]
    const std::optional<Tensor<int, 3>> *extra_indices;        // [batch_size, seqlen_q, extra_topk]
    const std::optional<Tensor<int, 1>> *extra_topk_length;    // [batch_size]

    const std::optional<Tensor<float, 1>> *attn_sink;          // [num_heads_q]

    const Tensor<bfloat16_t, 4> *o;                            // [batch_size, seqlen_q, num_heads_q, head_dim_v]
    const Tensor<float, 3> *softmax_lse;                       // [batch_size, seqlen_q, num_heads_q]

    Tensor<float, 3> softmax_lseaccum;                         // [total_num_splits, seqlen_q, num_heads_q]
    Tensor<float, 4> oaccum;                                   // [total_num_splits, seqlen_q, num_heads_q, head_dim_v]

    void *threads_buffer;

    FlashMLAMetaHandle meta;
};

} // namespace flash_mla

} // namespace kutacc