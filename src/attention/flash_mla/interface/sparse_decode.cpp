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
#include "kutacc.h"

#include "tensor/tensor.h"
#include "attention/flash_mla/meta.h"
#include "attention/flash_mla/utils.h"
#include "attention/flash_mla/decode/sched.h"
#include "attention/flash_mla/decode/params.h"
#include "attention/flash_mla/decode/splitkv_mla.h"

namespace kutacc {

void flash_mla_sparse_decode_sched(
    int64_t batch_size,
    int64_t seqlen_q,
    int64_t num_heads_q,
    int64_t head_dim,
    int64_t head_dim_v,
    int64_t topk,
    int64_t extra_topk,
    const std::optional<Tensor<int, 1>> &topk_length,                  // [batch_size]
    const std::optional<Tensor<int, 1>> &extra_topk_length,            // [batch_size]
    int64_t &extra_bytes,
    FlashMLAMetaHandle meta
)
{
    FLASH_MLA_CHECK(batch_size > 0, "");
    FLASH_MLA_CHECK(seqlen_q > 0, "");
    FLASH_MLA_CHECK(num_heads_q > 0, "");
    FLASH_MLA_CHECK(head_dim == 512 || head_dim == 576, "");
    FLASH_MLA_CHECK(head_dim_v == 512, "");

    FLASH_MLA_CHECK(topk > 0, "");
    if (topk_length.has_value()) {
        FLASH_MLA_CHECK_SHAPE(topk_length.value(), batch_size);
    }

    if (extra_topk > 0) {
        if (extra_topk_length.has_value()) {
            FLASH_MLA_CHECK_SHAPE(extra_topk_length.value(), batch_size);
        }
    } else {
        FLASH_MLA_CHECK(extra_topk == 0, "");
        FLASH_MLA_CHECK(!extra_topk_length.has_value(), "");
    }

    auto &kernel_meta = meta->kernel_meta;
    kernel_meta = {true, false, 64, 64, head_dim, head_dim_v, 3, flash_mla::MAX_THREAD_BUFFER_BYTES};

    std::vector<int64_t> seqlens_kv(batch_size);
    int64_t sum_seqlen = 0;
    for (int64_t i = 0; i < batch_size; ++i) {
        seqlens_kv[i] = topk_length.has_value() ? topk_length.value().index({i}) : topk;
        if (extra_topk > 0) {
            seqlens_kv[i] += extra_topk_length.has_value() ? extra_topk_length.value().index({i}) : extra_topk;
        }
        FLASH_MLA_CHECK(seqlens_kv[i] > 0, "");
        sum_seqlen += seqlens_kv[i];
    }

    auto num_threads = get_thread_num();
    int threshold = 1024 * (2048 - 64) / 32;
    if (batch_size * seqlen_q * num_heads_q / num_threads >= 12 * 128 / 32) {
        threshold = 1024 * (1536 - 64) / 32;
    }
    if (sum_seqlen * seqlen_q * num_heads_q / num_threads <= threshold) {
        kernel_meta.q_block_size = 32;
    }

    constexpr auto align_up = [](int64_t size) {
        return flash_mla::ceil(size, flash_mla::ALIGNMENT);
    };
    meta->threads_buffer_bytes = align_up(kernel_meta.bytes_per_thread * num_threads);
    meta->softmax_lseaccum_bytes = align_up((batch_size + num_threads) * seqlen_q * num_heads_q * sizeof(float));
    meta->oaccum_bytes = align_up((batch_size + num_threads) * seqlen_q * num_heads_q * head_dim_v * sizeof(float));
    extra_bytes = meta->softmax_lseaccum_bytes + meta->oaccum_bytes + meta->threads_buffer_bytes;

    meta->num_thread_parts = num_threads / seqlen_q / flash_mla::ceil_div(num_heads_q, kernel_meta.q_block_size);
    if (!meta->num_thread_parts) {
        meta->num_thread_parts = 1;
    }

    meta->batch_size = batch_size;

    if (batch_size > meta->max_batch_size) {
        if (meta->num_splits != nullptr) {
            free(meta->num_splits);
        }
        meta->max_batch_size = batch_size;
        meta->num_splits = (int *)aligned_alloc(flash_mla::ALIGNMENT, (meta->max_batch_size + 1) * sizeof(int));
    }
    FLASH_MLA_CHECK(meta->num_splits != nullptr, "");

    flash_mla::decode_sched(seqlens_kv, extra_bytes, meta);
}

void flash_mla_sparse_decode(
    const Tensor<bfloat16_t, 4> &q,                              // [batch_size, seqlen_q, num_heads_q, head_dim]
    const Tensor<bfloat16_t, 3> &kvcache,                        // [num_blocks, page_block_size, head_dim]
    const Tensor<int, 3> &indices,                               // [batch_size, seqlen_q, topk]
    const std::optional<Tensor<int, 1>> &topk_length,            // [batch_size]
    const std::optional<Tensor<bfloat16_t, 3>> &extra_kvcache,   // [extra_num_blocks, extra_page_block_size, head_dim]
    const std::optional<Tensor<int, 3>> &extra_indices,          // [batch_size, seqlen_q, extra_topk]
    const std::optional<Tensor<int, 1>> &extra_topk_length,      // [batch_size]
    const std::optional<Tensor<float, 1>> &attn_sink,            // [num_heads_q]
    const Tensor<bfloat16_t, 4> &o,                              // [batch_size, seqlen_q, num_heads_q, head_dim_v]
    const Tensor<float, 3> &softmax_lse,                         // [batch_size, seqlen_q, num_heads_q]
    float softmax_scale,
    void *extra_buffer,
    FlashMLAMetaHandle meta
)
{
    const int64_t batch_size = q.size(0);
    const int64_t seqlen_q = q.size(1);
    const int64_t num_heads_q = q.size(2);
    const int64_t head_dim = q.size(3);
    const int64_t head_dim_v = o.size(-1);

    const int64_t num_blocks = kvcache.size(0);
    const int64_t page_block_size = kvcache.size(1);
    const int64_t topk = indices.size(-1);

    int64_t extra_num_blocks = 0;
    int64_t extra_page_block_size = 0;
    int64_t extra_topk = 0;

    FLASH_MLA_CHECK_LAST_DIM_CONTIGUOUS(q);

    FLASH_MLA_CHECK_SHAPE(kvcache, num_blocks, page_block_size, head_dim);
    FLASH_MLA_CHECK_CONTIGUOUS(kvcache);

    FLASH_MLA_CHECK_SHAPE(indices, batch_size, seqlen_q, topk);
    FLASH_MLA_CHECK_LAST_DIM_CONTIGUOUS(indices);

    FLASH_MLA_CHECK_SHAPE(o, batch_size, seqlen_q, num_heads_q, head_dim_v);
    FLASH_MLA_CHECK_LAST_DIM_CONTIGUOUS(o);

    FLASH_MLA_CHECK_SHAPE(softmax_lse, batch_size, seqlen_q, num_heads_q);

    if (attn_sink.has_value()) {
        FLASH_MLA_CHECK_SHAPE(attn_sink.value(), num_heads_q);
        FLASH_MLA_CHECK_CONTIGUOUS(attn_sink.value());
    }

    FLASH_MLA_CHECK(extra_kvcache.has_value() == extra_indices.has_value(), "");
    if (extra_kvcache.has_value()) {
        extra_num_blocks = extra_kvcache.value().size(0);
        extra_page_block_size = extra_kvcache.value().size(1);
        extra_topk = extra_indices.value().size(2);

        FLASH_MLA_CHECK_SHAPE(extra_kvcache.value(), extra_num_blocks, extra_page_block_size, head_dim);
        FLASH_MLA_CHECK_CONTIGUOUS(extra_kvcache.value());

        FLASH_MLA_CHECK_SHAPE(extra_indices.value(), batch_size, seqlen_q, extra_topk);
        FLASH_MLA_CHECK_LAST_DIM_CONTIGUOUS(extra_indices.value());
    }

    FLASH_MLA_CHECK(topk > 0, "");
    if (topk_length.has_value()) {
        FLASH_MLA_CHECK_SHAPE(topk_length.value(), batch_size);
        FLASH_MLA_CHECK_CONTIGUOUS(topk_length.value());
    }

    if (extra_topk > 0) {
        if (extra_topk_length.has_value()) {
            FLASH_MLA_CHECK_SHAPE(extra_topk_length.value(), batch_size);
            FLASH_MLA_CHECK_CONTIGUOUS(extra_topk_length.value());
        }
    } else {
        FLASH_MLA_CHECK(extra_topk == 0, "");
        FLASH_MLA_CHECK(!extra_topk_length.has_value(), "");
    }

    FLASH_MLA_CHECK(meta->batch_size == batch_size, "");
    FLASH_MLA_CHECK(meta->kernel_meta.head_dim == head_dim, "");
    FLASH_MLA_CHECK(meta->kernel_meta.head_dim_v == head_dim_v, "");
    FLASH_MLA_CHECK(meta->kernel_meta.is_sparse, "");
    FLASH_MLA_CHECK(!meta->kernel_meta.is_kv_packed, "");

    flash_mla::SparseDecodeParams params;

    params.batch_size = batch_size;
    params.seqlen_q = seqlen_q;
    params.num_heads_q = num_heads_q;

    params.topk = topk;

    params.extra_topk = extra_topk;

    params.softmax_scale = softmax_scale;
    params.softmax_scale_log2 = softmax_scale * M_LOG2E;

    params.q = &q;
    
    params.kvcache = tensor_view<bfloat16_t, 3, 2>(kvcache, {num_blocks * page_block_size, head_dim});
    params.indices = &indices;
    params.topk_length = &topk_length;
    
    if (extra_kvcache.has_value()) {
        params.extra_kvcache = tensor_view<bfloat16_t, 3, 2>(
            extra_kvcache.value(), {extra_num_blocks * extra_page_block_size, head_dim});
    } else {
        params.extra_kvcache = std::nullopt;
    }

    params.extra_indices = &extra_indices;
    params.extra_topk_length = &extra_topk_length;

    params.attn_sink = &attn_sink;

    params.o = &o;
    params.softmax_lse = &softmax_lse;

    void *softmax_lseaccum_ptr = extra_buffer;
    void *oaccum_ptr = (int8_t *)softmax_lseaccum_ptr + meta->softmax_lseaccum_bytes;
    params.threads_buffer = (int8_t *)oaccum_ptr + meta->oaccum_bytes;

    const int64_t max_num_splits = batch_size + get_thread_num();
    params.softmax_lseaccum =
        Tensor<float, 3>((float *)softmax_lseaccum_ptr, {max_num_splits, seqlen_q, num_heads_q});
    params.oaccum =
        Tensor<float, 4>((float *)oaccum_ptr, {max_num_splits, seqlen_q, num_heads_q, head_dim_v});

    params.meta = meta;

    flash_mla::sparse_decode_splitkv_mla(params);
}

} // namespace kutacc