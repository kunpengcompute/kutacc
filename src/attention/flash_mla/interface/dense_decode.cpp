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

void flash_mla_dense_decode_sched(
    const Tensor<int, 1> &seqlens_kv_,              // [batch_size]
    int64_t seqlen_q,
    int64_t num_heads_q,
    int64_t head_dim,
    int64_t head_dim_v,
    int64_t page_block_size,
    bool is_kv_packed,
    int64_t &extra_bytes,
    FlashMLAMetaHandle meta
)
{
    auto batch_size = seqlens_kv_.size(0);

    FLASH_MLA_CHECK(batch_size > 0, "");
    FLASH_MLA_CHECK(seqlen_q > 0, "");
    FLASH_MLA_CHECK(num_heads_q > 0, "");
    FLASH_MLA_CHECK(head_dim == 576, "");
    FLASH_MLA_CHECK(head_dim_v == 512, "");
    FLASH_MLA_CHECK(page_block_size == 64, "");

    auto &kernel_meta = meta->kernel_meta;
    kernel_meta = {false, is_kv_packed, 64, 64, head_dim, head_dim_v, 3, flash_mla::MAX_THREAD_BUFFER_BYTES};

    std::vector<int64_t> seqlens_kv(batch_size);
    int64_t sum_seqlen = 0;
    for (int64_t i = 0; i < batch_size; ++i) {
        seqlens_kv[i] = seqlens_kv_.index({i});
        sum_seqlen += seqlens_kv[i];
    }

    auto num_threads = get_thread_num();
    if (!is_kv_packed) {
        int threshold = 1024 * (2048 - 64) / 32;
        if (batch_size * seqlen_q * num_heads_q / num_threads >= 12 * 128 / 32) {
            threshold = 1024 * (1536 - 64) / 32;
        }
        if (sum_seqlen * seqlen_q * num_heads_q / num_threads <= threshold) {
            kernel_meta.q_block_size = 32;
        }
    }

    constexpr auto align_up = [](int64_t size) {
        return flash_mla::ceil(size, flash_mla::ALIGNMENT);
    };
    meta->threads_buffer_bytes = align_up(kernel_meta.bytes_per_thread * num_threads);
    meta->softmax_lseaccum_bytes = align_up((batch_size + num_threads) * seqlen_q * num_heads_q * sizeof(float));
    meta->oaccum_bytes = align_up((batch_size + num_threads) * seqlen_q * num_heads_q * head_dim_v * sizeof(float));
    extra_bytes = meta->softmax_lseaccum_bytes + meta->oaccum_bytes + meta->threads_buffer_bytes;

    meta->num_thread_parts = num_threads / flash_mla::ceil_div(num_heads_q * seqlen_q, kernel_meta.q_block_size);
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

void flash_mla_dense_decode(
    const Tensor<bfloat16_t, 4> &q_ori,                     // [batch_size, seqlen_q_ori, num_heads_q_ori, head_dim]
    const Tensor<bfloat16_t, 3> &kcache,                    // [num_blocks, page_block_size, head_dim]
    const std::optional<Tensor<bfloat16_t, 3>> &vcache,     // [num_blocks, page_block_size, head_dim_v]
    const Tensor<int, 2> &block_table,                      // [batch_size, max_num_blocks_per_seq]
    const Tensor<int, 1> &seqlens_kv,                       // [batch_size]
    const Tensor<bfloat16_t, 4> &o_ori,                     // [batch_size, seqlen_q_ori, num_heads_q_ori, head_dim_v]
    const Tensor<float, 3> &softmax_lse_ori,                // [batch_size, seqlen_q_ori, num_heads_q_ori]
    float softmax_scale,
    bool is_causal,
    void *extra_buffer,
    FlashMLAMetaHandle meta
)
{
    const int64_t batch_size = q_ori.size(0);
    const int64_t seqlen_q_ori = q_ori.size(1);
    const int64_t num_heads_q_ori = q_ori.size(2);
    const int64_t head_dim = q_ori.size(3);
    const int64_t head_dim_v = o_ori.size(-1);
    
    if (seqlen_q_ori == 1) {
        is_causal = false;
    }

    const int64_t seqlen_q = 1;
    const int64_t num_heads_q = seqlen_q_ori * num_heads_q_ori;
    const int64_t casual_group_size = num_heads_q_ori;

    auto q = tensor_view<bfloat16_t, 4, 4>(q_ori, {batch_size, seqlen_q, num_heads_q, head_dim});
    auto o = tensor_view<bfloat16_t, 4, 4>(o_ori, {batch_size, seqlen_q, num_heads_q, head_dim_v});
    auto softmax_lse = tensor_view<float, 3, 3>(softmax_lse_ori, {batch_size, seqlen_q, num_heads_q});

    const bool is_kv_packed = meta->kernel_meta.is_kv_packed;

    const int64_t num_blocks = kcache.size(0);
    const int64_t page_block_size = kcache.size(1);

    const int64_t max_num_blocks_per_seq = block_table.size(1);

    FLASH_MLA_CHECK(page_block_size == meta->kernel_meta.kv_block_size, "");

    FLASH_MLA_CHECK_LAST_DIM_CONTIGUOUS(q);

    FLASH_MLA_CHECK_SHAPE(kcache, num_blocks, page_block_size, head_dim);
    FLASH_MLA_CHECK_CONTIGUOUS(kcache);

    if (is_kv_packed) {
        FLASH_MLA_CHECK_SHAPE(vcache.value(), num_blocks, page_block_size, head_dim_v);
        FLASH_MLA_CHECK_CONTIGUOUS(vcache.value());
    } else {
        FLASH_MLA_CHECK(!vcache.has_value(), "");
    }

    FLASH_MLA_CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    FLASH_MLA_CHECK_LAST_DIM_CONTIGUOUS(block_table);

    FLASH_MLA_CHECK_SHAPE(seqlens_kv, batch_size);
    FLASH_MLA_CHECK_CONTIGUOUS(seqlens_kv);

    FLASH_MLA_CHECK_SHAPE(o, batch_size, seqlen_q, num_heads_q, head_dim_v);
    FLASH_MLA_CHECK_LAST_DIM_CONTIGUOUS(o);

    FLASH_MLA_CHECK_SHAPE(softmax_lse, batch_size, seqlen_q, num_heads_q);

    FLASH_MLA_CHECK(meta->batch_size == batch_size, "");
    FLASH_MLA_CHECK(meta->kernel_meta.head_dim == head_dim, "");
    FLASH_MLA_CHECK(meta->kernel_meta.head_dim_v == head_dim_v, "");
    FLASH_MLA_CHECK(!meta->kernel_meta.is_sparse, "");
    FLASH_MLA_CHECK(meta->kernel_meta.is_kv_packed == is_kv_packed, "");

    flash_mla::DenseDecodeParams params;

    params.batch_size = batch_size;
    params.seqlen_q = seqlen_q;
    params.num_heads_q = num_heads_q;

    params.casual_group_size = casual_group_size;

    params.is_casual = is_causal;

    params.softmax_scale = softmax_scale;
    params.softmax_scale_log2 = softmax_scale * M_LOG2E;

    params.q = &q;
    
    params.kcache = &kcache;
    params.vcache = &vcache;
    params.block_table = &block_table;
    params.seqlens_kv = &seqlens_kv;

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

    flash_mla::dense_decode_splitkv_mla(params);
}

} // namespace kutacc