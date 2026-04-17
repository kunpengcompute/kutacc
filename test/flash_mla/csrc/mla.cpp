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
#include <cmath>

#include "kutacc.h"
#include "utils.h"
#include "mla.h"

namespace {

#define CHECK_SHAPE(x, ...) FLASH_ASSERT(get_tensor_sizes(x) == std::vector<int>({__VA_ARGS__}))

inline std::vector<int> get_tensor_sizes(const Tensor &a)
{
    std::vector<int> sizes(a.dim);
    for (int i = 0; i < a.dim; ++i) {
        sizes[i] = a.sizes[i];
    }
    return sizes;
}

template <typename T>
constexpr T ceil_div(const T &x, const T &y)
{
    return (x + y - 1) / y;
}

int extra_buffer_size_per_thread;
int tile_scheduler_metadata_size;
int block_size_m;
int block_size_n;

} // namespace

void flash_mla_get_metadata(const Tensor &seqlens_k,
                            const int64_t &num_heads_per_head_k,
                            const int64_t &num_heads_k,
                            Tensor &tile_scheduler_metadata,
                            Tensor &num_splits,
                            char *&mem_ptr)
{
    // This should match the logic in the MLA kernel.

    static constexpr int fixed_overhead_num_blocks = 5;

    CHECK_CONTIGUOUS(seqlens_k);
    FLASH_ASSERT(seqlens_k.dtype() == ScalarType::kInt);

    kutacc::flash_mla_get_constant(
        extra_buffer_size_per_thread, tile_scheduler_metadata_size, block_size_m, block_size_n);

    int batch_size = seqlens_k.size(0);
    int *seqlens_k_ptr = seqlens_k.data_ptr<int>();

    int threads_count = kutacc::get_thread_num();
    int num_thread_parts = threads_count / num_heads_k / ceil_div(num_heads_per_head_k, (int64_t)block_size_m);
    if (!num_thread_parts) {
        num_thread_parts = 1;
    }

    if (tile_scheduler_metadata.ptr == nullptr) {
        tile_scheduler_metadata = Tensor::inplace_create(
            ScalarType::kInt, {num_thread_parts, tile_scheduler_metadata_size}, mem_ptr);
    }
    if (num_splits.ptr == nullptr) {
        num_splits = Tensor::inplace_create(ScalarType::kInt, {batch_size + 1}, mem_ptr);
    }

    int *tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    int *num_splits_ptr = num_splits.data_ptr<int>();

    kutacc::FlashMLAMetadataParams params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_thread_parts = num_thread_parts;

    kutacc::flash_mla_get_metadata(params);
}

static PersistentBuffer softmax_lseaccum_pbuf, oaccum_pbuf, softmax_lse_pbuf;

void flash_mla_with_kvcache(Tensor q,
                            const Tensor &kvcache,
                            const Tensor &block_table,
                            const Tensor &seqlens_k,
                            const int &head_size_v,
                            const Tensor &tile_scheduler_metadata,
                            const Tensor &num_splits,
                            const double &softmax_scale,
                            bool is_causal,
                            Tensor &out)
{
    auto q_dtype = q.dtype();

#ifdef ENABLE_CHECK
    FLASH_ASSERT(kvcache.dtype() == q_dtype);
    FLASH_ASSERT(q.stride(-1) == 1);
    FLASH_ASSERT(kvcache.stride(-1) == 1);
    FLASH_ASSERT(block_table.dtype() == ScalarType::kInt);
    FLASH_ASSERT(block_table.stride(-1) == 1);
#endif

    const int batch_size = q.sizes[0];
    const int seqlen_q_ori = 1;
    const int num_heads_ori = q.sizes[1];
    const int head_size = q.sizes[2];

#ifdef ENABLE_CHECK
    FLASH_ASSERT(head_size == 576);
    FLASH_ASSERT(head_size_v == 512);
#endif

    const int max_num_blocks_per_seq = block_table.sizes[1];
    const int num_blocks = kvcache.sizes[0];
    const int page_block_size = kvcache.sizes[1];
    const int num_heads_k = 1;

#ifdef ENABLE_CHECK
    FLASH_ASSERT(batch_size > 0);
    FLASH_ASSERT(num_heads_k == 1);
#endif

    if (seqlen_q_ori == 1) {
        is_causal = false;
    }

    const int ngroups = num_heads_ori / num_heads_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;

    q = Tensor::view(q, {batch_size, seqlen_q, num_heads, head_size});

    int head_size_k = head_size;

#ifdef ENABLE_CHECK
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(kvcache, num_blocks, page_block_size, head_size_k);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    FLASH_ASSERT(seqlens_k.dtype() == ScalarType::kInt);
    CHECK_CONTIGUOUS(seqlens_k);
    CHECK_SHAPE(seqlens_k, batch_size);
    CHECK_SHAPE(out, batch_size, num_heads_ori, head_size_v);
#endif

    out = Tensor::view(out, {batch_size, seqlen_q, num_heads, head_size_v});

    kutacc::FlashMLAFwdParams params = {};

    // Set the sizes.
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.cu_seqlens_k = seqlens_k.data_ptr<int>();
    params.h = num_heads;
    params.ngroups = ngroups;
    params.is_causal = is_causal;
    params.d = head_size;
    params.d_v = head_size_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr<bfloat16_t>();
    params.k_ptr = params.v_ptr = kvcache.data_ptr<bfloat16_t>();
    params.o_ptr = out.data_ptr<bfloat16_t>();
    params.softmax_lse_ptr = softmax_lse_pbuf.malloc_buffer(batch_size * num_heads * seqlen_q * sizeof(float));

    // All stride are in elements, not bytes.
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = kvcache.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = kvcache.stride(-2);
    params.o_row_stride = out.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = kvcache.stride(-2);
    params.o_head_stride = out.stride(-2);

    params.block_table = block_table.data_ptr<int>();
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = page_block_size;

#ifdef ENABLE_CHECK
    FLASH_ASSERT(tile_scheduler_metadata.dtype() == ScalarType::kInt);
    FLASH_ASSERT(tile_scheduler_metadata.size(1) == tile_scheduler_metadata_size);
    CHECK_CONTIGUOUS(tile_scheduler_metadata);
#endif

    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
    params.num_thread_parts = tile_scheduler_metadata.sizes[0];

#ifdef ENABLE_CHECK
    FLASH_ASSERT(num_splits.dtype() == ScalarType::kInt);
    CHECK_CONTIGUOUS(num_splits);
#endif

    params.num_splits_ptr = num_splits.data_ptr<int>();

    params.softmax_lseaccum_ptr = softmax_lseaccum_pbuf.malloc_buffer(
        (batch_size + params.num_thread_parts) * num_heads * seqlen_q * sizeof(float));
    params.oaccum_ptr = oaccum_pbuf.malloc_buffer(
        (batch_size + params.num_thread_parts) * num_heads * seqlen_q * head_size_v * sizeof(float));

    if (q_dtype == ScalarType::kBFloat16) {
        kutacc::flash_mla_run_fwd<bfloat16_t, 576>(params);
    } else {
        FLASH_ASSERT(false);
    }

    out = Tensor::view(out, {batch_size, num_heads_ori, head_size_v});
}

void naive_mla_with_kvcache(const Tensor &q,
                            const Tensor &kvcache,
                            const Tensor &block_table,
                            const Tensor &seqlens_k,
                            const int &head_size_v,
                            const double &softmax_scale,
                            Tensor &out)
{
    const int batch_size = q.sizes[0];
    const int num_heads = q.sizes[1];
    const int head_size = q.sizes[2];
    const int max_num_blocks_per_seq = block_table.sizes[1];
    const int num_blocks = kvcache.sizes[0];
    const int page_block_size = kvcache.sizes[1];

#ifdef ENABLE_CHECK
    FLASH_ASSERT(kvcache.dtype() == q.dtype());
    FLASH_ASSERT(block_table.dtype() == ScalarType::kInt);
    FLASH_ASSERT(seqlens_k.dtype() == ScalarType::kInt);
    FLASH_ASSERT(q.stride(-1) == 1);
    FLASH_ASSERT(kvcache.stride(-1) == 1);
    FLASH_ASSERT(block_table.stride(-1) == 1);
    FLASH_ASSERT(out.stride(-1) == 1);
    CHECK_CONTIGUOUS(seqlens_k);
    CHECK_SHAPE(q, batch_size, num_heads, head_size);
    CHECK_SHAPE(kvcache, num_blocks, page_block_size, head_size_k);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    CHECK_SHAPE(seqlens_k, batch_size);
    CHECK_SHAPE(out, batch_size, num_heads, head_size_v);
#endif
    FLASH_ASSERT(q.dtype() == ScalarType::kBFloat16);

    static PersistentBuffer extra_buffer;
    int tnum = kutacc::get_thread_num();
    int max_seqlen = 0;
    for (int i = 0; i < batch_size; ++i) {
        max_seqlen = std::max(max_seqlen, seqlens_k.data_ptr<int>()[i]);
    }
    extra_buffer.malloc_buffer(tnum * num_heads * max_seqlen * sizeof(float));

    kutacc::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
        int tid = kutacc::get_thread_id();
        for (int b = begin; b < end; ++b) {
            float *s_ptr = (float *)extra_buffer.g_buf + tid * num_heads * max_seqlen;
            bfloat16_t *q_ptr = q.data_ptr<bfloat16_t>() + b * q.strides[0];
            bfloat16_t *o_ptr = out.data_ptr<bfloat16_t>() + b * out.strides[0];
            int *block_table_ptr = block_table.data_ptr<int>() + b * block_table.strides[0];
            int seqlen = seqlens_k.data_ptr<int>()[b];
            for (int n_block = 0; n_block < ceil_div(seqlen, page_block_size); ++n_block) {
                bfloat16_t *kv_ptr = kvcache.data_ptr<bfloat16_t>() + block_table_ptr[n_block] * kvcache.strides[0];
                int cur_len = std::min(seqlen - n_block * page_block_size, page_block_size);
                for (int i = 0; i < num_heads; ++i) {
                    for (int j = 0; j < cur_len; ++j) {
                        s_ptr[i * seqlen + n_block * page_block_size + j] = 0;
                        for (int k = 0; k < head_size; ++k) {
                            float a = q_ptr[i * q.strides[1] + k];
                            float b = kv_ptr[j * kvcache.strides[1] + k];
                            s_ptr[i * seqlen + n_block * page_block_size + j] += a * b;
                        }
                    }
                }
            }
            for (int i = 0; i < num_heads; ++i) {
                float mx = s_ptr[i * seqlen];
                for (int j = 0; j < seqlen; ++j) {
                    mx = std::max(mx, s_ptr[i * seqlen + j]);
                }
                float sum = 0;
                for (int j = 0; j < seqlen; ++j) {
                    sum += exp((s_ptr[i * seqlen + j] - mx) * softmax_scale);
                }
                for (int j = 0; j < seqlen; ++j) {
                    s_ptr[i * seqlen + j] = exp((s_ptr[i * seqlen + j] - mx) * softmax_scale) / sum;
                }
            }
            for (int i = 0; i < num_heads; ++i) {
                for (int j = 0; j < head_size_v; ++j) {
                    float res_o = 0;
                    for (int n_block = 0; n_block < ceil_div(seqlen, page_block_size); ++n_block) {
                        bfloat16_t *kv_ptr = kvcache.data_ptr<bfloat16_t>() + block_table_ptr[n_block] * kvcache.strides[0];
                        int cur_len = std::min(seqlen - n_block * page_block_size, page_block_size);
                        for (int k = 0; k < cur_len; ++k) {
                            float a = s_ptr[i * seqlen + n_block * page_block_size + k];
                            float b = kv_ptr[k * kvcache.strides[1] + j];
                            o_ptr[i * out.strides[1] + j] += a * b;
                            res_o += a * b;
                        }
                    }
                    o_ptr[i * out.strides[1] + j] = res_o;
                }
            }
        }
    });
}