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

#define CHECK_SHAPE(x, ...) FLASH_ASSERT(get_tensor_sizes(x) == std::vector<int>({__VA_ARGS__}))

namespace {

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

int64_t extra_buffer_size;

kutacc::FlashMLAMetaHandle meta;

} // namespace

void flash_mla_init()
{
    kutacc::flash_mla_meta_create(meta);
}

void flash_mla_finalize()
{
    kutacc::flash_mla_meta_destory(meta);
}

void flash_mla_get_metadata(const Tensor &seqlens_k,
                            int64_t seqlen_q,
                            int64_t num_heads_q,
                            int64_t head_size,
                            int64_t head_size_v,
                            int64_t page_block_size,
                            bool is_kv_packed)
{
    kutacc::flash_mla_dense_decode_sched(
        seqlens_k.to_kutacc_tensor<int, 1>(),
        seqlen_q,
        num_heads_q,
        head_size,
        head_size_v,
        page_block_size,
        is_kv_packed,
        extra_buffer_size,
        meta);
}

static PersistentBuffer softmax_lse_pbuf, extra_pbuf;

void flash_mla_with_kvcache(Tensor q,
                            const Tensor &kcache,
                            const Tensor &vcache,
                            const Tensor &block_table,
                            const Tensor &seqlens_k,
                            const double &softmax_scale,
                            bool is_causal,
                            Tensor &out)
{
    char *temp_buffer = (char *)softmax_lse_pbuf.malloc_buffer(q.sizes[0] * q.sizes[1] * q.sizes[2] * sizeof(float));
    Tensor softmax_lse = make_tensor("float32", {q.sizes[0], q.sizes[1], q.sizes[2]}, temp_buffer);
    extra_pbuf.malloc_buffer(extra_buffer_size);

    kutacc::flash_mla_dense_decode(
        q.to_kutacc_tensor<bfloat16_t, 4>(),
        kcache.to_kutacc_tensor<bfloat16_t, 3>(),
        vcache.to_optional_kutacc_tensor<bfloat16_t, 3>(),
        block_table.to_kutacc_tensor<int, 2>(),
        seqlens_k.to_kutacc_tensor<int, 1>(),
        out.to_kutacc_tensor<bfloat16_t, 4>(),
        softmax_lse.to_kutacc_tensor<float, 3>(),
        softmax_scale,
        is_causal,
        extra_pbuf.malloc_buffer(extra_buffer_size),
        meta);
}

void naive_mla_with_kvcache(const Tensor &q,
                            const Tensor &kvcache,
                            const Tensor &block_table,
                            const Tensor &seqlens_k,
                            const int &head_size_v,
                            const double &softmax_scale,
                            bool is_causal,
                            Tensor &out)
{
    const int batch_size = q.sizes[0];
    const int seqlen_q_ori = q.sizes[1];
    const int num_heads_ori = q.sizes[2];
    const int head_size = q.sizes[3];
    const int max_num_blocks_per_seq = block_table.sizes[1];
    const int num_blocks = kvcache.sizes[0];
    const int page_block_size = kvcache.sizes[1];

    FLASH_ASSERT(kvcache.dtype() == q.dtype());
    FLASH_ASSERT(block_table.dtype() == ScalarType::kInt);
    FLASH_ASSERT(seqlens_k.dtype() == ScalarType::kInt);
    FLASH_ASSERT(q.stride(-1) == 1);
    FLASH_ASSERT(kvcache.stride(-1) == 1);
    FLASH_ASSERT(block_table.stride(-1) == 1);
    FLASH_ASSERT(out.stride(-1) == 1);
    CHECK_CONTIGUOUS(seqlens_k);
    CHECK_SHAPE(q, batch_size, seqlen_q_ori, num_heads_ori, head_size);
    CHECK_SHAPE(kvcache, num_blocks, page_block_size, head_size);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    CHECK_SHAPE(seqlens_k, batch_size);
    CHECK_SHAPE(out, batch_size, seqlen_q_ori, num_heads_ori, head_size_v);

    FLASH_ASSERT(q.dtype() == ScalarType::kBFloat16);

    int num_heads = seqlen_q_ori * num_heads_ori;
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
                            float a = q_ptr[i * head_size + k];
                            float b = kv_ptr[j * kvcache.strides[1] + k];
                            s_ptr[i * seqlen + n_block * page_block_size + j] += a * b;
                        }
                    }
                }
            }
            for (int i = 0; i < num_heads; ++i) {
                float mx = s_ptr[i * seqlen];
                int len = is_causal ? seqlen - (seqlen_q_ori - 1 - i / num_heads_ori) : seqlen;
                for (int j = 0; j < len; ++j) {
                    mx = std::max(mx, s_ptr[i * seqlen + j]);
                }
                float sum = 0;
                for (int j = 0; j < len; ++j) {
                    sum += exp((s_ptr[i * seqlen + j] - mx) * softmax_scale);
                }
                for (int j = 0; j < len; ++j) {
                    s_ptr[i * seqlen + j] = exp((s_ptr[i * seqlen + j] - mx) * softmax_scale) / sum;
                }
                for (int j = len; j < seqlen; ++j) {
                    s_ptr[i * seqlen + j] = 0;
                }
            }
            for (int i = 0; i < num_heads; ++i) {
                for (int j = 0; j < head_size_v; ++j) {
                    float res_o = 0;
                    for (int n_block = 0; n_block < ceil_div(seqlen, page_block_size); ++n_block) {
                        bfloat16_t *kv_ptr =
                            kvcache.data_ptr<bfloat16_t>() + block_table_ptr[n_block] * kvcache.strides[0];
                        int cur_len = std::min(seqlen - n_block * page_block_size, page_block_size);
                        for (int k = 0; k < cur_len; ++k) {
                            float a = s_ptr[i * seqlen + n_block * page_block_size + k];
                            float b = kv_ptr[k * kvcache.strides[1] + j];
                            o_ptr[i * head_size_v + j] += a * b;
                            res_o += a * b;
                        }
                    }
                    o_ptr[i * head_size_v + j] = res_o;
                }
            }
        }
    });
}

void mla_pack_kvcache(const Tensor &kvcache,
                      const Tensor &packed_kcache,
                      const Tensor &packed_vcache)
{
    int num_blocks = kvcache.sizes[0];
    kutacc::parallel_for(0, num_blocks, 1, [&](int64_t start, int64_t end) {
        for (int block_id = start; block_id < end; ++block_id) {
            bfloat16_t *kv = kvcache.data_ptr<bfloat16_t>() + block_id * kvcache.strides[0];
            bfloat16_t *packed_k = packed_kcache.data_ptr<bfloat16_t>() + block_id * packed_kcache.strides[0];
            bfloat16_t *packed_v = packed_vcache.data_ptr<bfloat16_t>() + block_id * packed_vcache.strides[0];
            int block_size = kvcache.sizes[1];
            int head_size = kvcache.sizes[2];
            int head_size_v = packed_vcache.sizes[2];
            for (int i0 = 0; i0 < block_size; i0 += 16) {
                for (int j = 0; j < head_size; j += 2) {
                    for (int i = 0; i < 16; ++i) {
                        packed_k[i0 * head_size + j * 16 + i * 2] = kv[(i0 + i) * head_size + j];
                        packed_k[i0 * head_size + j * 16 + i * 2 + 1] = kv[(i0 + i) * head_size + j + 1];
                    }
                }
            }
            for (int i0 = 0; i0 < head_size_v; i0 += 16) {
                for (int j = 0; j < block_size; j += 2) {
                    for (int i = 0; i < 16; ++i) {
                        packed_v[i0 * block_size + j * 16 + i * 2] = kv[j * head_size + i0 + i];
                        packed_v[i0 * block_size + j * 16 + i * 2 + 1] = kv[(j + 1) * head_size + i0 + i];
                    }
                }
            }
        }
    });
}