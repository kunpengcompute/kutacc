/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT license. See LICENSE in the
 * project root for license information.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef KUTACC_H
#define KUTACC_H

#include <map>
#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <arm_bf16.h>
#include <kupl.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief exposed KuTACC symbol table*/
#define kutacc_export __attribute__((visibility("default")))

/** @brief status in KuTACC library*/
#define KUTACC_OK 0
#define KUTACC_ERROR (-1)

/** @brief KuTACC version info*/
typedef struct kutacc_version {
    const char *product_name;
    const char *product_version;
    const char *component_name;
    const char *component_version;
    const char *component_appendinfo;
} kutacc_version_t;

/**
 * @brief get the kutacc version info
 * @param [out] version     the kutacc version info
 *
 * @return KUTACC_OK for get version info success, other for failed.
 */
kutacc_export int kutacc_get_version(kutacc_version_t *version);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace kutacc {

/** @brief KuTACC Tensor */
template <typename dtype, int64_t dim>
struct kutacc_export Tensor {
    static_assert(dim > 0);

public:
    Tensor() = default;

    Tensor(dtype *data_ptr, const int64_t *sizes, const int64_t *strides) : data_ptr_(data_ptr)
    {
        for (int i = 0; i < dim; ++i) {
            sizes_[i] = sizes[i];
            strides_[i] = strides[i];
        }
    }

    Tensor(dtype *data_ptr, const std::array<int64_t, dim> &sizes, const std::array<int64_t, dim> &strides)
        : data_ptr_(data_ptr), sizes_(sizes), strides_(strides)
    {
    }

    Tensor(dtype *data_ptr, const int64_t *sizes) : data_ptr_(data_ptr)
    {
        for (int64_t i = dim - 1; i >= 0; --i) {
            sizes_[i] = sizes[i];
            strides_[i] = i + 1 < dim ? sizes[i + 1] * strides_[i + 1] : 1;
        }
    }

    Tensor(dtype *data_ptr, const std::array<int64_t, dim> &sizes) : data_ptr_(data_ptr), sizes_(sizes)
    {
        for (int64_t i = dim - 1; i >= 0; --i) {
            strides_[i] = i + 1 < dim ? sizes[i + 1] * strides_[i + 1] : 1;
        }
    }

    int64_t numel() const
    {
        int64_t n = 1;
        for (const auto &x : sizes_) {
            n *= x;
        }
        return n;
    }

    const std::array<int64_t, dim> &sizes() const
    {
        return sizes_;
    }

    const std::array<int64_t, dim> &strides() const
    {
        return strides_;
    }

    int64_t size(int64_t i) const
    {
        return sizes_[i < 0 ? dim + i : i];
    }

    int64_t stride(int64_t i) const
    {
        return strides_[i < 0 ? dim + i : i];
    }

    dtype *data_ptr() const
    {
        return data_ptr_;
    }

    void set_data_ptr(dtype *ptr)
    {
        data_ptr_ = ptr;
    }

    dtype &index(const std::array<int64_t, dim> &indices) const
    {
        int64_t offset = 0;
        for (int64_t i = 0; i < dim; ++i) {
            offset += indices[i] * strides_[i];
        }
        return data_ptr_[offset];
    }

    bool is_contiguous() const
    {
        for (int i = 0; i < dim - 1; ++i) {
            if (strides_[i] != strides_[i + 1] * sizes_[i + 1]) {
                return false;
            }
        }
        return strides_[dim - 1] == 1;
    }
private:
    dtype *data_ptr_;
    std::array<int64_t, dim> sizes_;
    std::array<int64_t, dim> strides_;
};

/**
 * @brief Opaque handle for FlashMLA metadata.
 *
 * This structure holds scheduling information, kernel parameters, and temporary
 * buffer sizes for FlashMLA dense and sparse decode kernels.
 */
typedef struct FlashMLAMeta *FlashMLAMetaHandle;

/**
 * @brief Create a FlashMLA metadata handle.
 *
 * Allocates and initializes a FlashMLAMeta object, including per-thread scheduling
 * metadata. Must be called before using any FlashMLA decode functions.
 *
 * @param [out] meta  Handle to the newly allocated FlashMLAMeta.
 */
kutacc_export void flash_mla_meta_create(FlashMLAMetaHandle &meta);

/**
 * @brief Destroy a FlashMLA metadata handle.
 *
 * Frees all resources associated with the metadata handle.
 *
 * @param [in] meta  Handle to the FlashMLAMeta to destroy.
 */
kutacc_export void flash_mla_meta_destory(FlashMLAMetaHandle meta);

/**
 * @brief Schedule the dense MLA decode operation and compute required extra buffer size.
 *
 * This function determines tiling parameters, thread partitioning, and temporary
 * buffer requirements for a dense MLA decode kernel. It must be called before
 * `flash_mla_dense_decode` to allocate the `extra_buffer`.
 *
 * @param [in] seqlens_kv       1D tensor of sequence lengths for keys/values per batch (int64_t).
 * @param [in] seqlen_q         Query sequence length (number of query tokens).
 * @param [in] num_heads_q      Number of query heads.
 * @param [in] head_dim         Dimension of the query/key head (must be 576 for MLA).
 * @param [in] head_dim_v       Dimension of the value head (must be 512 for MLA).
 * @param [in] page_block_size  Page block size for paged KV cache (must be 64 for MLA).
 * @param [in] is_kv_packed     If true, the key/value caches are already packed.
 * @param [out] extra_bytes     Required size (in bytes) for the extra buffer (softmax_lse, oaccum, threads_buffer).
 * @param [in,out] meta         FlashMLA metadata handle (updated with scheduling decisions).
 */
kutacc_export void flash_mla_dense_decode_sched(const Tensor<int, 1> &seqlens_kv, int64_t seqlen_q, int64_t num_heads_q,
    int64_t head_dim, int64_t head_dim_v, int64_t page_block_size, bool is_kv_packed, int64_t &extra_bytes,
    FlashMLAMetaHandle meta);

/**
 * @brief Schedule the sparse MLA decode operation and compute required extra buffer size.
 *
 * This function determines tiling parameters and temporary buffer requirements
 * for a sparse MLA decode kernel (with top‑k selection). It must be called before
 * `flash_mla_sparse_decode` to allocate the `extra_buffer`.
 *
 * @param [in] batch_size          Number of sequences in the batch.
 * @param [in] seqlen_q            Query sequence length.
 * @param [in] num_heads_q         Number of query heads.
 * @param [in] head_dim            Query/key head dimension (512 or 576).
 * @param [in] head_dim_v          Value head dimension (must be 512).
 * @param [in] topk                Number of top‑k indices per query (dense part).
 * @param [in] extra_topk          Number of extra top‑k indices (sparse part, 0 if none).
 * @param [in] topk_length         Optional per‑batch actual lengths for dense top‑k (overrides topk).
 * @param [in] extra_topk_length   Optional per‑batch actual lengths for sparse top‑k.
 * @param [out] extra_bytes        Required size (in bytes) for the extra buffer.
 * @param [in,out] meta            FlashMLA metadata handle (updated with scheduling decisions).
 */
kutacc_export void flash_mla_sparse_decode_sched(int64_t batch_size, int64_t seqlen_q, int64_t num_heads_q,
    int64_t head_dim, int64_t head_dim_v, int64_t topk, int64_t extra_topk,
    const std::optional<Tensor<int, 1>> &topk_length, const std::optional<Tensor<int, 1>> &extra_topk_length,
    int64_t &extra_bytes, FlashMLAMetaHandle meta);

/**
 * @brief Perform dense MLA decode (forward pass) with paged KV cache.
 *
 * This kernel computes attention using the Multi‑head Latent Attention (MLA)
 * approach for dense (full) key/value lookup. It uses a split‑KV technique
 * and online softmax, and writes the output and log‑sum‑exp (LSE) tensors.
 *
 * @param [in] q               Query tensor (bfloat16) of shape
 *                             [batch_size, seqlen_q_ori, num_heads_q_ori, head_dim].
 * @param [in] kcache          Key cache tensor (bfloat16) of shape
 *                             [num_blocks, page_block_size, head_dim].
 * @param [in] vcache          Optional value cache tensor (bfloat16) of shape
 *                             [num_blocks, page_block_size, head_dim_v]; required only if is_kv_packed is true.
 * @param [in] block_table     Block table tensor (int) of shape
 *                             [batch_size, max_num_blocks_per_seq] mapping sequences to cache blocks.
 * @param [in] seqlens_kv      Actual key/value sequence lengths per batch (int).
 * @param [out] o              Output tensor (bfloat16) of shape
 *                             [batch_size, seqlen_q_ori, num_heads_q_ori, head_dim_v].
 * @param [out] softmax_lse    Log‑sum‑exp tensor (float) of shape
 *                             [batch_size, seqlen_q_ori, num_heads_q_ori] for online softmax.
 * @param [in] softmax_scale   Scaling factor for Q·K^T (usually 1/sqrt(head_dim)).
 * @param [in] is_causal       If true, apply causal masking.
 * @param [in] extra_buffer    Pre‑allocated buffer of size computed by `flash_mla_dense_decode_sched`.
 * @param [in] meta            FlashMLA metadata handle (prepared by sched).
 */
kutacc_export void flash_mla_dense_decode(const Tensor<bfloat16_t, 4> &q, const Tensor<bfloat16_t, 3> &kcache,
    const std::optional<Tensor<bfloat16_t, 3>> &vcache, const Tensor<int, 2> &block_table,
    const Tensor<int, 1> &seqlens_kv, const Tensor<bfloat16_t, 4> &o, const Tensor<float, 3> &softmax_lse,
    float softmax_scale, bool is_causal, void *extra_buffer, FlashMLAMetaHandle meta);

/**
 * @brief Perform sparse MLA decode with top‑k selection and optional extra KV cache.
 *
 * This kernel selects only a subset of key/value entries (top‑k) per query
 * to reduce computation. It supports an additional extra KV cache (e.g., for
 * sink tokens or special positions) and optionally applies an attention sink bias.
 *
 * @param [in] q                   Query tensor (bfloat16) of shape
 *                                 [batch_size, seqlen_q, num_heads_q, head_dim].
 * @param [in] kvcache             Key/value cache (bfloat16) of shape
 *                                 [num_blocks, page_block_size, head_dim] (contiguous in memory).
 * @param [in] indices             Dense top‑k indices (int) of shape
 *                                 [batch_size, seqlen_q, topk] indicating which cache entries to attend.
 * @param [in] topk_length         Optional per‑batch actual number of dense top‑k entries (int).
 * @param [in] extra_kvcache       Optional extra KV cache (bfloat16) of shape
 *                                 [extra_num_blocks, extra_page_block_size, head_dim].
 * @param [in] extra_indices       Optional extra top‑k indices (int) of shape
 *                                 [batch_size, seqlen_q, extra_topk].
 * @param [in] extra_topk_length   Optional per‑batch actual lengths for extra top‑k.
 * @param [in] attn_sink           Optional attention sink bias (float) per head, of shape [num_heads_q].
 * @param [out] o                  Output tensor (bfloat16) of shape
 *                                 [batch_size, seqlen_q, num_heads_q, head_dim_v].
 * @param [out] softmax_lse        Log‑sum‑exp tensor (float) of shape
 *                                 [batch_size, seqlen_q, num_heads_q].
 * @param [in] softmax_scale       Scaling factor for Q·K^T.
 * @param [in] extra_buffer        Pre‑allocated buffer of size computed by `flash_mla_sparse_decode_sched`.
 * @param [in] meta                FlashMLA metadata handle (prepared by sched).
 */
kutacc_export void flash_mla_sparse_decode(const Tensor<bfloat16_t, 4> &q, const Tensor<bfloat16_t, 3> &kvcache,
    const Tensor<int, 3> &indices, const std::optional<Tensor<int, 1>> &topk_length,
    const std::optional<Tensor<bfloat16_t, 3>> &extra_kvcache, const std::optional<Tensor<int, 3>> &extra_indices,
    const std::optional<Tensor<int, 1>> &extra_topk_length, const std::optional<Tensor<float, 1>> &attn_sink,
    const Tensor<bfloat16_t, 4> &o, const Tensor<float, 3> &softmax_lse, float softmax_scale, void *extra_buffer,
    FlashMLAMetaHandle meta);

/**
 * @brief Get the block size parameters used by the Flash Attention kernel.
 *
 * The returned tuple (br, bc) defines the number of rows (queries) and
 * columns (keys/values) per block in the blocked attention computation.
 *
 * @return std::tuple<int64_t, int64_t>  (br, bc) block sizes.
 */
kutacc_export std::tuple<int64_t, int64_t> get_flash_attention_block();

/**
 * @brief Pack the value tensor into a blocked layout for Flash Attention.
 *
 * The input value tensor of shape (kv_len, num_heads, vo_head_dim) is reorganized
 * into a blocked format that improves memory locality during the online softmax
 * and matrix multiplication stages. The output layout is specific to the Flash
 * Attention implementation.
 *
 * @param [in] kv_len           Number of key/value tokens.
 * @param [in] num_heads        Number of attention heads.
 * @param [in] vo_head_dim      Dimension of the value head (output feature size).
 * @param [in] output_len       Length of the output buffer (must be at least the
 *                              padded length of kv_len rounded up to bc blocks).
 * @param [in] input_stride0    Stride between different tokens in input (leading dimension).
 * @param [in] input_stride1    Stride between different heads in input (i.e., head dimension).
 * @param [in] input            Input value tensor (bfloat16) in shape (kv_len, num_heads, vo_head_dim).
 * @param [out] output          Output packed value tensor (bfloat16) of shape
 *                              (num_heads, padded_len, vo_head_dim) in blocked order.
 */
kutacc_export void flash_attention_v_block_pack(int64_t kv_len, int64_t num_heads, int64_t vo_head_dim,
    int64_t output_len, int64_t input_stride0, int64_t input_stride1, bfloat16_t *input, bfloat16_t *output);

/**
 * @brief Pack the key tensor into a blocked layout for Flash Attention.
 *
 * The input key tensor of shape (kv_len, num_heads, qk_head_dim) is rearranged into
 * a blocked format suitable for the Q·K^T matrix multiplication in Flash Attention.
 *
 * @param [in] kv_len           Number of key/value tokens.
 * @param [in] num_heads        Number of attention heads.
 * @param [in] qk_head_dim      Dimension of the query/key head (attention feature size).
 * @param [in] output_len       Length of the output buffer (must be at least the
 *                              padded length of kv_len rounded up to bc blocks).
 * @param [in] input_stride0    Stride between different tokens in input.
 * @param [in] input_stride1    Stride between different heads in input.
 * @param [in] input            Input key tensor (bfloat16) in shape (kv_len, num_heads, qk_head_dim).
 * @param [out] output          Output packed key tensor (bfloat16) in blocked order.
 */
kutacc_export void flash_attention_k_block_pack(int64_t kv_len, int64_t num_heads, int64_t qk_head_dim,
    int64_t output_len, int64_t input_stride0, int64_t input_stride1, bfloat16_t *input, bfloat16_t *output);

/**
 * @brief Perform Flash Attention with optional causal masking and variable sequence lengths.
 *
 * This is a blocked, online softmax based attention implementation that supports
 * chunked prefill and variable length sequences (varlen). The kernel uses pre‑packed
 * Q, K, V tensors for efficiency and reuses temporary buffers (out_block_old/new,
 * max_block_old/new, base_block_old/new) per thread.
 *
 * @param [in] q                    Query tensor (bfloat16) of shape (total_q_tokens, num_heads, qk_head_dim).
 * @param [in] k                    Key tensor (bfloat16) of shape (total_kv_tokens, num_heads, qk_head_dim).
 * @param [in] v                    Value tensor (bfloat16) of shape (total_kv_tokens, num_heads, vo_head_dim).
 * @param [out] out                 Output tensor (bfloat16) of shape (total_q_tokens, num_heads, vo_head_dim).
 * @param [in] pack_attn_q          Pre‑packed query tensor, shape (num_threads, br * qk_head_dim).
 * @param [in] pack_attn_k          Pre‑packed key tensor. If `kv_is_packed` (single sequence),
 *                                  shape (num_heads, padded_k_len, qk_head_dim); otherwise unused.
 * @param [in] pack_attn_v          Pre‑packed value tensor, similar layout to pack_attn_k.
 * @param [in] attn_s               Temporary score buffer (float) per thread,
 *                                  shape (num_threads, br * bc) where br, bc are block sizes.
 * @param [in] attn_out_block_old   Temporary output buffer (float) per thread,
 *                                  shape (num_threads, br, vo_head_dim).
 * @param [in] attn_out_block_new   Second temporary output buffer (float) for ping‑pong.
 * @param [in] attn_max_block_old   Temporary max‑value buffer per thread,
 *                                  shape (num_threads, br).
 * @param [in] attn_max_block_new   Second max‑value buffer.
 * @param [in] attn_base_block_old  Temporary sum‑of‑exp buffer per thread,
 *                                  shape (num_threads, br).
 * @param [in] attn_base_block_new  Second sum‑of‑exp buffer.
 * @param [in] causal               If true, apply causal masking (prevents attending to future tokens).
 * @param [in] softmax_scale        Scaling factor applied to Q·K^T before softmax (usually 1/sqrt(head_dim)).
 * @param [in] query_start_loc      Cumulative sequence lengths for queries (size num_seqs+1).
 * @param [in] key_start_loc        Cumulative sequence lengths for keys (size num_seqs+1).
 * @param [in] chunked_prefill_size Size of chunked prefill blocks (used for incremental decoding).
 * @param [in,out] seq_lens         Current total lengths of each sequence (modified during prefill).
 * @param [in,out] cur_lens         Current processed lengths of each sequence (modified during prefill).
 * @param [in] is_kv_packed         Only seq_len == 1 + kv prepacked -> is_kv_packed == true.
 */
kutacc_export void flash_attention(const Tensor<bfloat16_t, 3> &(q), const Tensor<bfloat16_t, 3> &(k),
    const Tensor<bfloat16_t, 3> &(v), const Tensor<bfloat16_t, 3> &(out), const Tensor<bfloat16_t, 2> &(pack_attn_q),
    const Tensor<bfloat16_t, 3> &(pack_attn_k), const Tensor<bfloat16_t, 3> &(pack_attn_v),
    const Tensor<float, 2> &(attn_s), const Tensor<float, 3> &(attn_out_block_old),
    const Tensor<float, 3> &(attn_out_block_new), const Tensor<float, 2> &(attn_max_block_old),
    const Tensor<float, 2> &(attn_max_block_new), const Tensor<float, 2> &(attn_base_block_old),
    const Tensor<float, 2> &(attn_base_block_new), bool causal, double softmax_scale,
    const Tensor<int, 1> &(query_start_loc), const Tensor<int, 1> &(key_start_loc), int chunked_prefill_size,
    std::vector<int64_t> &seq_lens, std::vector<int64_t> &cur_lens, bool is_kv_packed);

/**
 * @brief Simplified variable‑length attention (without pre‑packing and chunked prefill).
 *
 * This function implements standard attention for variable length sequences.
 * It slices each sequence from the concatenated input tensors using cumulative
 * start locations, then applies the `attention_impl` kernel per sequence and head.
 *
 * @param [in] q               Query tensor (bfloat16) of shape (total_q_tokens, num_heads, qk_head_dim).
 * @param [in] k               Key tensor (bfloat16) of shape (total_kv_tokens, num_heads, qk_head_dim).
 * @param [in] v               Value tensor (bfloat16) of shape (total_kv_tokens, num_heads, vo_head_dim).
 * @param [out] out            Output tensor (bfloat16) of shape (total_q_tokens, num_heads, vo_head_dim).
 * @param [in] causal          If true, apply causal masking.
 * @param [in] softmax_scale   Scaling factor for Q·K^T.
 * @param [in] query_start_loc Cumulative sequence lengths for queries (size num_seqs+1).
 * @param [in] key_start_loc   Cumulative sequence lengths for keys (size num_seqs+1).
 */

kutacc_export void varlen_attention(const Tensor<bfloat16_t, 3> &(q), const Tensor<bfloat16_t, 3> &(k),
    const Tensor<bfloat16_t, 3> &(v), const Tensor<bfloat16_t, 3> &(out), bool causal, double softmax_scale,
    const Tensor<int, 1> &(query_start_loc), const Tensor<int, 1> &(key_start_loc));

/**
 * @brief int8 gemm kernels
 * @details For input matrices mat1(M, K) and mat2(N, K), the data is first packed in an SME-friendly
 *          layout via s8_gemm_pack.
 * @note Key considerations:
 *       1. During packing/gemm tiling, M/N/K must be integer multiples of tile_m, tile_n, tile_k.
 *       2. block_x = X / tile_x, and we require block_m * block_n * block_k = thread_num to fully utilize compute resources.
 *       3. K % 4 == 0 && tile_k % 4 == 0
 * @example
 * @li M = 19, N = 2048, K = 1024
 *     Here M is prime, so tile_m must be 19. Then (N / tile_n) * (K / tile_k) = thread_num suffices.
 */
using MatrixTilingBlock = std::tuple<int64_t, int64_t, int64_t>;

/**
 * @brief Get the required size for the temporary buffer (tmpc) used in s8_s8_gemm_bf16_dq.
 *
 * The temporary buffer stores intermediate accumulation results when K is split into multiple
 * tile_k blocks. The required size is M * N * (2 * K / tile_k) bytes for bfloat16.
 *
 * @param [in] m            Number of rows of the left matrix (M).
 * @param [in] n            Number of columns of the right matrix (N).
 * @param [in] k            Inner dimension (K).
 * @param [in] t            Tiling block (tile_m, tile_n, tile_k). Only tile_k is used.
 * @return int64_t          Size in bytes of the temporary buffer needed.
 */
kutacc_export int64_t get_s8_s8_gemm_bf16_dq_tmpc_size(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t);

/**
 * @brief Perform int8 matrix multiplication with dequantization to bfloat16, including packing of the left matrix.
 *
 * This kernel packs the left matrix act_ptr (or input_ptr) on-the-fly, then calls the packed GEMM.
 * It is suitable when the left matrix is not pre‑packed.
 *
 * @param [in] m                Number of rows of the left matrix (M).
 * @param [in] n                Number of columns of the right matrix (N).
 * @param [in] k                Inner dimension (K).
 * @param [in] t                Tiling block (tile_m, tile_n, tile_k). Must satisfy tile_m == m.
 * @param [in] act_ptr          Input activation matrix (int8_t) of shape (m, k), row‑major.
 * @param [out] input_ptr       Temporary buffer used as packed left matrix (size determined by pack).
 * @param [in] weight_ptr       Weight matrix (int8_t) of shape (n, k), row‑major.
 * @param [in] act_scale_ptr    Per‑row activation scale factors (float) of length m.
 * @param [in] weight_scale_ptr Per‑row weight scale factors (float) of length n.
 * @param [out] output_ptr      Output matrix (bfloat16_t) of shape (m, n), row‑major.
 * @param [out] workspace       Temporary workspace buffer (size from get_s8_s8_gemm_bf16_dq_tmpc_size).
 */
kutacc_export void s8_s8_gemm_bf16_dq(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr,
    int8_t *input_ptr, int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr,
    bfloat16_t *workspace);

/**
 * @brief Perform int8 matrix multiplication with dequantization using pre‑packed matrices.
 *
 * Both left and right matrices must have been packed via s8_gemm_pack (or s8_gemm_pack_fusedmoe)
 * with compatible tiling parameters. This kernel is the core GEMM after packing.
 *
 * @param [in] m                Number of rows (M).
 * @param [in] n                Number of columns (N).
 * @param [in] k                Inner dimension (K).
 * @param [in] t                Tiling block (tile_m, tile_n, tile_k).
 * @param [in] act_ptr          Packed left matrix (int8_t) of shape (m, k).
 * @param [in] weight_ptr       Packed right matrix (int8_t) of shape (n, k).
 * @param [in] act_scale_ptr    Per‑row activation scale factors (float) of length m.
 * @param [in] weight_scale_ptr Per‑row weight scale factors (float) of length n.
 * @param [out] output_ptr      Output matrix (bfloat16_t) of shape (m, n), row‑major.
 * @param [out] workspace       Temporary workspace buffer.
 */
kutacc_export void s8_s8_packed_gemm_bf16_dq(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr,
    int8_t *weight_ptr, float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr, bfloat16_t *workspace);

/**
 * @brief Pack an int8 matrix into an SME‑friendly blocked layout.
 *
 * The input matrix of shape (r, c) is rearranged into blocks of size (split_r, split_c).
 * This packing improves memory locality and enables efficient outer‑product based GEMM.
 *
 * @param [in] r               Number of rows of the input matrix.
 * @param [in] c               Number of columns of the input matrix.
 * @param [in] split_r         Row block size for packing (should be a multiple of 16 for best performance).
 * @param [in] split_c         Column block size for packing (should be a multiple of 64 for best performance).
 * @param [in] input_ptr       Input matrix (int8_t) in row‑major order.
 * @param [out] output_ptr     Packed output matrix (layout specific).
 * @param [in] ldc             Leading dimension of input_ptr (if 0, assumed to be c).
 * @param [in] with_idx        If true, perform indexed packing using idx array (for fused MoE).
 * @param [in] idx             Optional index array (int) of length r, mapping each row to a token ID.
 */
kutacc_export void s8_gemm_pack(int64_t r, int64_t c, int64_t split_r, int64_t split_c, int8_t *input_ptr,
    int8_t *output_ptr, int64_t ldc = 0, bool with_idx = 0, int *idx = nullptr);

/**
 * @brief Get the required workspace sizes for fused MoE int8 GEMM (gateup/down).
 *
 * The workspace consists of three parts:
 * - tmpx: packed activation buffer (tilebuf_size * K)
 * - tmpy: intermediate accumulation buffer (tilebuf_size * N * (K / tile_k) / 2)
 * - tmp_scales: temporary scale buffer (tilebuf_size * 4)
 *
 * @param [in] N               Number of output features (columns of the weight matrix).
 * @param [in] K               Inner dimension (common feature size).
 * @param [in] tilebuf_size    Maximum number of tokens processed in one batch (tile in M dimension).
 * @param [in] t               Tiling block (tile_m, tile_n, tile_k). Only tile_k is used.
 * @return std::tuple<int64_t, int64_t, int64_t>  (tmpx_size, tmpy_size, tmpscale_size) in elements (not bytes).
 */
kutacc_export std::tuple<int64_t, int64_t, int64_t> get_fusedmoe_s8gemm_workspace_size(
    int64_t N, int64_t K, int64_t tilebuf_size, MatrixTilingBlock t);

/**
 * @brief Pack a slice of the activation matrix for fused MoE with optional per‑token indexing.
 *
 * This is a specialized pack for fused MoE where each row may belong to a different expert.
 * If with_idx is true, the idx array provides the original token IDs used to fetch scales.
 *
 * @param [in] r               Number of rows (tokens) to pack.
 * @param [in] c               Number of columns (K).
 * @param [in] split_c         Column block size for packing (tile_k).
 * @param [in] input_ptr       Input activation matrix (int8_t) in row‑major order.
 * @param [out] output_ptr     Packed output buffer.
 * @param [in] ldc             Leading dimension of input_ptr (if 0, assumed to be c).
 * @param [in] with_idx        If true, use idx for indexed packing.
 * @param [in] idx             Index array of length r (token IDs).
 */
kutacc_export void s8_gemm_pack_fusedmoe(int64_t r, int64_t c, int64_t split_c, int8_t *input_ptr,
    int8_t *output_ptr, int64_t ldc = 0, bool with_idx = 0, int *idx = nullptr);

/**
 * @brief Fused MoE gate‑up projection (int8 gemm with SiLU and scaling).
 *
 * This kernel computes the gate and up projections for a mixture‑of‑experts layer.
 * It handles per‑expert token gathering, activation scaling, packed GEMM, and dequantization.
 *
 * @param [in] total_bs        Total number of tokens across all experts.
 * @param [in] K               Inner dimension (feature size).
 * @param [in] N               Output feature size (intermediate dimension after gate/up).
 * @param [in] num_experts     Number of experts.
 * @param [in] lda             Leading dimension of the input activation matrix (K).
 * @param [in] ldas            Leading dimension of the act_scale array (stride per token).
 * @param [in] acts            Activation matrix (int8_t) of shape (total_bs?, K).
 * @param [in] weights         Weight matrix (int8_t) of shape (num_experts * N, K).
 * @param [in] acts_scale      Per‑token activation scale factors (float) of length total_bs.
 * @param [in] weights_scale   Per‑expert weight scale factors (float) of shape (num_experts, N).
 * @param [in] token_ids       Token ID mapping (int) of length total_bs (to fetch correct act_scale).
 * @param [in] experts_offset  Offsets into token_ids for each expert (length num_experts+1).
 * @param [out] output         Output matrix (bfloat16_t) of shape (total_bs, N).
 * @param [out] tmpx           Temporary buffer for packed activation (size from get_fusedmoe_s8gemm_workspace_size).
 * @param [out] tmpy           Temporary accumulation buffer (float).
 * @param [out] tmp_scales     Temporary scale buffer (float).
 * @param [in] t               Tiling block (tile_m, tile_n, tile_k).
 * @param [in] tilebuf_size    Maximum number of tokens processed in one tile (for buffer‑limited cases).
 * @param [in] n_slice         Optional slice in N dimension for dual‑expert parallel optimization.
 */
kutacc_export void fusedmoe_gateup(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int64_t lda,
    int64_t ldas, int8_t *acts, int8_t *weights, float *acts_scale, float *weights_scale, int *token_ids,
    int *experts_offset, bfloat16_t *output, int8_t *tmpx, float *tmpy, float *tmp_scales, MatrixTilingBlock t,
    int64_t tilebuf_size, std::optional<int64_t> n_slice);

/**
 * @brief Fused MoE down projection (int8 gemm).
 *
 * This kernel computes the down projection after the gate/up combination.
 * It uses the same packing and GEMM infrastructure but no SiLU activation.
 *
 * @param [in] total_bs        Total number of tokens across all experts.
 * @param [in] K               Inner dimension (feature size from gate/up output).
 * @param [in] N               Output feature size (final hidden dimension).
 * @param [in] num_experts     Number of experts.
 * @param [in] acts            Activation matrix (int8_t) from gate/up (already processed).
 * @param [in] weights         Weight matrix (int8_t) of shape (num_experts * N, K).
 * @param [in] acts_scale      Per‑token activation scale factors (float) of length total_bs.
 * @param [in] weights_scale   Per‑expert weight scale factors (float) of shape (num_experts, N).
 * @param [in] experts_offset  Offsets into token_ids for each expert.
 * @param [out] output         Output matrix (bfloat16_t) of shape (total_bs, N).
 * @param [out] tmpx           Temporary buffer for packed activation.
 * @param [out] tmpy           Temporary accumulation buffer (float).
 * @param [in] t               Tiling block.
 * @param [in] tilebuf_size    Maximum tokens per tile.
 * @param [in] n_slice         Optional slice in N dimension for dual‑expert parallel optimization.
 */
kutacc_export void fusedmoe_down(int64_t total_bs, int64_t K, int64_t N, int64_t num_experts, int8_t *acts,
    int8_t *weights, float *acts_scale, float *weights_scale, int *experts_offset, bfloat16_t *output, int8_t *tmpx,
    float *tmpy, MatrixTilingBlock t, int64_t tilebuf_size, std::optional<int64_t> n_slice);

/**
 * @brief bf16 gemm kernels
 * @details For input matrices mat1(M, K) and mat2(N, K), the data is first packed in an SME-friendly
 *          layout via bf16_gemm_pack.
 * @note Key considerations:
 *       1. During packing/gemm tiling, M/N/K must be integer multiples of tile_m, tile_n, tile_k.
 *       2. block_x = X / tile_x, and we require block_m * block_n * block_k = thread_num to fully utilize compute resources.
 *       3. K % 2 == 0 && tile_k % 2 == 0
 * @example
 * @li M = 19, N = 2048, K = 1024
 *     Here M is prime, so tile_m must be 19. Then (N / tile_n) * (K / tile_k) = thread_num suffices.
 */

/**
 * @brief Perform bfloat16 matrix multiplication using pre‑packed matrices.
 *
 * Both left and right matrices must be packed via bf16_gemm_pack. The kernel uses SME
 * outer‑product instructions for high performance.
 *
 * @param [in] m              Number of rows of the left matrix (M).
 * @param [in] n              Number of columns of the right matrix (N).
 * @param [in] k              Inner dimension (K).
 * @param [in] t              Tiling block (tile_m, tile_n, tile_k).
 * @param [in] act_ptr        Packed left matrix (bfloat16_t) of shape (m, k).
 * @param [in] weight_ptr     Packed right matrix (bfloat16_t) of shape (n, k).
 * @param [out] output_ptr    Output matrix (bfloat16_t) of shape (m, n), row‑major.
 * @param [out] tmpc          Temporary workspace buffer (for K‑split cases).
 * @param [in] bias           Bias buffer (default for nullptr).
 * @param [in] row_bias       Bias type flag: if true, use per-row bias, else use per-col bias (default for false). 
 */
kutacc_export void bf16_packed_gemm(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, bfloat16_t *act_ptr,
    bfloat16_t *weight_ptr, bfloat16_t *output_ptr, bfloat16_t *tmpc, float *bias = nullptr, bool row_bias = false);

/**
 * @brief Pack a bfloat16 matrix into an SME‑friendly blocked layout.
 *
 * The input matrix of shape (r, c) is rearranged into blocks of size (split_r, split_c).
 *
 * @param [in] r                Number of rows.
 * @param [in] c                Number of columns.
 * @param [in] split_r          Row block size (should be multiple of 16).
 * @param [in] split_c          Column block size (should be multiple of 64).
 * @param [in] input_ptr        Input bfloat16 matrix, row‑major.
 * @param [out] output_ptr      Packed output matrix.
 */
kutacc_export void bf16_gemm_pack(int64_t r, int64_t c, int64_t split_r, int64_t split_c, bfloat16_t *input_ptr,
    bfloat16_t *output_ptr);

/**
 * @brief Batch pack multiple bfloat16 matrices into a blocked layout.
 *
 * This kernel packs a batch of matrices with shape (bs, m, n) into a contiguous
 * packed buffer. The packing parameters (split_r, split_c) are determined internally.
 *
 * @param [in] bs               Batch size.
 * @param [in] m                Number of rows per matrix.
 * @param [in] n                Number of columns per matrix.
 * @param [in] stride_bs        Stride between matrices in the source buffer (in elements).
 * @param [in] stride_m         Stride between rows within a matrix in the source buffer.
 * @param [in] src              Source buffer (void*), actual type determined by dtype.
 * @param [out] dst             Destination packed buffer.
 * @param [in] dtype            Data type: 1 for float32? (implementation uses 1 for A, 2 for B) – see code.
 */
kutacc_export void batch_bf16_gemm_pack(int64_t bs, int64_t m, int64_t n, int64_t stride_bs, int64_t stride_m,
    void *src, void *dst, int64_t dtype);

/**
 * @brief Perform batched mixed‑precision GEMM: bfloat16 activation × int8 weight → bfloat16 output.
 *
 * For each batch, the kernel computes C_b = A_b × B_b, where A is bfloat16, B is int8,
 * and the result is written as bfloat16. Per‑batch row scaling (rscale) and column scaling (cscale)
 * can be applied.
 *
 * @param [in] bs               Batch size.
 * @param [in] m                Number of rows in A and C (M per batch).
 * @param [in] n                Number of columns in B and C (N per batch).
 * @param [in] k                Inner dimension (K).
 * @param [in] stride_bs        Stride between batches in A and C (in elements).
 * @param [in] stride_m         Stride between rows within a batch (in elements).
 * @param [in] act              Activation tensor (bfloat16_t) of shape (bs, m, k).
 * @param [in] weight           Weight tensor (int8_t) of shape (bs, n, k).
 * @param [out] out             Output tensor (bfloat16_t) of shape (bs, m, n).
 * @param [in] rscale           Optional per‑batch row scaling factors (float) of shape (bs, n).
 * @param [in] cscale           Optional per‑batch column scaling factors (float) of shape (bs, k).
 */
kutacc_export void batch_bf16_s8_packed_gemm_bf16(int64_t bs, int64_t m, int64_t n, int64_t k, int64_t stride_bs,
    int64_t stride_m, bfloat16_t *act, int8_t *weight, bfloat16_t *out, float *rscale, float *cscale);

/**
 * @brief Quantize a bfloat16 matrix to int8 using per‑row scaling.
 *
 * For each row i, compute the absolute max, scale = max / 127.0f, then quantize each element
 * to int8 via rounding and clamping. The scale is stored in scale[i].
 *
 * @param [in] height        Number of rows.
 * @param [in] width         Number of columns.
 * @param [in] input         Input bfloat16 matrix.
 * @param [in] input_stride  Stride between rows of input.
 * @param [out] out          Output int8 matrix.
 * @param [in] out_stride    Stride between rows of out.
 * @param [out] scale        Per‑row scale factor (float array of length height).
 */
kutacc_export void quant(int64_t height, int64_t width, const bfloat16_t *input, int64_t input_stride, int8_t *out,
    int64_t out_stride, float *scale);

/**
 * @brief Quantize and pack a bfloat16 matrix into a blocked layout for efficient matrix multiplication.
 *
 * The input is first scaled per‑row, then the quantized int8 values are rearranged into
 * a packing format (e.g., to improve cache locality). Scaling factors are computed per row.
 *
 * @param [in] height        Number of rows.
 * @param [in] width         Number of columns.
 * @param [in] input_data    Input bfloat16 matrix.
 * @param [out] output_data  Output packed int8 data (layout specific).
 * @param [out] scale_data   Per‑row scale factors (float array of length height).
 */
kutacc_export void quant_pack(int64_t height, int64_t width, bfloat16_t *input_data, int8_t *output_data,
    float *scale_data);

/**
 * @brief Quantize, optionally save unpacked result, and also produce a packed representation.
 *
 * This kernel combines quantization with two outputs: a standard row‑major int8 matrix
 * (output_data) and a packed layout (packed_output_data). Scaling factors are written both
 * to scale_data (with given stride) and to packed_scale_data.
 *
 * @param [in] height             Number of rows.
 * @param [in] width              Number of columns.
 * @param [in] input_data         Input bfloat16 matrix.
 * @param [in] ldo                Leading dimension of output_data (row stride).
 * @param [out] output_data       Output int8 matrix in row‑major order.
 * @param [in] lds                Leading dimension of scale_data (row stride).
 * @param [out] scale_data        Per‑row scale factors (float array of shape (height, lds?)).
 * @param [out] packed_output_data  Output int8 data in packed layout.
 * @param [out] packed_scale_data   Per‑row scale factors stored for the packed layout.
 */
kutacc_export void quant_save_pack(int64_t height, int64_t width, bfloat16_t *input_data, int64_t ldo,
    int8_t *output_data, int64_t lds, float *scale_data, int8_t *packed_output_data, float *packed_scale_data);

/**
 * @brief Apply RMS normalization (Root Mean Square) to the input activations.
 *
 * For each row i, compute the RMS of the row (or residual+acts if has_residual is true),
 * then normalize with weights and write to outs. If has_residual is true, the residual
 * is added to acts before normalization.
 *
 * @tparam has_residual      If true, residual tensor is added to acts before normalization.
 * @param [in] height        Number of rows.
 * @param [in] width         Number of columns per row.
 * @param [in] acts          Input activation matrix (bfloat16), stride acts_stride.
 * @param [in] acts_stride   Stride between rows of acts.
 * @param [in] weights       RMS weight vector of length width.
 * @param [in] eps           Small epsilon to avoid division by zero.
 * @param [in] residual      Residual matrix (bfloat16) of same shape, only used if has_residual.
 * @param [out] outs         Output normalized matrix (bfloat16), row‑major.
 */
template <bool has_residual>
kutacc_export void rmsnorm(int64_t height, int64_t width, bfloat16_t *acts, int64_t acts_stride,
    const bfloat16_t *weights, float eps, bfloat16_t *residual, bfloat16_t *outs);

/**
 * @brief Apply RMS normalization and quantize the result to int8 with per‑row scaling.
 *
 * Similar to rmsnorm, but after normalization the output is quantized to int8 using
 * a dynamically computed per‑row scale (max / 127.0f). The scale factors are stored.
 *
 * @tparam has_residual      If true, the residual is added to acts before normalization.
 * @param [in] height        Number of rows.
 * @param [in] width         Number of columns per row.
 * @param [in] acts          Input activation matrix (bfloat16), stride acts_stride.
 * @param [in] acts_stride   Stride of acts.
 * @param [in] weights       RMS weight vector.
 * @param [in] eps           Epsilon for RMS.
 * @param [in] residual      Residual matrix (bfloat16), stride res_stride, used if has_residual.
 * @param [in] res_stride    Stride of residual.
 * @param [out] outs         Output int8 matrix, stride outs_stride.
 * @param [in] outs_stride   Stride of outs.
 * @param [out] scales       Per‑row scale factors, stride scales_stride.
 * @param [in] scales_stride Stride of scales.
 */
template <bool has_residual>
kutacc_export void rmsnorm_quant(int64_t height, int64_t width, bfloat16_t *acts, int64_t acts_stride,
    const bfloat16_t *weights, float eps, bfloat16_t *residual, int64_t res_stride, int8_t *outs, int64_t outs_stride,
    float *scales, int64_t scales_stride);

/**
 * @brief Embedding table lookup with optional vocabulary range.
 *
 * For each token index in input, copy the corresponding row from the weight table
 * (vocab_size × hidden) into out. If the index is outside [vocab_start, vocab_end),
 * the output row is set to zero.
 *
 * @param [in] input         1D array of token indices (int64_t) of length n_tokens.
 * @param [in] weight        Embedding table (element_size * hidden per row).
 * @param [out] out          Output buffer (n_tokens × hidden).
 * @param [in] element_size  Size in bytes of each embedding element (e.g., 2 for bf16).
 * @param [in] n_tokens      Number of tokens to lookup.
 * @param [in] hidden        Hidden dimension size.
 * @param [in] vocab_start   Start index of valid vocabulary range (inclusive).
 * @param [in] vocab_end     End index of valid vocabulary range (exclusive).
 */
kutacc_export void embedding(const int64_t *input, const void *weight, void *out, int64_t element_size,
    int64_t n_tokens, int64_t hidden, int64_t vocab_start, int64_t vocab_end);

/** @brief the data type of kurmcl*/
typedef enum kurmcl_datatype {
    KURMCL_DATATYPE_CHAR = 0,
    KURMCL_DATATYPE_INT,
    KURMCL_DATATYPE_LONG,
    KURMCL_DATATYPE_FLOAT,
    KURMCL_DATATYPE_DOUBLE
} kurmcl_datatype_t;

/**
 * @brief callback function type for out-of-band allgather operation
 *
 * @param [in] sendbuf        send buffer for allgather
 * @param [in] recvbuf        receive buffer for allgather
 * @param [in] count          number of data elements
 * @param [in] group          communication group context
 * @param [in] datatype       data type of elements
 * @return int                return 0 for success, non-zero for failure
 */
typedef int (*kurmcl_oob_allgather_cb_t)(
    const void *sendbuf, void *recvbuf, int count, void *group, kurmcl_datatype_t datatype);

/**
 * @brief callback function type for out-of-band barrier operation
 *
 * @param [in] group          communication group context
 * @return int                return 0 for success, non-zero for failure
 */
typedef int (*kurmcl_oob_barrier_cb_t)(void *group);

/**
 * @brief callback function type for out-of-band alltoall operation
 *
 * @param [in] sendbuf        send buffer for alltoall
 * @param [in] sendcount      number of elements to send per process
 * @param [in] sendtype       data type of send elements
 * @param [in] recvbuf        receive buffer for alltoall
 * @param [in] recvcount      number of elements to receive per process
 * @param [in] recvtype       data type of receive elements
 * @param [in] group          communication group context
 * @return int                return 0 for success, non-zero for failure
 */
typedef int (*kurmcl_oob_alltoall_cb_t)(const void *sendbuf, int sendcount, kurmcl_datatype_t sendtype, void *recvbuf,
    int recvcount, kurmcl_datatype_t recvtype, void *group);

/** @brief collection of out-of-band communication callback functions */
typedef struct kurmcl_oob_cb {
    kurmcl_oob_allgather_cb_t oob_allgather;
    kurmcl_oob_barrier_cb_t oob_barrier;
    kurmcl_oob_alltoall_cb_t oob_alltoall;
} kurmcl_oob_cb_t, *kurmcl_oob_cb_h;

/** @brief the parameters of kurmcl communicator */
typedef struct kurmcl_conn_info *kurmcl_conn_info_h;

/**
 * @brief kurmcl communicator create function.
 *
 * @param [in] size        the size of communicator
 * @param [in] rank        the rank of communicator
 * @param [in] oob_cbs     out of band callback function
 * @param [in] group       mpi communicator
 * @param [out] conn_info  kurmcl communicator
 * @return int             return 0 if success, -1 if failed
 */
kutacc_export int kurmcl_comm_create(int size, int rank, kurmcl_oob_cb_h oob_cbs, void *group,
    kurmcl_conn_info_h *conn_info);

/**
 * @brief kurmcl barrier function.
 *
 * @param [in] conn_info   kurmcl communicator
 */
kutacc_export void kurmcl_barrier(kurmcl_conn_info_h conn_info);

/**
 * @brief initialize dispatch related data structures, register memory and address exchange
 *
 * @param [in] x_data                               send buffer for storing token
 * @param [in] recv_src_info_data                   main table for received tokens and source process IDs
 * @param [in] recv_src_info_bak_data               backup table (alternate use to prevent overwrite)
 * @param [in] num_experts                          number of experts
 * @param [in] multiple                             coefficient for recv buffer size in prefill scenario
 * @param [in] num_max_dispatch_tokens_per_rank     maximum dispatched tokens per process
 * @param [in] hidden                               feature dimension of token
 * @param [in] max_num_tokens                       max batch size
 * @param [in] dtp                                  expert parallel coefficient, dtp = ep / dp
 * @param [in] src_info                             token destination info table from sender side
 * @param [in] disbuf_baseptr                       base address of dispatch receive buffer
 * @param [in] ds_conn_info                         kurmcl communicator
 */
kutacc_export void moe_dispatch_init(uint8_t *x_data, int16_t *recv_src_info_data,
    int16_t *recv_src_info_bak_data, int64_t num_experts, int64_t multiple,
    int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int64_t max_num_tokens, int64_t dtp, int16_t *src_info,
    void *disbuf_baseptr, kurmcl_conn_info_h ds_conn_info);

/**
 * @brief dispatch tokens to corresponding experts based on TopK information
 *
 * @param [in] x_data                               send buffer for storing token
 * @param [in] topk_idx_data                        TopK routing table for token dispatch
 * @param [in] num_tokens                           batch size
 * @param [in] num_topk                             TopK value for token routing
 * @param [in] num_max_dispatch_tokens_per_rank     maximum dispatched tokens per process
 * @param [in] hidden                               feature dimension of token
 * @param [in] parallel_policy_data                 array storing parallel policy (ep, dp, tp)
 * @param [in] batch_id                             batch ID of current dispatch task
 * @param [in] ds_conn_info                         kurmcl communicator
 */
kutacc_export void moe_dispatch_send(uint8_t *x_data, int16_t *topk_idx_data, int64_t num_tokens, int64_t num_topk,
    int64_t num_max_dispatch_tokens_per_rank, int64_t hidden, int16_t *parallel_policy_data,
    int64_t batch_id, kurmcl_conn_info_h ds_conn_info);

/**
 * @brief block until the required token is received
 *
 * @param [in] batch_id                             batch ID of current receive task
 * @param [in] ds_conn_info                         kurmcl communicator
 */
kutacc_export void moe_dispatch_recv(int64_t batch_id, kurmcl_conn_info_h ds_conn_info);

/**
 * @brief release dispatch related data structures
 */
kutacc_export void moe_dispatch_finalize();

/**
 * @brief initialize combine related data structures, register memory and address exchange
 *
 * @param [in] new_packed_recv_x_data               Packed send buffer for combine
 * @param [in] num_tokens                           batch size
 * @param [in] num_experts                          number of experts
 * @param [in] num_max_dispatch_tokens_per_rank     maximum dispatched tokens per process
 * @param [in] n_activated_experts                  number of activated experts
 * @param [in] hidden                               feature dimension of token
 * @param [in] group_ptr                            Array storing final allgather results of DP group
 * @param [in] local_rank                           Local rank in DP communication group
 * @param [in] recv_group                           Array of shared memory recv buffer addresses for DP group
 * @param [in] ds_conn_info                         kurmcl communicator
 */
kutacc_export void moe_combine_init(bfloat16_t *new_packed_recv_x_data, int64_t num_tokens, int16_t num_experts,
    int16_t num_max_dispatch_tokens_per_rank, int16_t n_activated_experts, int16_t hidden,
    std::vector<bfloat16_t *> &&group_ptr, int local_rank, std::vector<bfloat16_t *> &&recv_group,
    kurmcl_conn_info_h ds_conn_info);

/**
 * @brief send activation values back to the original experts
 *
 * @param [in] x_data                               deprecated and unused buffer
 * @param [in] src_info_data                        token destination info from dispatch phase
 * @param [in] num_max_dispatch_tokens_per_rank     maximum dispatched tokens per process
 * @param [in] num_experts                          number of experts
 * @param [in] hidden                               feature dimension of token
 * @param [in] parallel_policy_data                 array storing parallel policy (ep, dp, tp)
 * @param [in] batch_id                             batch ID of current combine task
 * @param [in] ds_conn_info                         kurmcl communicator
 * @param [in] combined_x_data                      buffer storing the final combine result
 * @param [in] topk_idx_data                        TopK routing table for token dispatch
 * @param [in] topk_weights_data                    TopK weights table
 * @param [in] num_tokens                           batch size
 * @param [in] num_topk                             TopK value for token routing
 * @param [in] enable_allgather                     switch to control allgather in final combine stage
 */
kutacc_export void moe_combine_send(bfloat16_t *x_data, int16_t *src_info_data,
    int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, int64_t hidden, int16_t *parallel_policy_data,
    int64_t batch_id, kurmcl_conn_info_h ds_conn_info, bfloat16_t *combined_x_data, int16_t *topk_idx_data,
    float *topk_weights_data, int64_t num_tokens, int64_t num_topk, bool enable_allgather = true);

/**
 * @brief confirm receipt of activation value and multiply-add with weight
 *
 * @param [in, out] combined_x_data                 buffer storing the final combine result
 * @param [in] topk_idx_data                        TopK routing table for token dispatch
 * @param [in] topk_weights_data                    TopK weights table
 * @param [in] num_tokens                           batch size
 * @param [in] num_max_dispatch_tokens_per_rank     maximum dispatched tokens per process
 * @param [in] num_topk                             TopK value for token routing
 * @param [in] hidden                               feature dimension of token
 * @param [in] batch_id                             batch ID of current combine task
 * @param [in] win                                  shared memory window handle
 * @param [in] ds_conn_info                         kurmcl communicator
 */
kutacc_export void moe_combine_recv(bfloat16_t *combined_x_data, int16_t *topk_idx_data, float *topk_weights_data,
    int64_t num_tokens, int64_t num_max_dispatch_tokens_per_rank, int64_t num_topk, int64_t hidden, int64_t batch_id,
    kupl_shm_win_h win, kurmcl_conn_info_h ds_conn_info);

/**
 * @brief release combine related data structures
 */
kutacc_export void moe_combine_finalize();

/**
 * @brief initialization of cross communicator transmission
 *
 * @param [in] base_ptr                         Base address for PP activation transfer
 * @param [in] size                             Max buffer size for PP activation transfer
 * @param [in] ds_conn_info                     kurmcl communicator
 */
kutacc_export void pp_init(uint8_t *base_ptr, int64_t size, kurmcl_conn_info_h ds_conn_info);

/**
 * @brief send the activation value to the specified dest_rank
 *
 * @param [in] dest_rank                        Global peer rank to send activation values to
 * @param [in] send_offset                      Offset relative to the base address for PP activation transfer
 * @param [in] size                             buffer size for PP activation transfer
 * @param [in] ds_conn_info                     kurmcl communicator
 */
kutacc_export void pp_put(int64_t dest_rank, int64_t send_offset, int64_t size, kurmcl_conn_info_h ds_conn_info);

/**
 * @brief waiting fro the activation value of src_rank
 *
 * @param [in] src_rank                         Global peer rank to receive activation values from
 * @param [in] ds_conn_info                     kurmcl communicator
 */
kutacc_export void pp_recv(int64_t src_rank, kurmcl_conn_info_h ds_conn_info);

/**
 * @brief point to point barrier
 *
 * @param [in] remote_rank                      Remote peer rank for P2P barrier
 * @param [in] ds_conn_info                     kurmcl communicator
 */
kutacc_export void pp_barrier(int64_t remote_rank, kurmcl_conn_info_h ds_conn_info);

/** @brief the data type of shared memory communication*/
typedef enum shm_datatype {
    SHM_DATATYPE_INT8 = 0,
    SHM_DATATYPE_UINT8,
    SHM_DATATYPE_BFLOAT16
} shm_datatype_t;

/** @brief the request of alltoall by shared memory*/
typedef struct shm_alltoall_request *shm_alltoall_request_h;

/**
 * @brief create the request of alltoall
 *
 * @param [in] comm_rank                            the relative rank of current process in the communication group
 * @param [in] comm_size                            the communication size of alltoall (only support 8 or 16 now)
 * @param [in] max_num_elements                     the maximum number of elements
 * @param [in] datatype                             the datatype of elements in communication buffers
 * @param [out] extra_buffer_size                   the extra size required by shm_alltoall
 * @param [out] request                             the request of alltoall
 */
kutacc_export void shm_alltoall_request_create(int comm_rank, int comm_size, size_t max_num_elements,
    shm_datatype_t datatype, size_t &extra_buffer_size, shm_alltoall_request_h &request);

/**
 * @brief alltoall request initialize
 *
 * @param [in] extra_buffers                the extra buffers required by alltoall (guaranteed to be shared memory)
 * @param [in] comm_win                     the kupl win of the comm
 * @param [in] die_win                      the kupl win of the die
 * @param [in] socket_win                   the kupl win of the socket
 * @param [in] node_win                     the kupl win of the node
 * @param [in] request                      the request of alltoall
 */
kutacc_export void shm_alltoall_request_init(void **extra_buffers, kupl_shm_win_h comm_win,
    kupl_shm_win_h die_win, kupl_shm_win_h socket_win, kupl_shm_win_h node_win,
    const shm_alltoall_request_h &request);

/**
 * @brief destroy the request of alltoall
 *
 * @param [in] request                              the request of alltoall
 */
kutacc_export void shm_alltoall_request_destroy(const shm_alltoall_request_h &request);

/**
 * @brief alltoall by shared memory
 *
 * @param [in] send_buffers                 the send buffers in the group (guaranteed to be shared memory)
 * @param [in, out] recv_buffers            the recv buffer of the current processe
 * @param [in] height                       the value of the height dimension
 * @param [in] width                        the value of the width dimension
 * @param [in] dim                          the dimension of communication
 * @param [in] need_comm_fence              if need comm fence in the function
 * @param [in] request                      the request of alltoall
 */
kutacc_export void shm_alltoall2D(void **send_buffers, void *recv_buffers, int64_t height, int64_t width, int64_t dim,
    bool need_comm_fence, const shm_alltoall_request_h &request);

/** @brief the request of allreduce by shared memory */
typedef struct shm_allreduce_request *shm_allreduce_request_h;

/**
 * @brief create the request of allreduce
 *
 * @param [in] comm_rank                            the relative rank of current process in the communication group
 * @param [in] comm_size                            the communication size of allreduce (only support 8 or 16 now)
 * @param [in] max_num_elements                     the maximum number of elements
 * @param [in] datatype                             the datatype of elements in communication buffers
 * @param [out] extra_buffer_size                   the extra size required by shm_allreduce
 * @param [out] request                             the request of allreduce
 */
kutacc_export void shm_allreduce_request_create(int comm_rank, int comm_size, size_t max_num_elements,
    shm_datatype_t datatype, size_t &extra_buffer_size, shm_allreduce_request_h &request);

/**
 * @brief allreduce request initialize
 *
 * @param [in] extra_buffers                the extra buffers required by allreduce (guaranteed to be shared memory)
 * @param [in] comm_win                     the kupl win of the comm
 * @param [in] die_win                      the kupl win of the die
 * @param [in] socket_win                   the kupl win of the socket
 * @param [in] node_win                     the kupl win of the node
 * @param [in] request                      the request of allreduce
 */
kutacc_export void shm_allreduce_request_init(void **extra_buffers, kupl_shm_win_h comm_win, kupl_shm_win_h die_win,
    kupl_shm_win_h socket_win, kupl_shm_win_h node_win, const shm_allreduce_request_h &request);

/**
 * @brief destroy the request of allreduce
 *
 * @param [in] request                              the request of allreduce
 */
kutacc_export void shm_allreduce_request_destroy(const shm_allreduce_request_h &request);

/**
 * @brief allreduce by shared memory
 *
 * @param [in, out] buffers                 the communication buffers in the group (guaranteed to be shared memory)
 * @param [in] num_elements                 the number of elements
 * @param [in] request                      the request of allreduce
 */
kutacc_export void shm_allreduce(void **buffers, size_t num_elements, const shm_allreduce_request_h &request);

/** @brief the request of reduce_scatter by shared memory */
typedef struct shm_reduce_scatter_request *shm_reduce_scatter_request_h;

/**
 * @brief create the request of reduce_scatter
 *
 * @param [in] rank                                 the relative rank of current process in the communication group
 * @param [in] comm_size                            the communication size of reduce_scatter (only support 8 or 16 now)
 * @param [in] datatype                             the datatype of elements in communication buffers
 * @param [out] fence_buffer_size                   the extra fence_buffer overhead used by the reduce_scatter algorithm
 * @param [out] request                             the request of reduce_scatter
 */
kutacc_export void shm_reduce_scatter_request_create(int rank, int comm_size, shm_datatype_t datatype,
    size_t &fence_buffer_size, shm_reduce_scatter_request_h &request);

/**
 * @brief reduce_scatter request initialize
 *
 * @param [in] fence_buffers                the fence buffers required by reduce_scatter(guaranteed to be shared memory)
 * @param [in] shm_win_intra_die            the kupl win of the die
 * @param [in] shm_win_intra_socket         the kupl win of the socket
 * @param [in] shm_win_intra_node           the kupl win of the node
 * @param [in] request                      the request of reduce_scatter
 */
kutacc_export void shm_reduce_scatter_request_init(void **fence_buffers, kupl_shm_win_h shm_win_intra_die,
    kupl_shm_win_h shm_win_intra_socket, kupl_shm_win_h shm_win_intra_node,
    const shm_reduce_scatter_request_h &request);

/** @brief destroy the request of reduce_scatter
 *
 * @param [in] request                              the request of allreduce
 */
kutacc_export void shm_reduce_scatter_request_destroy(const shm_reduce_scatter_request_h &request);

/**
 * @brief reduce_scatter by shared memory
 *
 * @param [in, out] buffers                 the buffer address array of other processes in DP
 * @param [in] height                       the height dimension parameter
 * @param [in] width                        the width dimension parameter
 * @param [in] request                      the request of reduce_scatter
 */
kutacc_export void shm_reduce_scatter(void **buffers, int64_t height, int64_t width,
    const shm_reduce_scatter_request_h &request);

/** @brief the request of batch_allgather by shared memory */
typedef struct shm_allgather_request *shm_allgather_request_h;

/**
 * @brief create the request of allgather
 *
 * @param [in] rank                                 the relative rank of current process in the communication group
 * @param [in] comm_size                            the communication size of allgather (only support 8 or 16 now)
 * @param [out] request                             the request of allgather
 */
kutacc_export void shm_allgather_request_create(int rank, int comm_size, shm_allgather_request_h &request);

/**
 * @brief allgather request initialize
 *
 * @param [in] shm_win_intra_die            the kupl win of the die
 * @param [in] shm_win_intra_socket         the kupl win of the socket
 * @param [in] shm_win_intra_node           the kupl win of the node
 * @param [in] request                      the request of allgather
 */
kutacc_export void shm_allgather_request_init(kupl_shm_win_h shm_win_intra_die,
    kupl_shm_win_h shm_win_intra_socket, kupl_shm_win_h shm_win_intra_node,
    const shm_allgather_request_h &request);

/**
 * @brief destroy the request of allgather
 *
 * @param [in] request                      the request of allgather
 */
kutacc_export void shm_allgather_request_destroy(const shm_allgather_request_h &request);

/**
 * @brief perform batch shared memory allgather
 *
 * @param [in] batch                             batch size
 * @param [in] sendbuf                           send buffer for allgather
 * @param [in] sendcount                         total byte size of send data
 * @param [in, out] recvbuf                      receive buffer for allgather
 * @param [in] recvcount                         total byte size of receive buffer
 * @param [in] datatype                          the datatype of elements in communication buffers
 * @param [in] remote_sendbuf                    send buffer addresses array of DP processes
 * @param [in, out] remote_recvbuf               receive buffer addresses array of DP processes
 * @param [in] buffer_size                       the size of each shared memory buffer
 * @param [in] request                           the request of allgather
 * @param [in] is_hierarchical                   if use hierarchical communication
 * @param [in] use_base                          switch to select allgather version
 */
kutacc_export void shm_batch_allgather(int64_t batch, void *sendbuf, int64_t sendcount, void *recvbuf,
    int64_t recvcount, shm_datatype_t datatype, uint8_t **remote_sendbuf, uint8_t **remote_recvbuf,
    int64_t buffer_size, const shm_allgather_request_h &request,
    bool is_hierarchical, bool use_base);

/**
 * @brief perform dual-path shared memory allgather operation
 *
 * @param [in] src0                                 send buffer address for the 1st allgather
 * @param [in] src0_size                            data size of send buffer for the 1st allgather
 * @param [out] dst0                                local destination buffer for the 1st allgather
 * @param [in] dst0_size                            data size of local destination buffer for the 1st allgather
 * @param [in] src1                                 send buffer address for the 2nd allgather
 * @param [in] src1_size                            data size of send buffer for the 2nd allgather
 * @param [out] dst1                                local destination buffer for the 2nd allgather
 * @param [in] dst1_size                            data size of local destination buffer for the 2nd allgather
 * @param [in, out] remote_buffers0                 recv buffer array for 1st allgather (includes all DP process addresses)
 * @param [in, out] remote_buffers1                 recv buffer array for 2nd allgather (includes all DP process addresses)
 * @param [in] datatype                             the datatype of elements in communication buffers
 * @param [in] barrier                              barrier function of the communication group
 * @param [in] request                              the request of allgather
 */
kutacc_export void shm_dual_allgather(void *src0, int64_t src0_size, void *dst0, int64_t dst0_size,
    void *src1, int64_t src1_size, void *dst1, int64_t dst1_size,
    void **remote_buffers0, void **remote_buffers1, shm_datatype_t datatype,
    const std::function<void()> &barrier, const shm_allgather_request_h &request);

/**
 * @brief get the total number of threads
 *
 * @return int64_t                                  the total number of threads
 */
kutacc_export int64_t get_thread_num();

/**
 * @brief get the id of the current thread
 *
 * @return int64_t                                  the id of the current thread
 */
kutacc_export int64_t get_thread_id();

/**
 * @brief create a parallel for the specific function in the specific range
 *
 * @param [in] begin                                the begin of the range
 * @param [in] end                                  the end of the range
 * @param [in] grain_size                           the grain size for each thread
 * @param [in] func                                 the function for this parallel
 */
kutacc_export void parallel_for(int64_t begin, int64_t end, int64_t grain_size,
    const std::function<void(int64_t, int64_t)> &func);

/**
 * @brief barrier in parallel region
 */
kutacc_export void parallel_barrier();

/**
 * @brief launch global parallel when using kupl-global backend
 *
 * @param [in] f          the function for global parallel
 */
kutacc_export void global_parallel_launch(const std::function<void()> &f);

/**
 * @brief handle type of async thread group
 */
typedef struct kuasync_exec *async_tg_h;

/**
 * @brief submit task to async thread group
 *
 * @param [in] executor                      handle of async thread group
 * @param [in] routine                       function pointer of the task to run
 * @param [in] args                          arguments for the task function
 */
kutacc_export void async_tg_task_submit(async_tg_h executor, void (*routine)(void *), void *args);

/**
 * @brief wait for all running tasks to finish
 *
 * @param [in] exec                          handle of async thread group
 */
kutacc_export void async_tg_wait(async_tg_h exec);

/**
 * @brief create a new async thread group
 *
 * @param [out] executor                     handle of created thread group
 * @param [in] num_threads                   total number of worker threads
 * @param [in] core_offset_in_numa           core offset inside numa node for thread binding
 * @return int                               return 1 if success, 0 if failed
 */
kutacc_export int async_tg_create(async_tg_h &executor, int num_threads, int core_offset_in_numa);

/**
 * @brief destroy async thread group and release resources
 *
 * @param [in, out] executor                 handle of thread group to destroy
 */
kutacc_export void async_tg_destroy(async_tg_h &executor);

/**
 * @brief get total thread count of current thread group
 *
 * @return int                               total number of threads in group
 */
kutacc_export int async_tg_get_thread_num();

/**
 * @brief get group id of current worker thread
 *
 * @return int                               unique id of current thread
 */
kutacc_export int async_tg_get_thread_id();

/**
 * @brief Multiply input tensor by a scalar and optionally add to output tensor.
 *
 * For each element: o_ptr[i] = (load_output ? o_ptr[i] : 0) + alpha * i_ptr[i].
 * The operation is performed in-place on o_ptr.
 *
 * @param [in] i_ptr       Input tensor (bfloat16_t).
 * @param [in,out] o_ptr   Output tensor (bfloat16_t). Written in-place.
 * @param [in] num         Number of elements to process.
 * @param [in] alpha       Scalar multiplier.
 * @param [in] load_output If true, load existing output value before adding; otherwise start from zero.
 */
kutacc_export void mul_scalar_add(bfloat16_t *i_ptr, bfloat16_t *o_ptr, int64_t num, float alpha, bool load_output);

/**
 * @brief Apply scaling, softmax, and convert back to bfloat16 over each row of activation.
 *
 * For each row i (height), the row is scaled by scale[i], then softmax is applied,
 * and the result is stored back into act. A temporary buffer buf is used per thread.
 *
 * @param [in,out] act        Activation tensor of shape (height, width) in bfloat16.
 * @param [in] act_stride     Stride between rows of act.
 * @param [in] scale          Per‑row scale factor (float array of length height).
 * @param [out] buf           Temporary buffer (float) of size at least width per thread.
 * @param [in] buf_stride     Stride between per‑thread buffers in buf.
 * @param [in] height         Number of rows.
 * @param [in] width          Number of columns per row.
 */
kutacc_export void scale_softmax(__bf16 *act, int64_t act_stride, float *scale, float *buf, int64_t buf_stride,
    int64_t height, int64_t width);

/**
 * @brief Apply rotary position embedding (RoPE) to query and key tensors.
 *
 * For each token, the corresponding cos/sin values from the cache are used to
 * rotate the embedding of each head. The result is written to q_out and k_out.
 *
 * @param [in] position_ids  1D tensor of token position indices (int64_t).
 * @param [in] q             3D query tensor (tokens, heads, head_size).
 * @param [in] k             3D key tensor (tokens, heads, head_size).
 * @param [out] q_out        3D output query tensor, same shape as q.
 * @param [out] k_out        3D output key tensor, same shape as k.
 * @param [in] cos_sin_cache 2D cache tensor (max_seq_len, head_size) storing concatenated cos and sin values.
 */
kutacc_export void rope(
    const Tensor<int64_t, 1>& position_ids,
    const Tensor<bfloat16_t, 3>& q,
    const Tensor<bfloat16_t, 3>& k,
    const Tensor<bfloat16_t, 3>& q_out,
    const Tensor<bfloat16_t, 3>& k_out,
    const Tensor<bfloat16_t, 2>& cos_sin_cache
);

/**
 * @brief Apply SiLU (Swish) activation to the gate part, multiply by the up part,
 *        then quantize the result to int8 with per‑row online scaling.
 *
 * The input gateup_data is expected to interleave gate and up values: for each row,
 * the first half (gate) is transformed by SiLU and multiplied by the second half (up).
 * The result is quantized to int8 using per‑row scale computed from the absolute maximum.
 *
 * @param [in] gateupsize       Number of columns in each row (total width = 2 * gateupsize).
 * @param [in] gateupnumel      Total number of elements in gateup_data (height * width).
 * @param [out] outdata_ptr     Output int8 tensor, shape (height, gateupsize).
 * @param [in,out] gateupdata_ptr  Input bfloat16 tensor (height, width). The second half is overwritten with intermediate results.
 * @param [out] online_scalesdata_ptr  Per‑row scale factor (float array of length height).
 */
kutacc_export void silu_mul_quant(int64_t gateupsize, int64_t gateupnumel, int8_t *outdata_ptr,
    bfloat16_t *gateupdata_ptr, float *online_scalesdata_ptr);


// --------------------------- AlphaFold3 ---------------------------
/**
 * @brief MSA attention core SME compute kernel
 *
 * @param [in] logits     Attention score logits [h, q, k], float
 * @param [in] v          Value feature input [b, h, k, c], bfloat16
 * @param [out] v_avg     Weighted average value output [b, h, q, c], bfloat16
 * @param [in] h          Number of attention heads
 * @param [in] q          Query sequence length
 * @param [in] k          Key sequence length
 * @param [in] b          Batch size
 * @param [in] c          Per-head hidden channel dimension
 */
kutacc_export void msa_attention_sme(float* logits, bfloat16_t* v, bfloat16_t* v_avg, int h, int q, int k, int b, int c);

/**
 * @brief MSA attention linear projection kernel
 *
 * @param [in] act             Input feature [batch, seq_len, nchannels]
 * @param [in] pair_act        Pair act input feature [output_batches, seq_len, noutput_channels]
 * @param [out] pair_logits    Pair logits buffer [output_batche, seq_len, nheads]
 * @param [out] v              Value feature [batch, seq_len, nchannels]
 * @param [out] gate           Gating feature [batch, seq_len, nchannels]
 * @param [in] pair_logits_w   Weight for pair logits [noutput_channels, nheads]
 * @param [in] value_w         Value projection weight [nchannels, nchannels]
 * @param [in] gating_w        Gating projection weight [nchannels, nchannels]
 * @param [in] batch           Batch dimension size
 * @param [in] seq_len         Sequence token length
 * @param [in] nchannels       Input hidden dimension
 * @param [in] nheads          Attention head count
 * @param [in] noutput_channels Output hidden channels per head
 * @param [in] output_batches  Split output batch partition size
 */
kutacc_export void msa_attention_linear(bfloat16_t *act, bfloat16_t *pair_act, bfloat16_t *pair_logits, bfloat16_t *v, bfloat16_t *gate, 
    bfloat16_t *pair_logits_w, bfloat16_t *value_w, bfloat16_t *gating_w, int64_t batch, int64_t seq_len, int64_t nchannels, int64_t nheads, 
    int64_t noutput_channels, int64_t output_batches);

/**
 * @brief MSA attention output linear and sigmoid kernel
 *
 * @param [in] v          Weighted value feature [batch, seq_len, nchannels]
 * @param [in] v_avg      Weighted average value [batch, seq_len, nchannels]
 * @param [in] gate       Gating activation [batch, seq_len, nchannels]
 * @param [out] out       Final fused MSA output [batch, seq_len, nchannels]
 * @param [in] batch      Batch size
 * @param [in] seq_len    Sequence length
 * @param [in] nchannels  Hidden channel dimension
 */
kutacc_export void msa_attention_out(bfloat16_t *v, bfloat16_t *v_avg, bfloat16_t *gate, bfloat16_t *out, int64_t batch, int64_t seq_len, int64_t nchannels);

/**
 * @brief Transpose tensor B N H D -> B H N D
 *
 * @param [in] in  Input tensor shape [B, N, H, D]
 * @param [out] o  Output tensor shape [B, H, N, D]
 * @param [in] B   Batch size
 * @param [in] H   Head / second dimension
 * @param [in] N   Token / first inner dimension
 * @param [in] D   Channel dim (fixed 1 for this transpose kernel)
 */
kutacc_export void mha_transpose_s_h(bfloat16_t* in, bfloat16_t* o, int B, int H, int N, int D);

/**
 * @brief Linear projection without bias, support packed input weight
 *
 * @tparam T Output buffer data type
 * @param [in] xptr         Input feature buffer, [bucket, hidden_size]
 * @param [in] wptr         Linear weight matrix [hidden_size, num_channels]
 * @param [out] resultptr   Linear output buffer [bucket, num_channels]
 * @param [in] bucket       Block bucket partition size
 * @param [in] num_channels Input feature channels
 * @param [in] hidden_size  Output hidden dimension
 * @param [in] x_packed     Flag whether input matrix is pre-packed
 */
template <typename T>
kutacc_export void linear_nobias(bfloat16_t* xptr, bfloat16_t* wptr, T* resultptr, int bucket, int num_channels, int hidden_size, bool x_packed);

/**
 * @brief Linear projection with bias + sigmoid activation
 *
 * @param [in] single_cond_ptr Input feature [bucket, hidden_size]
 * @param [in] wptr            Linear weight matrix [hidden_size, num_channels]
 * @param [in] bptr            Bias vector (float) [num_channels]
 * @param [in] resultptr       Float intermediate output before cast [samples, bucket, num_channels]
 * @param [out] resultptr_bf16 Bfloat16 cast final output [samples, bucket, num_channels]
 * @param [in] samples         Total sample count
 * @param [in] bucket          Block bucket partition size
 * @param [in] num_channels    Input feature channels
 * @param [in] hidden_size     Hidden output dimension
 */
kutacc_export void linear_bias_sigmoid(bfloat16_t* single_cond_ptr, bfloat16_t* wptr, float* bptr, float* resultptr, bfloat16_t* resultptr_bf16,
    int samples, int bucket, int num_channels, int hidden_size);

/**
 * @brief LayerNorm without affine weight/bias params, float input/output
 *
 * @param [in,out] x    Float feature tensor [B, M, N]
 * @param [in] B        Batch size
 * @param [in] M        Sequence length
 * @param [in] N        Hidden channel dimension
 * @param [in] eps      Numerical stable epsilon for variance
 */
kutacc_export void layernorm_noparams(float* x, int B, int M, int N, float eps);

/**
 * @brief LayerNorm + cast to bfloat16 with pack transform
 *
 * @param [in] x          Float input feature [M, N]
 * @param [out] x_r       Packed bfloat16 normalized feature [M, N]
 * @param [in] ln_weight  LayerNorm scale weight
 * @param [in] ln_bias    LayerNorm shift bias
 * @param [in] M          Sequence length
 * @param [in] N          Hidden channel dim
 * @param [in] eps        Variance stable epsilon
 */
kutacc_export void layernorm_pack_bf16(float* x, bfloat16_t* x_r, float* ln_weight, float* ln_bias, int M, int N, float eps);

/**
 * @brief Adaptive layer norm merge conditional scale/shift + optional pack
 *
 * @param [in] x         Raw float input feature
 * @param [in] s_c       Conditional scale bias bfloat16
 * @param [in] s_w       Conditional scale weight bfloat16
 * @param [in] s_b       Conditional shift float bias
 * @param [in] c_b2      Extra conditional bias term
 * @param [in] B         Batch size
 * @param [in] M         Sequence length
 * @param [in] D         Main hidden dimension
 * @param [in] D1        Conditional branch hidden dimension
 * @param [out] pack_x   Packed normalized output
 * @param [in] dopack    Flag whether output matrix need packed
 */
kutacc_export void adpln_merge_bf16(float* x, bfloat16_t* s_c, bfloat16_t* s_w, float* s_b, bfloat16_t* c_b2, int B, int M, int D, int D1, bfloat16_t* pack_x, bool dopack);

/**
 * @brief Cross attention pair logits block packing generic template
 *
 * @tparam T Pair logits original buffer type
 * @param [in] pair_logits_ori Raw pair bias logits input
 * @param [out] pair_logits_r  Packed float logits buffer
 * @param [in] nblock_q Query side block tile count
 * @param [in] nblock_kv Key-value side block tile count
 * @param [in] N_a      Auxiliary sequence dimension
 * @param [in] H        Attention head count
 * @param [in] Nq       Query token length
 * @param [in] Nkv      Key-value token length
 */
template <typename T>
kutacc_export void cross_pack_pair_logits(T pair_logits_ori, float* pair_logits_r, int nblock_q, int nblock_kv, int N_a, int H, int Nq, int Nkv);

/**
 * @brief Self attention pair logits block packing
 *
 * @param [in] pair_logits_ori Raw pair bias logits bfloat16 input
 * @param [out] pair_logits_r  Packed float logits buffer
 * @param [in] nblock_q Query tile block number
 * @param [in] nblock_kv Key-value tile block number
 * @param [in] H        Attention head count
 * @param [in] Nq       Query token length
 * @param [in] Nkv      Key-value token length
 */
kutacc_export void self_pack_pair_logits(bfloat16_t* pair_logits_ori, float* pair_logits_r, int nblock_q, int nblock_kv, int H, int Nq, int Nkv);

/**
 * @brief Grid self attention pair logits block packing
 *
 * @param [in] pair_logits_ori Raw pair bias logits bfloat16 input
 * @param [out] pair_logits_r  Packed float logits buffer
 * @param [in] nblock_q Query tile block number
 * @param [in] nblock_kv Key-value tile block number
 * @param [in] H        Attention head count
 * @param [in] Nq       Query token length
 * @param [in] Nkv      Key-value token length
 */
kutacc_export void grid_pack_pair_logits(bfloat16_t* pair_logits_ori, float* pair_logits_r, int nblock_q, int nblock_kv, int H, int Nq, int Nkv);

/**
 * @brief Grid flash attention with mask/bias, head dim = 16
 *
 * @param [in] B        Batch size
 * @param [in] H        Attention head count
 * @param [in] N        Sequence token length
 * @param [in] D        Per-head hidden dimension
 * @param [in] q        Query feature [B,H,N,D]
 * @param [in] k        Key feature [B,H,N,D]
 * @param [in] v        Value feature [B,H,N,D]
 * @param [out] out     Attention weighted sum output
 * @param [in] mask     Float attention mask buffer
 * @param [in] bias_r   Preprocessed pair bias float buffer
 * @param [in] scale    Attention logit scaling factor
 * @param [in] m_min    Mask minimum clamp threshold
 */
kutacc_export void grid_flash_attn_mask_bias_bf16_hd16(int B, int H, int N, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, 
    float* mask, float* bias_r, float scale, float m_min);

/**
 * @brief Self flash attention with pair bias, head dim = 24
 *
 * @param [in] B        Batch size
 * @param [in] H        Attention head count
 * @param [in] Nq       Query token length
 * @param [in] Nkv      Key-value token length
 * @param [in] D        Per-head hidden dimension
 * @param [in] q        Query feature buffer
 * @param [in] k        Key feature buffer
 * @param [in] v        Value feature buffer
 * @param [out] out     Attention output
 * @param [in] bias     Raw pair bias float buffer
 * @param [in] pair_logits_r Packed pair logits float buffer
 * @param [in] scale    Logit scale factor
 * @param [in] m_min    Mask lower clamp limit
 */
kutacc_export void self_flash_attn_mask_bias_bf16_hd24(int B, int H, int Nq, int Nkv, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, 
    float* bias, float* pair_logits_r, float scale, float m_min);

/**
 * @brief Cross flash attention with pair bias, head dim = 32
 *
 * @param [in] B        Batch size
 * @param [in] N_a      Auxiliary sequence dimension
 * @param [in] H        Attention head count
 * @param [in] Nq       Query token length
 * @param [in] Nkv      Key-value token length
 * @param [in] D        Per-head hidden dimension
 * @param [in] q        Query feature buffer
 * @param [in] k        Key feature buffer
 * @param [in] v        Value feature buffer
 * @param [out] out     Cross attention output
 * @param [in] bias     Raw pair bias float buffer
 * @param [in] pair_logits_r Packed pair logits float buffer
 * @param [in] scale    Logit scaling factor
 * @param [in] m_min    Mask clamp minimum value
 */
kutacc_export void cross_flash_attn_mask_bias_bf16_hd32(int B, int N_a, int H, int Nq, int Nkv, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, 
    float* bias, float* pair_logits_r, float scale, float m_min);

/**
 * @brief Grid flash attention mask+bias kernel head dim=32
 *
 * @param [in] B        Batch size
 * @param [in] H        Attention head count
 * @param [in] N        Sequence token length
 * @param [in] D        Per-head hidden dimension
 * @param [in] q        Query feature
 * @param [in] k        Key feature
 * @param [in] v        Value feature
 * @param [out] out     Attention output
 * @param [in] mask     Attention mask float buffer
 * @param [in] bias_r   Preprocessed pair bias float buffer
 * @param [in] scale    Logit scale factor
 * @param [in] m_min    Mask lower bound clamp
 */
kutacc_export void grid_flash_attn_mask_bias_bf16_hd32(int B, int H, int N, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, 
    float* mask, float* bias_r, float scale, float m_min);

/**
 * @brief Self flash attention mask+bias kernel head dim=48
 *
 * @param [in] B        Batch size
 * @param [in] H        Attention head count
 * @param [in] Nq       Query token length
 * @param [in] Nkv      Key-value token length
 * @param [in] D        Per-head hidden dimension
 * @param [in] q        Query feature buffer
 * @param [in] k        Key feature buffer
 * @param [in] v        Value feature buffer
 * @param [out] out     Attention output buffer
 * @param [in] bias     Raw pair bias float buffer
 * @param [in] pair_logits_r Packed pair logits float buffer
 * @param [in] scale    Logit scaling factor
 * @param [in] m_min    Mask clamp minimum threshold
 */
kutacc_export void self_flash_attn_mask_bias_bf16_hd48(int B, int H, int Nq, int Nkv, int D, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* out, 
    float* bias, float* pair_logits_r, float scale, float m_min);

/**
 * @brief Pack linear weight matrix for GLU kernel
 *
 * @param [in] weight      Raw weight matrix
 * @param [out] packed_weight Pre-packed weight tile buffer
 * @param [in] M           Output channel dim
 * @param [in] N           Input channel dim
 */
kutacc_export void glu_pack_weight_bf16(bfloat16_t* weight, bfloat16_t* packed_weight, int M, int N);

/**
 * @brief Pack input activation matrix for GLU left operand
 *
 * @param [in] matA        Raw input feature matrix
 * @param [out] packed_matA Packed tile buffer
 * @param [in] M           Row dimension
 * @param [in] N           Column dimension
 */
kutacc_export void glu_pack_left_bf16(bfloat16_t* matA, bfloat16_t* packed_matA, int M, int N);

/**
 * @brief GLU linear + sigmoid activation
 *
 * @param [in] xptr        Input feature buffer
 * @param [in] wptr        Linear weight float buffer
 * @param [in] maskptr     Mask condition bfloat16 buffer
 * @param [out] resultptr   GLU output buffer
 * @param [in] Bn          Batch block partition size
 * @param [in] Nb          Token block partition size
 * @param [in] num_channels Hidden channel dimension
 */
kutacc_export void glu_sigmoid(bfloat16_t* xptr,float* wptr,bfloat16_t* maskptr,bfloat16_t* resultptr, int Bn, int Nb, int num_channels);

/**
 * @brief GLU mask permute after matrix multiplication
 *
 * @param [in] packed_x    Packed input feature
 * @param [in] packed_weight Packed linear weight
 * @param [out] o          Output permuted feature
 * @param [in] maskptr    Mask condition buffer
 * @param [in] Bn         Batch block size
 * @param [in] Nb         Token block size
 * @param [in] C          Main hidden channel
 * @param [in] C1         Split hidden channel for gate
 */
kutacc_export void glu_mask_permute(bfloat16_t* packed_x, bfloat16_t* packed_weight, bfloat16_t* o, float* maskptr, int Bn, int Nb, int C, int C1);

/**
 * @brief GLU swish activation two branch linear projection template
 *
 * @tparam x_t Input feature type
 * @tparam r_t Output result type
 * @tparam w1_t First branch weight type
 * @tparam w2_t Second branch weight type
 * @param [in] xptr         Input feature buffer
 * @param [out] resultptr   Swish fused output
 * @param [in] shape0       Outer batch/token dimension
 * @param [in] bucket       Block bucket tile size
 * @param [in] num_channels Input hidden channels
 * @param [in] num_intermediate GLU intermediate split channels
 * @param [in] w1           First branch linear weight
 * @param [in] w2           Second branch linear weight
 * @param [in] hasbatch     Flag whether batch outer dim exists
 */
template<typename x_t, typename r_t, typename w1_t, typename w2_t>
kutacc_export void glu_swish(x_t* xptr, r_t* resultptr, int shape0, int bucket, int num_channels, int num_intermediate, w1_t* w1, w2_t* w2, bool hasbatch);

/**
 * @brief Self attention weighted average + gate fusion with optional pack
 *
 * @param [in] weighted_avg Attention weighted sum feature
 * @param [in] gate         Gating activation
 * @param [out] pack_wavg   Packed fused output
 * @param [in] ndim         Hidden channel dimension
 * @param [in] batch        Batch size
 * @param [in] seq_len      Sequence token length
 * @param [in] nchannels    Input hidden channels
 * @param [in] dopack       Enable weight/feature pack transform
 */
kutacc_export void self_attention_out(bfloat16_t* weighted_avg, bfloat16_t* gate, bfloat16_t* pack_wavg, int ndim, int batch, int seq_len, int nchannels, bool dopack);

/**
 * @brief Self attention Q/K/V/G linear projection + transpose, no single_cond
 *
 * @param [in] x             Input feature [batch, seq_len, nheads * head_size]
 * @param [in] q_proj        Q projection weight [nheads * head_size, nheads * head_size]
 * @param [in] q_proj_bias   Q bias float buffer [nheads * head_size]
 * @param [in] k_proj        K projection weight [nheads * head_size, nheads * head_size]
 * @param [in] v_proj        V projection weight [nheads * head_size, nheads * head_size]
 * @param [in] g_proj        Gate projection weight [nheads * head_size, nheads * head_size]
 * @param [out] q            Transposed Q feature[batch, nheads, seq_len, head_size]
 * @param [out] k            Transposed K feature[batch, nheads, seq_len, head_size]
 * @param [out] v            Transposed V feature[batch, nheads, seq_len, head_size]
 * @param [out] g            gate feature[batch, seq_len, nheads, head_size]
 * @param [in] batch         Batch size
 * @param [in] seq_len       Sequence token length
 * @param [in] nheads        Attention head count
 * @param [in] head_size     Per-head hidden dimension
 */
kutacc_export void self_attention_linear_transpose_nosingle(bfloat16_t* x, bfloat16_t* q_proj, float* q_proj_bias, bfloat16_t* k_proj, bfloat16_t* v_proj, bfloat16_t* g_proj,
     bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* g, int batch, int seq_len, int nheads, int head_size);

/**
 * @brief Self attention Q/K/V/G linear projection + transpose
 *
 * @param [in] x             Input feature [batch, seq_len, nheads * head_size]
 * @param [in] q_proj        Q projection weight [nheads * head_size, nheads * head_size]
 * @param [in] q_proj_bias   Q bias float buffer [nheads * head_size]
 * @param [in] k_proj        K projection weight [nheads * head_size, nheads * head_size]
 * @param [in] v_proj        V projection weight [nheads * head_size, nheads * head_size]
 * @param [in] g_proj        Gate projection weight [nheads * head_size, nheads * head_size]
 * @param [out] q            Transposed Q feature[batch, nheads, seq_len, head_size]
 * @param [out] k            Transposed K feature[batch, nheads, seq_len, head_size]
 * @param [out] v            Transposed V feature[batch, nheads, seq_len, head_size]
 * @param [out] g            Gate feature[batch, seq_len, nheads, head_size]
 * @param [in] batch         Batch size
 * @param [in] seq_len       Sequence token length
 * @param [in] nheads        Attention head count
 * @param [in] head_size     Per-head hidden dimension
 */
kutacc_export void self_attention_linear_transpose(bfloat16_t* x, bfloat16_t* q_proj, float* q_proj_bias, bfloat16_t* k_proj, bfloat16_t* v_proj, bfloat16_t* g_proj,
     bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* g, int batch, int seq_len, int nheads, int head_size);

/**
 * @brief Cross attention weighted average + gate fusion with pack transform
 *
 * @param [in] weighted_avg Cross-attn weighted value sum
 * @param [in] gate         Gating activation
 * @param [out] pack_wavg   Packed fused output feature
 * @param [in] ndim         Hidden channel dimension
 * @param [in] batch        Query batch size
 * @param [in] seq_len      Query sequence length
 * @param [in] n_res        Key-value sequence length
 * @param [in] nchannels    Input hidden channels
 */
kutacc_export void cross_attention_out_pack(bfloat16_t* weighted_avg, bfloat16_t* gate, bfloat16_t* pack_wavg, int ndim, int batch, int seq_len, int n_res, int nchannels);

/**
 * @brief Cross attention Q/K/V/G linear projection + transpose kernel
 *
 * @param [in] x_q           Query side input feature [batch_q, seq_len_q, nheads * head_size]
 * @param [in] x_k           Key-value side input feature [batch_k, seq_len_k, nheads * head_size]
 * @param [in] q_proj        Q projection weight [nheads * head_size, nheads * head_size]
 * @param [in] q_proj_bias   Q bias float buffer [nheads * head_size]
 * @param [in] k_proj        K projection weight [nheads * head_size, nheads * head_size]
 * @param [in] v_proj        V projection weight [nheads * head_size, nheads * head_size]
 * @param [in] g_proj        Gate projection weight [nheads * head_size, nheads * head_size]
 * @param [out] q            Transposed Q feature [batch_q, nheads, seq_len_q, head_size]
 * @param [out] k            Transposed K feature [batch_k, nheads, seq_len_k, head_size]
 * @param [out] v            Transposed V feature [batch_k, nheads, seq_len_k, head_size]
 * @param [out] g            gate feature [batch_q, seq_len_q, nheads, head_size]
 * @param [in] batch_q       Query batch size
 * @param [in] seq_len_q     Query token length
 * @param [in] batch_k       Key-value batch size
 * @param [in] seq_len_k     Key-value token length
 * @param [in] nheads        Attention head count
 * @param [in] head_size     Per-head hidden dimension
 */
kutacc_export void cross_attention_linear_transpose(bfloat16_t* x_q, bfloat16_t* x_k, bfloat16_t* q_proj, float* q_proj_bias, bfloat16_t* k_proj, 
    bfloat16_t* v_proj, bfloat16_t* g_proj, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* g, int batch_q, int seq_len_q, int batch_k, int seq_len_k, int nheads, int head_size);

/**
 * @brief Boolean mask cast to float attention mask
 *
 * @param [in] mask      Bool mask buffer [B,N]
 * @param [out] mask_data Float converted mask buffer
 * @param [in] B         Batch size
 * @param [in] N         Sequence token length
 */
kutacc_export void mask_b2f(bool* mask, float* mask_data, int B, int N);

/**
 * @brief Grid self attention LayerNorm + cast + pack bfloat16
 *
 * @param [in] x          Float raw input feature
 * @param [out] x_bf16    Bfloat16 cast intermediate
 * @param [out] x_r       Packed normalized feature
 * @param [in] ln_weight  LN scale weight
 * @param [in] ln_bias    LN shift bias
 * @param [in] batch      Batch size
 * @param [in] seq_len    Sequence token length
 * @param [in] nchannels  Hidden channel dim
 * @param [in] eps        Variance stable epsilon
 */
kutacc_export void gridself_layernorm_bias_bf16_pack(float* x, bfloat16_t* x_bf16, bfloat16_t* x_r, float* ln_weight,  float* ln_bias, int batch, int seq_len, int nchannels, float eps);

/**
 * @brief Grid self attention weighted average + gate fuse output
 *
 * @param [in] weights     Attention score softmax weight
 * @param [in] weighted_avg Value weighted sum feature
 * @param [in] gate        Gating activation
 * @param [out] out        Final fused output
 * @param [in] batch       Batch size
 * @param [in] seq_len     Sequence token length
 * @param [in] nchannels   Hidden channel dimension
 */
kutacc_export void gridself_attention_out(bfloat16_t* weights, bfloat16_t* weighted_avg, bfloat16_t* gate,  bfloat16_t* out, int batch, int seq_len, int nchannels);

/**
 * @brief Grid attention pair bias linear projection
 *
 * @param [in] input      Pair bias input feature
 * @param [in] bias_weight Bias projection weight
 * @param [out] bias      Output pair bias logits
 * @param [in] batch      Batch size
 * @param [in] seq_len    Sequence token length
 * @param [in] nchannels  Hidden channel dim
 * @param [in] nheads     Attention head count
 */
kutacc_export void gridself_attention_bias_linear(bfloat16_t* input, bfloat16_t* bias_weight, bfloat16_t* bias, int batch, int seq_len, int nchannels, int nheads);

/**
 * @brief Grid self attention Q/K/V/G/pair bias linear + transpose, support transpose flag
 *
 * @param [in] x             Raw input feature [batch, seq_len, nheads, head_size](trans = false) / [seq_len, batch, nheads, head_size](trans = true)
 * @param [in] pack_x        Packed input buffer [batch, seq_len, nheads, head_size](trans = false) / [seq_len, batch, nheads, head_size](trans = true)
 * @param [in] q_proj        Q projection weight [nheads*head_size, nheads*head_size]
 * @param [in] k_proj        K projection weight [nheads*head_size, nheads*head_size]
 * @param [in] v_proj        V projection weight [nheads*head_size, nheads*head_size]
 * @param [in] g_proj        Gate projection weight [nheads*head_size, nheads*head_size]
 * @param [in] pbias_proj    Pair bias projection weight [nheads*head_size, nheads]
 * @param [out] q            Transposed Q feature[batch, nheads, seq_len, head_size]
 * @param [out] k            Transposed K feature[batch, nheads, seq_len, head_size]
 * @param [out] v            Transposed V feature[batch, nheads, seq_len, head_size]
 * @param [out] g            Gate feature[batch, seq_len, nheads, head_size]
 * @param [out] pair_bias    pair bias [batch, seq_len, nheads]
 * @param [in] batch         Batch size
 * @param [in] seq_len       Sequence token length
 * @param [in] nheads        Attention head count
 * @param [in] head_size     Per-head hidden dim
 * @param [in] trans         Flag whether input matrix and Q/K/V/G is transposed in dim0 and dim1
 */
kutacc_export void gridself_attention_linear_transpose_trans(bfloat16_t* x, bfloat16_t* pack_x, bfloat16_t* q_proj, bfloat16_t* k_proj, bfloat16_t* v_proj, bfloat16_t* g_proj,
bfloat16_t* pbias_proj, bfloat16_t* q, bfloat16_t* k, bfloat16_t* v, bfloat16_t* g, bfloat16_t* pair_bias, int batch, int seq_len, int nheads, int head_size, bool trans);

/**
 * @brief BGEMM weight matrix tile packing bfloat16
 *
 * @param [in] weight      Raw weight matrix
 * @param [out] packed_weight Pre-packed tile buffer for SME BGEMM
 * @param [in] M           Matrix dim0 shape 
 * @param [in] N           Matrix dim1 shape
 */
kutacc_export void bgemm_pack_weight_bf16(bfloat16_t *weight,bfloat16_t *packed_weight,int M, int N);

/**
 * @brief GLU float weight packing template for different output type
 *
 * @tparam w_t Packed weight buffer data type
 * @param [in] wptr        Raw float weight matrix
 * @param [out] packed_weight Tile packed weight output
 * @param [in] num_channels Input hidden channels
 * @param [in] num_intermediate GLU split intermediate channels
 * @param [in] hasbatch    Outer batch dimension exists flag
 */
template<typename w_t>
kutacc_export void glu_pack_weight(float *wptr, w_t *packed_weight,int num_channels, int num_intermediate, bool hasbatch);

/**
 * @brief Pairformer transition block feed-forward output fusion
 *
 * @param [in] act      Hidden activation feature
 * @param [in] weights  Output projection weight
 * @param [out] out     Transition block final output
 * @param [in] batch    Batch size
 * @param [in] seq_len  Sequence token length
 * @param [in] nchannels Hidden channel dimension
 * @param [in] nheads   Attention head count
 */
kutacc_export void transitionblock_out(bfloat16_t* act, bfloat16_t* weights, bfloat16_t* out, int batch, int seq_len, int nchannels, int nheads);

/**
 * @brief Triangle multiplication input kernel packing einsum(...cik,...cjk->...cij) layout
 *
 * @param [in] a         Raw input matrix
 * @param [out] packed_a Packed tile buffer
 * @param [in] B         Batch dimension
 * @param [in] N         Sequence inner dimension
 */
kutacc_export void pack_kernel_eq1(bfloat16_t* a, bfloat16_t* packed_a, int B, int N);

/**
 * @brief Einsum(...cik,...cjk->...cij) compute kernel for triangle multiplication
 *
 * @param [in] aptr Packed input A matrix
 * @param [in] bptr Packed input B matrix
 * @param [out] o   Einsum output feature
 * @param [in] C    Channel dimension
 * @param [in] B    Batch dimension
 * @param [in] N    Sequence inner dimension
 */
kutacc_export void einsum_eq1(bfloat16_t* aptr, bfloat16_t* bptr, bfloat16_t* o, int C, int B, int N);

/**
 * @brief Triangle multiplication input kernel packing einsum(...ckj,...cki->...cij) layout
 *
 * @param [in] a         Raw input matrix
 * @param [out] packed_a Packed tile buffer eq2 layout
 * @param [in] N         First inner dimension
 * @param [in] B         Batch dimension
 */
kutacc_export void pack_kernel_eq2(bfloat16_t* a, bfloat16_t* packed_a, int N, int B);

/**
 * @brief Einsum(...ckj,...cki->...cij) compute kernel for triangle multiplication
 *
 * @param [in] aptr Packed input A matrix
 * @param [in] bptr Packed input B matrix
 * @param [out] o   Einsum output feature
 * @param [in] C    Channel dimension
 * @param [in] B    Batch dimension
 * @param [in] N    Sequence inner dimension
 */
kutacc_export void einsum_eq2(bfloat16_t* aptr, bfloat16_t* bptr, bfloat16_t* o, int C, int B, int N);

/**
 * @brief Triangle multiplication fused linear + sigmoid kernel
 *
 * @param [in] input      Raw input feature
 * @param [in] input_act  Preprocessed activation input
 * @param [in] output_w   Output projection weight
 * @param [in] gating_w   Gate projection weight
 * @param [out] output    Triangle multiplication main output
 * @param [in] gate       Gate temp buffer
 * @param [in] c_0        First channel partition size
 * @param [in] c_1        Second channel partition size
 * @param [in] c_2        Third channel partition size
 */
kutacc_export void trianglemultiplication(bfloat16_t* input, bfloat16_t* input_act, bfloat16_t* output_w, bfloat16_t* gating_w, bfloat16_t* output, bfloat16_t* gate, 
    int64_t c_0, int64_t c_1, int64_t c_2);

/**
 * @brief Shared memory all2all distributed data exchange , only support 8 processes
 *
 * @tparam data_t Element data type of buffer
 * @param [in] src             Source local data buffer
 * @param [out] out_ptr        Destination out buffer
 * @param [in] m_full          Full outer dimension size
 * @param [in] n_full          Full inner dimension size
 * @param [in] inner_size      Single slice inner element count
 * @param [in] size0           Local partition dim0 size
 * @param [in] size1           Local partition dim1 size
 * @param [in] input_stride0   Source tensor dim0 stride
 * @param [in] input_stride1   Source tensor dim1 stride
 * @param [in] output_stride0  Output tensor dim0 stride
 * @param [in] output_stride1  Output tensor dim1 stride
 * @param [in] m_rank          Current process rank
 * @param [in] m_buffer_size   Per-rank recv buffer element count
 * @param [in] m_kupl_recvbuf  Raw shared memory receive buffer ptr
 * @param [in] m_recvbuf_win   Kupl shm window handle
 */
template <typename data_t>
kutacc_export void shm_all2all(const data_t* src, data_t* out_ptr, int64_t m_full, int64_t n_full, int64_t inner_size, 
    int64_t size0, int64_t size1, int64_t input_stride0, int64_t input_stride1, int64_t output_stride0, int64_t output_stride1,
    int64_t m_rank, int64_t m_buffer_size, void * m_kupl_recvbuf, kupl_shm_win_h m_recvbuf_win);

/**
 * @brief Shared memory allgather along dim1, only support 8 processes
 *
 * @tparam data_t Input buffer element type
 * @param [in] src         Local rank input data
 * @param [in] B           Global batch dimension size
 * @param [in] H_local     Local head partition size of current rank
 * @param [in] inner_dims_numel Per-element inner flattened count
 * @param [in] m_rank      Current process rank
 * @param [in] m_kupl_recvbuf Shm receive buffer pointer
 * @param [in] m_recvbuf_win Kupl shm window handle
 * @return uint8_t* Pointer to gathered full output buffer
 */
template <typename data_t>
kutacc_export uint8_t* shm_all_gather_dim1(const data_t* src, int64_t B, int64_t H_local, int64_t inner_dims_numel, 
    int64_t m_rank, void * m_kupl_recvbuf, kupl_shm_win_h m_recvbuf_win);

/**
 * @brief Shared memory allgather along dim0, only support 8 processes
 *
 * @tparam data_t Input element type
 * @param [in] src           Local rank input buffer
 * @param [in] local_elements Local element count of current rank
 * @param [in] elem_size     Single element byte size
 * @param [in] m_rank        Current MPI rank
 * @param [in] m_world_size  Total world process count
 * @param [in] m_kupl_recvbuf Shm receive buffer raw pointer
 * @param [in] m_recvbuf_win Kupl shared memory window handle
 * @return uint8_t* Pointer to concatenated gathered buffer
 */
template <typename data_t>
kutacc_export uint8_t* shm_all_gather(const data_t* src, int64_t local_elements, size_t elem_size, 
    int64_t m_rank, int64_t m_world_size, void * m_kupl_recvbuf, kupl_shm_win_h m_recvbuf_win);

/**
 * @brief Compute default block partition size for outer product mean chunk parallel
 *
 * @param [in] n_seq        Global sequence / batch dimension
 * @param [in] n_res_gather Gather partition token length
 * @param [out] left_block_size Output left tile block size
 * @param [out] right_block_size Output right tile block size
 */
kutacc_export void default_block_size(int64_t n_seq, int64_t n_res_gather, int64_t &left_block_size, int64_t &right_block_size);

/**
 * @brief OuterProductMean core compute kernel with bf16 input/output
 *
 * @param [in] input_act      LN normalized input feature [n_seq, n_res, c_m]
 * @param [in] left_proj_w    Left projection weight permute(1,0) shape [c_i, c_m]
 * @param [in] right_proj_w   Right projection weight permute(1,0) shape [c_i, c_m]
 * @param [out] left_proj     Temporary left projection buffer [c_i, n_res, n_seq]
 * @param [out] right_proj    Temporary right projection buffer [c_i, n_res, n_seq]
 * @param [out] left_proj_    Transformed left projection temp [n_res, c_i, n_seq]
 * @param [out] right_proj_   Transformed right projection temp [n_res, c_i, n_seq]
 * @param [out] right_proj_new Gather partition right projection [n_res_gather, c_i, n_seq]
 * @param [in] output_w       Post projection weight flatten [c_z, c_i*c_i]
 * @param [in] output_b       Post projection bias [c_z]
 * @param [out] out           Final output [n_res, n_res_gather, c_z]
 * @param [in] mask           Transposed mask [n_res, n_seq]
 * @param [out] norm          Mask cross product norm denominator [n_seq, n_seq]
 * @param [in] c_i            Intermediate channel dimension
 * @param [in] c_m            Input hidden channel dimension
 * @param [in] c_z            Output hidden channel dimension
 * @param [in] n_res          Local sequence token length
 * @param [in] n_seq          Batch size (original B=1024)
 * @param [in] n_res_gather   Global gather partition token length
 * @param [in] mask_bias      Mask slice offset for multi-rank partition
 * @param [in] mask_stride0   Mask tensor row stride
 * @param [in] left_block_size Parallel chunk tile size left side
 * @param [in] right_block_size Parallel chunk tile size right side
 */
kutacc_export void outer_product_mean(bfloat16_t* input_act, bfloat16_t* left_proj_w, bfloat16_t* right_proj_w, bfloat16_t* left_proj, bfloat16_t* right_proj, 
    bfloat16_t* left_proj_, bfloat16_t* right_proj_, bfloat16_t* right_proj_new, bfloat16_t* output_w, bfloat16_t* output_b, bfloat16_t* out, bfloat16_t* mask, 
    bfloat16_t* norm, int64_t c_i, int64_t c_m, int64_t c_z, int64_t n_res, int64_t n_seq, int64_t n_res_gather, int64_t mask_bias, int64_t mask_stride0, 
    int64_t left_block_size, int64_t right_block_size);

}  // namespace kutacc

#endif
#endif