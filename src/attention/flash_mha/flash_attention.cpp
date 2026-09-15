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

#include "attention.h"

namespace kutacc {

namespace {
template <typename scalar_t, size_t Dims>
void tensor_fill(Tensor<scalar_t, Dims>& tensor, scalar_t el)
{
    scalar_t* data = reinterpret_cast<scalar_t*>(tensor.data_ptr());
    std::fill(data, data + tensor.numel(), el);
}
    
inline void softmax_fusekernel(int q_start, int k_start, int64_t q_step, int64_t k_step, float* s_data,
    float softmax_scale, float* max_block_old_data, float* base_block_old_data, float* max_block_new_data,
    float* base_block_new_data, bfloat16_t* pack_v_data, int prefill_start, int64_t chunked_loop_index)
{
    const int64_t svls = svcntw();
    int64_t padding_k_step = (k_step + svls - 1) / svls * svls;
    int64_t padding_q_step = (q_step + svls - 1) / svls * svls;
    svbool_t pg_32_all = svptrue_b32();
    svbool_t pg_16_all = svptrue_b16();

    for (int64_t col_start = 0; col_start < padding_q_step; col_start += svls) {
        svbool_t pg_col = pg_16_all;  // svwhilelt_b32(col_start, q_step);
        svfloat32_t new_maxs = svdup_f32(-INFINITY);
        svfloat32_t reduce = svdup_f32(0.0f);
        svfloat32_t old_maxs = svld1(pg_col, (max_block_old_data + col_start));
        svfloat32_t old_reduces = svld1(pg_col, (base_block_old_data + col_start));
        svint32_t q_causal_value = svindex_s32(col_start + q_start + prefill_start, 1);
        bfloat16_t* prf_base = pack_v_data + (col_start / 16) * 16 * k_step + (k_start / k_step) * 128 * k_step;
#pragma unroll(8)
        for (int row_start = 0; row_start < padding_k_step; row_start++) {
            svfloat32_t values = svld1(pg_col, (s_data + row_start * padding_q_step + col_start));
            svprfw(pg_16_all, prf_base + (row_start + 0) * 16,
                svprfop::SV_PLDL1KEEP);  // BR==BC==128时S可以预取同样大小的V
            svprfw(pg_16_all, prf_base + (row_start + 4) * 16, svprfop::SV_PLDL1KEEP);
            // mask & softmax_scale & max
            svint32_t k_causal_value = svdup_n_s32_z(pg_col, row_start + chunked_loop_index + k_start);
            values = svsel(svcmpgt(pg_col, k_causal_value, q_causal_value), svdup_f32(-INFINITY), values);
            values = svmul_x(pg_col, values, softmax_scale);
            new_maxs = svmax_m(pg_col, new_maxs, values);
            svprfw(pg_16_all, prf_base + (row_start + 1) * 16, svprfop::SV_PLDL1KEEP);
            svprfw(pg_16_all, prf_base + (row_start + 5) * 16, svprfop::SV_PLDL1KEEP);
            svst1(pg_col, (s_data + row_start * padding_q_step + col_start), values);
        }

        new_maxs = svmax_m(pg_col, new_maxs, old_maxs);
        svst1(pg_col, (max_block_new_data + col_start), new_maxs);

        old_maxs = fast_exp(pg_col,
            svsub_m(pg_col, old_maxs, new_maxs));  // exp_(max_block_old_data - max_block_new_data)
        svst1(pg_col, (max_block_old_data + col_start), old_maxs);

#pragma unroll(2)
        for (int row_start = 0; row_start < padding_k_step; row_start += 2) {
            svfloat32_t values0 = svld1(pg_col, (s_data + row_start * padding_q_step + col_start));
            svfloat32_t values1 = svld1(pg_col, (s_data + (row_start + 1) * padding_q_step + col_start));

            svprfw(pg_16_all, prf_base + (row_start + 2) * 16, svprfop::SV_PLDL1KEEP);
            svprfw(pg_16_all, prf_base + (row_start + 6) * 16, svprfop::SV_PLDL1KEEP);

            values0 = fast_exp(pg_col, svsub_m(pg_col, values0, new_maxs));  // 1.0ms
            values1 = fast_exp(pg_col, svsub_m(pg_col, values1, new_maxs));

            reduce = svadd_m(pg_col, reduce, values0);
            reduce = svadd_m(pg_col, reduce, values1);

            svbfloat16_t values0_bf16 = svcvt_bf16_m(svbfloat16_t(), pg_col, values0);
            svbfloat16_t values1_bf16 = svcvt_bf16_m(svbfloat16_t(), pg_col, values1);
            svbfloat16_t values_mixed = svtrn1(values0_bf16, values1_bf16);

            svprfw(pg_16_all, prf_base + (row_start + 3) * 16, svprfop::SV_PLDL1KEEP);
            svprfw(pg_16_all, prf_base + (row_start + 7) * 16, svprfop::SV_PLDL1KEEP);

            svst1(pg_col, reinterpret_cast<bfloat16_t*>(s_data + row_start * padding_q_step + col_start), values_mixed);
        }

        reduce = svadd_m(pg_col, reduce, svmul_m(pg_col, old_reduces, old_maxs));
        svst1(pg_col, (base_block_new_data + col_start), reduce);
    }
}

void head_split_flash_attention_v2_sme_impl(
    const Tensor<bfloat16_t, 3> &(q),
    const Tensor<bfloat16_t, 3> &(k),
    const Tensor<bfloat16_t, 3> &(v),
    const Tensor<bfloat16_t, 3> &(out),
    const Tensor<bfloat16_t, 1> &(pack_q),
    const Tensor<bfloat16_t, 2> &(pack_k),
    const Tensor<bfloat16_t, 2> &(pack_v),
    const Tensor<float, 1> &(s),
    Tensor<float, 2> &(out_block_old),
    Tensor<float, 2> &(out_block_new),
    Tensor<float, 1> &(max_block_old),
    Tensor<float, 1> &(max_block_new),
    Tensor<float, 1> &(base_block_old),
    Tensor<float, 1> &(base_block_new),
    bool causal, double softmax_scale, uint64_t head_begin, uint64_t head_num,
    uint64_t full_head_num, [[maybe_unused]] int64_t chunked_prefill_size,
    int prefill_start, int k_len, bool kv_is_packed)
{
    int svls = svcntw();
    int svlh = svcnth();
    svbool_t pg_32_all = svptrue_b32();
    svbool_t pg_16_all = svptrue_b16();
    auto q_len = q.size(0);
    auto pack_q_len = (q_len + br - 1) / br * br;
    auto pack_k_len = (k_len + bc - 1) / bc * bc;
    auto num_heads = q.size(1);
    auto qk_head_dim = q.size(2);
    auto vo_head_dim = v.size(2);

    auto pack_q_data = pack_q.data_ptr();
    auto pack_k_data = pack_k.data_ptr();
    auto pack_v_data = pack_v.data_ptr();

    if (kv_is_packed) {
        pack_k_data = k.data_ptr();
        pack_v_data = v.data_ptr();
    }

    for (int hi = head_begin; hi < head_begin + head_num; hi += 1) {
        auto q_data_head_base = q.data_ptr() + hi * qk_head_dim;
        if (!kv_is_packed) {
            // pack KV, 均按照上取svls倍数为序列长进行补齐
            auto k_data_head_base = k.data_ptr() + hi * qk_head_dim;
            auto v_data_head_base = v.data_ptr() + hi * vo_head_dim;
            block_pack_bf16_1VL_trans_gemm_varlen(k_data_head_base, pack_k_data, k_len, qk_head_dim, k.stride(0), bc);
            block_pack_bf16_2VL_gemm_varlen(v_data_head_base, pack_v_data, k_len, vo_head_dim, v.stride(0), bc);
        } else {
            // total len aligned to 1024
            auto per_head_len = k.size(0) / 1024 * 1024 / full_head_num;
            auto cur_kv_start = per_head_len * hi;
            auto cur_kv_len = (((k_len) + bc - 1) / bc * bc) / full_head_num;

            pack_k_data = tensor_slice(k, 0, cur_kv_start, cur_kv_start + cur_kv_len).data_ptr();
            pack_v_data = tensor_slice(v, 0, cur_kv_start, cur_kv_start + cur_kv_len).data_ptr();
        }
        for (int64_t q_start = 0; q_start < q_len; q_start += br) {
            int64_t q_step = std::min(q_len - q_start, br);
            int64_t padding_q_step = (q_step + svls - 1) / svls * svls;
            int64_t q_end = q_start + q_step;
            // 加载Q O M L ...置零O M L
            auto q_data = q_data_head_base + q_start * q.stride(0);
            auto out_block_old_data = out_block_old.data_ptr();
            auto max_block_old_data = max_block_old.data_ptr();
            auto base_block_old_data = base_block_old.data_ptr();
            tensor_fill<float, 2>(out_block_old, 0.0f);
            tensor_fill<float, 1>(max_block_old, -INFINITY);
            tensor_fill<float, 1>(base_block_old, 0.0f);

            bfloat16_t* out_ptr = out.data_ptr() + q_start * out.stride(0) + hi * vo_head_dim;
            block_pack_bf16_1VL_trans_gemm_varlen(q_data, pack_q_data, q_step, qk_head_dim, q.stride(0), br);
            for (int64_t k_start = 0; k_start < k_len; k_start += bc) {
                int64_t k_step = std::min(k_len - k_start, bc);
                int64_t padding_k_step = (k_step + svls - 1) / svls * svls;
                int64_t k_end = k_start + k_step;

                auto s_data = s.data_ptr();
                auto out_block_new_data = out_block_new.data_ptr();
                auto max_block_new_data = max_block_new.data_ptr();
                auto base_block_new_data = base_block_new.data_ptr();
                auto k_data = pack_k_data + k_start * qk_head_dim;
                auto v_data = pack_v_data + k_start * vo_head_dim;

                if (!(causal && k_start + 1 > prefill_start + q_start + q_step)) {
                    qk_gemm_4vl_1vl(pack_q_data, k_data, s_data, padding_q_step, padding_k_step, qk_head_dim, q_start,
                        k_start + 0,
                        prefill_start);  // S=QK->[br,bc]=[br,qk_head_dim]*[qk_head_dim,bc]

                    softmax_fusekernel(q_start, k_start, q_step, k_step, s_data, softmax_scale, max_block_old_data,
                        base_block_old_data, max_block_new_data, base_block_new_data, v_data, prefill_start, 0);

                    pv_gemm_2vl_2vl_out_gemv(s_data, v_data, out_block_new_data, padding_q_step, vo_head_dim,
                        padding_k_step, out_block_old_data, max_block_old_data, q_start, k_start + 0, out_ptr, bc,
                        prefill_start);

                    // 交换新旧指针实现内存复用
                    swap_dataptr(out_block_old, out_block_new);
                    std::swap(out_block_old_data, out_block_new_data);
                    swap_dataptr(max_block_old, max_block_new);
                    std::swap(max_block_old_data, max_block_new_data);
                    swap_dataptr(base_block_old, base_block_new);
                    std::swap(base_block_old_data, base_block_new_data);
                    continue;
                }
                break;
            }
            for (uint64_t qi = 0; qi < q_step; qi++) {
                float vec_scale = 1.0 / base_block_old_data[qi];
                float* st_ptr = out_block_old_data + qi * out_block_old.stride(0);
                bfloat16_t* out_ptr_row = out_ptr + qi * out.stride(0);
#pragma unroll(4)
                for (uint64_t ki = 0; ki < vo_head_dim; ki += 2 * svls) {
                    svfloat32_t vec_out0 = svld1(pg_32_all, st_ptr + ki);
                    svfloat32_t vec_out1 = svld1(pg_32_all, st_ptr + ki + svls);
                    vec_out0 = svmul_x(pg_32_all, vec_out0, vec_scale);
                    vec_out1 = svmul_x(pg_32_all, vec_out1, vec_scale);
                    svbfloat16_t bf_vec_out0 = svcvt_bf16_x(pg_32_all, vec_out0);
                    svbfloat16_t bf_vec_out1 = svcvt_bf16_x(pg_32_all, vec_out1);
                    svbfloat16_t out_vec = svuzp1(bf_vec_out0, bf_vec_out1);
                    svst1(pg_16_all, out_ptr_row + ki, out_vec);
                }
            }
        }
    }
}
}  // namespace

void flash_attention(
    const Tensor<bfloat16_t, 3> &q,
    const Tensor<bfloat16_t, 3> &k,
    const Tensor<bfloat16_t, 3> &v,
    const Tensor<bfloat16_t, 3> &out,
    const Tensor<bfloat16_t, 2> &pack_q,
    const Tensor<bfloat16_t, 3> &pack_k,
    const Tensor<bfloat16_t, 3> &pack_v,
    const Tensor<float, 2> &s,
    const Tensor<float, 3> &out_block_old,
    const Tensor<float, 3> &out_block_new,
    const Tensor<float, 2> &max_block_old,
    const Tensor<float, 2> &max_block_new,
    const Tensor<float, 2> &base_block_old,
    const Tensor<float, 2> &base_block_new,
    bool causal,
    double softmax_scale,
    const Tensor<int, 1> &cu_seqlens_q,
    const Tensor<int, 1> &cu_seqlens_k,
    int chunked_prefill_size, std::vector<int64_t>& seq_lens, std::vector<int64_t>& cur_lens, bool kv_is_packed
    )
{
    uint64_t num_seqs = cu_seqlens_q.size(0) - 1;
    uint64_t threads_num = kutacc::get_thread_num();
    auto cu_seqlens_q_data = cu_seqlens_q.data_ptr();
    auto cu_seqlens_k_data = cu_seqlens_k.data_ptr();
    uint64_t head_nums = q.size(1);
    auto qk_head_dim = q.size(2);
    auto vo_head_dim = v.size(2);

    uint64_t head_split_num = std::max(1lu, std::min(threads_num, num_seqs * head_nums) / num_seqs);
    uint64_t q_split = std::max(1lu, threads_num / head_split_num / num_seqs);
    kutacc::parallel_for(0, num_seqs * head_split_num * q_split, 1, [&](uint64_t start, uint64_t end) {
        for (uint64_t bi = start; bi < end; bi++) {
            uint64_t seq_index = bi / (head_split_num * q_split);
            uint64_t q_index = bi % q_split;
            uint64_t head_index = bi / q_split;
            uint64_t seq_q_start = cu_seqlens_q_data[seq_index];
            uint64_t seq_q_end = cu_seqlens_q_data[seq_index + 1];
            uint64_t seq_q_parts = (seq_q_end - seq_q_start + q_split - 1) / q_split;
            auto q_start = seq_q_start + q_index * seq_q_parts;
            auto q_end = std::min(seq_q_end, seq_q_start + (q_index + 1) * seq_q_parts);
            if(q_end <= q_start){
                continue;
            }

            auto k_start = cu_seqlens_k_data[seq_index];
            auto k_end = cu_seqlens_k_data[seq_index + 1];

            uint64_t split_head_nums_raf = (head_nums + head_split_num - 1) / head_split_num;
            uint64_t head_begin = split_head_nums_raf * (head_index % head_split_num);
            uint64_t split_head_nums = std::min(split_head_nums_raf, head_nums - head_begin);
            int tid = kutacc::get_thread_id();

            auto out_block_old_thread = tensor_select(out_block_old, 0, tid);
            auto out_block_new_thread = tensor_select(out_block_new, 0, tid);
            auto max_block_old_thread = tensor_select(max_block_old, 0, tid);
            auto max_block_new_thread = tensor_select(max_block_new, 0, tid);
            auto base_block_old_thread = tensor_select(base_block_old, 0, tid);
            auto base_block_new_thread = tensor_select(base_block_new, 0, tid);

            if(kv_is_packed){
                KUTACC_CHECK(num_seqs == 1, "num_seqs must be 1 for kv_is_packed, now num_seqs = ", num_seqs);
            }
            auto actual_k = kv_is_packed ? pack_k : tensor_slice(k, 0, k_start, k_end);
            auto actual_v = kv_is_packed ? pack_v : tensor_slice(v, 0, k_start, k_end);

            head_split_flash_attention_v2_sme_impl(tensor_slice(q, 0, q_start, q_end), actual_k, actual_v,
                tensor_slice(out, 0, q_start, q_end), tensor_select(pack_q, 0, tid), tensor_select(pack_k, 0, tid),
                tensor_select(pack_v, 0, tid), tensor_select(s, 0, tid), out_block_old_thread, out_block_new_thread,
                max_block_old_thread, max_block_new_thread, base_block_old_thread, base_block_new_thread, causal,
                softmax_scale, head_begin, split_head_nums, head_nums, chunked_prefill_size,
                seq_lens[seq_index] - cur_lens[seq_index] + q_index * seq_q_parts, k_end - k_start, kv_is_packed);
        }
    });
}

}  // namespace attn
