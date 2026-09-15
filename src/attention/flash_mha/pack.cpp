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
#include "pack.h"
namespace kutacc {
std::tuple<int64_t, int64_t> get_flash_attention_block()
{
    return std::make_tuple(br, bc);
}

void flash_attention_v_block_pack(int64_t kv_len, int64_t num_heads, int64_t vo_head_dim, int64_t output_len,
    int64_t input_stride0, int64_t input_stride1, bfloat16_t* input, bfloat16_t* output)
{
    auto len_per_head = get_len_per_head(num_heads, output_len);
    KUTACC_CHECK(len_per_head * num_heads >= (kv_len + bc - 1) / bc * bc, len_per_head * num_heads, " ", kv_len);

    kutacc::parallel_for(0, num_heads, 1, [&](int64_t start, int64_t end) {
        for (int hi = start; hi < end; hi++) {
            auto head_bias = hi * len_per_head * num_heads * vo_head_dim;
            block_pack_bf16_2VL_gemm_varlen(input + hi * input_stride1, output + head_bias, kv_len, vo_head_dim,
                input_stride0, bc);
        }
    });
}

void flash_attention_k_block_pack(int64_t kv_len, int64_t num_heads, int64_t qk_head_dim, int64_t output_len,
    int64_t input_stride0, int64_t input_stride1, bfloat16_t* input, bfloat16_t* output)
{
    auto len_per_head = get_len_per_head(num_heads, output_len);
    KUTACC_CHECK(len_per_head * num_heads >= (kv_len + bc - 1) / bc * bc, len_per_head * num_heads, " ", kv_len);

    kutacc::parallel_for(0, num_heads, 1, [&](int64_t start, int64_t end) {
        for (int hi = start; hi < end; hi++) {
            auto head_bias = hi * len_per_head * num_heads * qk_head_dim;
            block_pack_bf16_1VL_trans_gemm_varlen(input + hi * input_stride1, output + head_bias, kv_len, qk_head_dim,
                input_stride0, bc);
        }
    });
}
}
