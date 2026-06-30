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
#include <random>
#include <memory>
#include <cstring>
#include <iostream>
#include <ctime>
#include <cmath>

#include <arm_bf16.h>
#include "kutacc.h"
#include "helper.h"
#include "check.h"
#include "scalar_type.h"
#include "span.h"
#include "tensor.h"

template <typename dtype, int64_t dim>
kutacc::Tensor<dtype, dim> convert_to_kutacc_tensor(const Tensor& x)
{
    return kutacc::Tensor<dtype, dim>(x.data_ptr<dtype>(), x.sizes, x.strides);
}

void test_rope(std::vector<int64_t> &cas)
{
    int64_t n_tokens     = cas[0];
    int64_t num_q_heads  = cas[1];
    int64_t num_k_heads  = cas[2];
    int64_t head_size    = cas[3];
    int64_t max_position = cas[4];

    if (head_size % 2 != 0) {
        std::cerr << "head_size must be even" << std::endl;
        exit(1);
        return;
    }

    const double freq_base = 10000.0;

    size_t ddr_size = 1024 * 1024 * 1024ULL;
    std::unique_ptr<uint8_t[]> mem(new uint8_t[ddr_size]);
    u8span ddr_available{mem.get(), ddr_size};

    Tensor position_ids = Tensor::alloc_from(
        ScalarType::Int64, {n_tokens}, ddr_available);
    Tensor q = Tensor::alloc_from(
        ScalarType::BFloat16, {n_tokens, num_q_heads, head_size}, ddr_available);
    Tensor k = Tensor::alloc_from(
        ScalarType::BFloat16, {n_tokens, num_k_heads, head_size}, ddr_available);
    Tensor q_out = Tensor::alloc_from(
        ScalarType::BFloat16, {n_tokens, num_q_heads, head_size}, ddr_available);
    Tensor k_out = Tensor::alloc_from(
        ScalarType::BFloat16, {n_tokens, num_k_heads, head_size}, ddr_available);
    Tensor cos_cache = Tensor::alloc_from(
        ScalarType::BFloat16, {max_position, head_size}, ddr_available);

    int64_t* pos_ids = position_ids.data_ptr<int64_t>();
    bfloat16_t* q_raw = q.data_ptr<bfloat16_t>();
    bfloat16_t* k_raw = k.data_ptr<bfloat16_t>();
    bfloat16_t* cos_sin_cache = cos_cache.data_ptr<bfloat16_t>();

    std::mt19937 rnd(time(0));
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int64_t> pos_dist(0, max_position - 1);

    for (int64_t i = 0; i < n_tokens; ++i)
        pos_ids[i] = pos_dist(rnd);

    int64_t q_elems = n_tokens * num_q_heads * head_size;
    int64_t k_elems = n_tokens * num_k_heads * head_size;
    for (int64_t i = 0; i < q_elems; ++i) q_raw[i] = to_bf16(val_dist(rnd));
    for (int64_t i = 0; i < k_elems; ++i) k_raw[i] = to_bf16(val_dist(rnd));

    // cos_sin_cache
    for (int64_t pos = 0; pos < max_position; ++pos) {
        for (int64_t j = 0; j < head_size / 2; ++j) {
            double theta = pos * std::pow(freq_base, -2.0 * j / head_size);
            float c = static_cast<float>(std::cos(theta));
            float s = static_cast<float>(std::sin(theta));
            cos_sin_cache[pos * head_size + j] = to_bf16(c);
            cos_sin_cache[pos * head_size + head_size / 2 + j] = to_bf16(s);
        }
    }

    std::unique_ptr<bfloat16_t[]> expect_q_out(new bfloat16_t[q_elems]);
    std::unique_ptr<bfloat16_t[]> expect_k_out(new bfloat16_t[k_elems]);

    for (int64_t t = 0; t < n_tokens; ++t) {
        int64_t pos = pos_ids[t];
        const bfloat16_t* cache_row = &cos_sin_cache[pos * head_size];

        for (int64_t h = 0; h < num_q_heads; ++h) {
            const bfloat16_t* src = &q_raw[t * num_q_heads * head_size + h * head_size];
            bfloat16_t* dst = &expect_q_out[t * num_q_heads * head_size + h * head_size];
            for (int64_t i = 0; i < head_size; i += 2) {
                float x1 = to_float(src[i]);
                float x2 = to_float(src[i + 1]);
                float cos_val = to_float(cache_row[i / 2]);
                float sin_val = to_float(cache_row[head_size / 2 + i / 2]);
                float y1 = x1 * cos_val - x2 * sin_val;
                float y2 = x2 * cos_val + x1 * sin_val;
                dst[i]     = to_bf16(y1);
                dst[i + 1] = to_bf16(y2);
            }
        }

        for (int64_t h = 0; h < num_k_heads; ++h) {
            const bfloat16_t* src = &k_raw[t * num_k_heads * head_size + h * head_size];
            bfloat16_t* dst = &expect_k_out[t * num_k_heads * head_size + h * head_size];
            for (int64_t i = 0; i < head_size; i += 2) {
                float x1 = to_float(src[i]);
                float x2 = to_float(src[i + 1]);
                float cos_val = to_float(cache_row[i / 2]);
                float sin_val = to_float(cache_row[head_size / 2 + i / 2]);
                float y1 = x1 * cos_val - x2 * sin_val;
                float y2 = x2 * cos_val + x1 * sin_val;
                dst[i]     = to_bf16(y1);
                dst[i + 1] = to_bf16(y2);
            }
        }
    }

    kutacc::rope(convert_to_kutacc_tensor<int64_t, 1>(position_ids),
        convert_to_kutacc_tensor<bfloat16_t, 3>(q),
        convert_to_kutacc_tensor<bfloat16_t, 3>(k),
        convert_to_kutacc_tensor<bfloat16_t, 3>(q_out),
        convert_to_kutacc_tensor<bfloat16_t, 3>(k_out),
        convert_to_kutacc_tensor<bfloat16_t, 2>(cos_cache));

    auto bf16_to_u16 = [](bfloat16_t val) -> uint16_t {
        uint16_t u;
        std::memcpy(&u, &val, sizeof(u));
        return u;
    };
    auto ulp_diff_within = [&](bfloat16_t a, bfloat16_t b, int max_ulp) -> bool {
        int32_t diff = static_cast<int32_t>(bf16_to_u16(a)) - static_cast<int32_t>(bf16_to_u16(b));
        return std::abs(diff) <= max_ulp;
    };

    bfloat16_t* q_out_ptr = q_out.data_ptr<bfloat16_t>();
    for (int64_t i = 0; i < q_elems; ++i) {
        if (!ulp_diff_within(q_out_ptr[i], expect_q_out[i], 2)) {
            std::fprintf(stderr, "Mismatch in q_out[%ld]: got %f, expected %f\n",
                         i, to_float(q_out_ptr[i]), to_float(expect_q_out[i]));
            std::exit(1);
        }
    }

    bfloat16_t* k_out_ptr = k_out.data_ptr<bfloat16_t>();
    for (int64_t i = 0; i < k_elems; ++i) {
        if (!ulp_diff_within(k_out_ptr[i], expect_k_out[i], 2)) {
            std::fprintf(stderr, "Mismatch in k_out[%ld]: got %f, expected %f\n",
                         i, to_float(k_out_ptr[i]), to_float(expect_k_out[i]));
            std::exit(1);
        }
    }

    std::cout << "rope test passed ("
              << n_tokens << " tokens, "
              << num_q_heads << " q_heads, "
              << num_k_heads << " k_heads, "
              << head_size << " head_size)" << std::endl;
}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases = {
            {1, 8, 8, 192, 8192}, {2, 16, 16, 192, 1024}, {3, 7, 7, 192, 611}, {3, 7, 1, 62, 2611}};
        read_args(cases, 5, argc, argv);
        for (auto &cas : cases) {
            test_rope(cas);
        }
    });
    return 0;
}
