/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under a modified version of the MIT License. See LICENSE in the project root for license information.
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
#include <iostream>
#include <ctime>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <optional>
#include <cassert>
#include <iomanip>
#include <algorithm>

#include "helper.h"
#include "kutacc.h"

enum class MoETestType {
    GATE_UP,
    DOWN
};

void random_fill_1_modern(int8_t* data, int64_t numel)
{
    assert(data != nullptr && numel > 0);

    const size_t cache_size = 313 * 7;
    static std::vector<int8_t> cache(cache_size);
    static std::once_flag cache_init_flag;

    std::call_once(cache_init_flag, []() {
        std::mt19937 rng(std::random_device{}());
        std::bernoulli_distribution dist(0.5);

        for (size_t i = 0; i < cache_size; ++i) {
            cache[i] = static_cast<int8_t>(dist(rng) ? 1 : -1);
        }
    });

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = cache[i % cache_size];
    }
}

void random_fill_float_range_modern(float* data, int64_t numel, float min_val = 0.0f, float max_val = 0.3f)
{
    assert(data != nullptr && numel > 0);
    assert(min_val <= max_val);

    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (int64_t i = 0; i < numel; ++i) {
        data[i] = dist(rng);
    }
}

void generate_experts_offset(int* experts_offset, int64_t num_experts, int64_t total_bs)
{
    assert(experts_offset != nullptr);
    std::mt19937 rnd(time(0));
    experts_offset[0] = 0;
    for (int64_t i = 1; i < num_experts; i++) {
        experts_offset[i] = experts_offset[i - 1] + (rnd() % (total_bs / num_experts) + 1);
    }
    experts_offset[num_experts] = total_bs;
}

void generate_token_ids(int* token_ids, int64_t total_bs)
{
    assert(token_ids != nullptr);
    std::mt19937 rnd(time(0));
    for (int64_t i = 0; i < total_bs; i++) {
        token_ids[i] = rnd() % total_bs;
    }
}

void pack_weights(int8_t* weights,
    int8_t* weights_buffer, int64_t num_experts, int64_t N, int64_t K,
    const kutacc::MatrixTilingBlock& t)
{
    assert(weights && weights_buffer);
    auto tile_k = std::get<2>(t);
    for (int64_t exp_id = 0; exp_id < num_experts; ++exp_id) {
        kutacc::s8_gemm_pack(N, K, N, tile_k, weights + N * K * exp_id, weights_buffer);
        memcpy(weights + N * K * exp_id, weights_buffer, N * K * sizeof(int8_t));
    }
}

void compute_expect_output(bfloat16_t* expect, const int8_t* acts, const int8_t* weights, const float* acts_scale,
    const float* weights_scale, const int* experts_offset, const int* token_ids, int64_t K, int64_t N,
    int64_t num_experts, MoETestType test_type)
{
    assert(expect && acts && weights && acts_scale && weights_scale && experts_offset);

    for (int64_t exp_id = 0; exp_id < num_experts; exp_id++) {
        int64_t exp_start = experts_offset[exp_id];
        int64_t exp_end = experts_offset[exp_id + 1];
        for (int64_t idx = exp_start; idx < exp_end; idx++) {
            int64_t act_idx = (test_type == MoETestType::GATE_UP) ? token_ids[idx] : idx;
            for (int64_t j = 0; j < N; j++) {
                int32_t dot_sum = 0;
                for (int64_t k = 0; k < K; k++) {
                    dot_sum += (int32_t)acts[act_idx * K + k] * (int32_t)weights[exp_id * N * K + j * K + k];
                }
                expect[idx * N + j] = to_bf16((float)dot_sum * acts_scale[act_idx] * weights_scale[exp_id * N + j]);
            }
        }
    }
}

void print_test_results(const bfloat16_t* output, const bfloat16_t* expect, int64_t total_elements)
{
    const int64_t PRINT_COUNT = 20000;
    const int64_t print_num = std::min(PRINT_COUNT, total_elements);

    std::cout << "\n=====================================" << std::endl;
    std::cout << "Print first " << print_num << " elements (float format)" << std::endl;
    std::cout << "Index\t\tOutput\t\tExpect" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    for (int64_t i = 0; i < print_num; i++) {
        float o = to_float(output[i]);
        float e = to_float(expect[i]);
        std::cout << i << "\t\t" << o << "\t" << e << std::endl;
    }
    std::cout << "=====================================\n" << std::endl;
}

double calculate_cosine_diff(const bfloat16_t* output, const bfloat16_t* expect, int64_t total_elements)
{
    double dot = 0.0;
    double norm_out = 0.0;
    double norm_exp = 0.0;

    for (int64_t i = 0; i < total_elements; i++) {
        float o = to_float(output[i]);
        float e = to_float(expect[i]);
        dot += (double)o * e;
        norm_out += (double)o * o;
        norm_exp += (double)e * e;
    }

    double cos_diff = 1.0;
    if (std::sqrt(norm_out) > 1e-8 && std::sqrt(norm_exp) > 1e-8) {
        cos_diff = 1.0 - dot / (std::sqrt(norm_out) * std::sqrt(norm_exp));
    } else if (norm_out < 1e-8 && norm_exp < 1e-8) {
        cos_diff = 0.0;
    }
    return cos_diff;
}

void test_fusedmoe(const std::vector<int64_t>& cas, MoETestType test_type)
{
    // 参数解析
    int64_t total_bs    = cas[0];
    int64_t K           = cas[1];
    int64_t N           = cas[2];
    int64_t num_experts = cas[3];
    int64_t lda         = cas[4];
    int64_t ldas        = cas[5];
    kutacc::MatrixTilingBlock t(cas[6], cas[7], cas[8]);
    int64_t tilebuf_size = cas[9];
    std::optional<int64_t> n_slice = cas[10];
    int64_t total_elements = total_bs * N;

    std::unique_ptr<int8_t[]> acts(new int8_t[total_bs * K]);
    std::unique_ptr<int8_t[]> weights(new int8_t[num_experts * N * K]);
    std::unique_ptr<int8_t[]> weights_buffer(new int8_t[N * K]);
    std::unique_ptr<float[]> acts_scale(new float[total_bs]);
    std::unique_ptr<float[]> weights_scale(new float[num_experts * N]);
    std::unique_ptr<bfloat16_t[]> output(new bfloat16_t[total_elements]);
    std::unique_ptr<bfloat16_t[]> expect(new bfloat16_t[total_elements]);
    std::unique_ptr<int[]> experts_offset(new int[num_experts + 1]);
    
    std::unique_ptr<int[]> token_ids;
    if (test_type == MoETestType::GATE_UP) {
        token_ids.reset(new int[total_bs]);
    }

    auto [tmpx_size, tmpy_size, tmpscale_size] = kutacc::get_fusedmoe_s8gemm_workspace_size(N, K, tilebuf_size, t);
    std::unique_ptr<int8_t[]> tmpx(new int8_t[tmpx_size * 4]);
    std::unique_ptr<float[]> tmpy(new float[tmpy_size * 4]);
    std::unique_ptr<float[]> tmp_scales;
    if (test_type == MoETestType::GATE_UP) {
        tmp_scales.reset(new float[tmpscale_size]);
    }

    random_fill_1_modern(acts.get(), total_bs * K);
    random_fill_1_modern(weights.get(), num_experts * N * K);
    random_fill_float_range_modern(acts_scale.get(), total_bs);
    random_fill_float_range_modern(weights_scale.get(), num_experts * N);

    generate_experts_offset(experts_offset.get(), num_experts, total_bs);

    if (test_type == MoETestType::GATE_UP) {
        generate_token_ids(token_ids.get(), total_bs);
    }

    compute_expect_output(expect.get(), acts.get(), weights.get(), acts_scale.get(), weights_scale.get(),
        experts_offset.get(), token_ids.get(), K, N, num_experts, test_type);

    pack_weights(weights.get(), weights_buffer.get(), num_experts, N, K, t);

    if (test_type == MoETestType::GATE_UP) {
        kutacc::fusedmoe_gateup(
            total_bs, K, N, num_experts, lda, ldas,
            acts.get(), weights.get(), acts_scale.get(), weights_scale.get(),
            token_ids.get(), experts_offset.get(), output.get(),
            tmpx.get(), tmpy.get(), tmp_scales.get(),
            t, tilebuf_size, n_slice);
    } else {
        kutacc::fusedmoe_down(
            total_bs, K, N, num_experts,
            acts.get(), weights.get(), acts_scale.get(), weights_scale.get(),
            experts_offset.get(), output.get(),
            tmpx.get(), tmpy.get(),
            t, tilebuf_size, n_slice);
    }

    print_test_results(output.get(), expect.get(), total_elements);

    double cos_diff = calculate_cosine_diff(output.get(), expect.get(), total_elements);

    const std::string test_name = (test_type == MoETestType::GATE_UP) ? "GateUp" : "Down";
    std::cerr << "Cos diff: " << cos_diff
              << " Shape: (" << total_bs << "," << N << "," << K << ") "
              << "Tile: (" << std::get<0>(t) << "," << std::get<1>(t) << "," << std::get<2>(t) << ")\n";

    if (cos_diff > 1e-3) {
        std::cerr << test_name << " Cos diff Too Big!!! Test Failed!\n";
        exit(1);
    }

    std::cout << test_name << " FusedMoE Test Passed! "
              << "Shape: (" << total_bs << ", " << K << ", " << N << ")\n";
}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        // 测试用例：[total_bs, K, N, num_experts, lda, ldas, tile_m, tile_n, tile_k, tilebuf_size, n_slice]
        std::vector<std::vector<int64_t>> cases = {
            {128, 2048, 7168, 4, 2048, 1, 128, 224, 2048, 64, 7168},
            {64, 7168, 4096, 2, 7168, 1, 64, 512, 1792, 64, 4096},
        };
        read_args(cases, 11, argc, argv);

        std::vector<MoETestType> test_types = {MoETestType::GATE_UP, MoETestType::DOWN};

        for (const auto& cas : cases) {
            for (auto type : test_types) {
                test_fusedmoe(cas, type);
                std::cout << "-----------------------------------------\n";
            }
        }
    });
    return 0;
}