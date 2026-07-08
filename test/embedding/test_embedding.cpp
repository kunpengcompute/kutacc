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
#include <vector>
#include <cstdlib>

#include <arm_neon.h>
#include "kutacc.h"

using scalar_t = __bf16;

inline scalar_t to_bf16(float x)
{
    return vcvth_bf16_f32(x);
}

inline float to_float(scalar_t x)
{
    return vcvtah_f32_bf16(x);
}

static uint16_t bf16_to_u16(scalar_t val)
{
    uint16_t u;
    std::memcpy(&u, &val, sizeof(u));
    return u;
}

static bool bf16_near(scalar_t a, scalar_t b, int max_ulp = 2)
{
    int32_t diff = static_cast<int32_t>(bf16_to_u16(a)) - static_cast<int32_t>(bf16_to_u16(b));
    return std::abs(diff) <= max_ulp;
}

static void read_args(std::vector<std::vector<int64_t>>& cases, int argc, char **argv)
{
    if (argc < 2) {
        return;
    }
    int num_args = argc - 1;
    if (num_args % 4 != 0) {
        std::cerr << "Error: number of arguments must be multiple of 4 (n_tokens hidden vocab_start vocab_end)\n";
        std::exit(1);
    }
    cases.clear();
    for (int i = 1; i < argc; i += 4) {
        int64_t n_tokens  = std::atoll(argv[i]);
        int64_t hidden    = std::atoll(argv[i + 1]);
        int64_t vocab_start = std::atoll(argv[i + 2]);
        int64_t vocab_end = std::atoll(argv[i + 3]);
        cases.push_back({n_tokens, hidden, vocab_start, vocab_end});
    }
}

static void compute_expected_row(const scalar_t* weight, int64_t hidden,
                                 int64_t vocab_start, int64_t vocab_end, int64_t token_id,
                                 scalar_t* out_row)
{
    if (token_id >= vocab_start && token_id < vocab_end) {
        std::memcpy(out_row, weight + (token_id - vocab_start) * hidden,
                    hidden * sizeof(scalar_t));
    } else {
        std::memset(out_row, 0, hidden * sizeof(scalar_t));
    }
}

void test_embedding(const std::vector<int64_t>& params)
{
    if (params.size() != 4) {
        std::cerr << "Invalid test case: need [n_tokens, hidden, vocab_start, vocab_end]" << std::endl;
        return;
    }
    int64_t n_tokens    = params[0];
    int64_t hidden      = params[1];
    int64_t vocab_start = params[2];
    int64_t vocab_end   = params[3];
    int64_t vocab_size  = vocab_end - vocab_start;

    std::unique_ptr<int64_t[]> input(new int64_t[n_tokens]);
    std::unique_ptr<scalar_t[]> weight(new scalar_t[vocab_size * hidden]);
    std::unique_ptr<scalar_t[]> out(new scalar_t[n_tokens * hidden]);
    std::unique_ptr<scalar_t[]> expected(new scalar_t[n_tokens * hidden]);

    std::mt19937 rnd(time(0));
    std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);

    const int64_t OUT_OF_RANGE_MAX = 129280;
    for (int64_t i = 0; i < n_tokens; ++i) {
        if (rnd() % 2 == 0) {
            input[i] = vocab_start + (rnd() % vocab_size);
        } else {
            input[i] = rnd() % OUT_OF_RANGE_MAX;
        }
    }

    for (int64_t i = 0; i < vocab_size; ++i) {
        for (int64_t j = 0; j < hidden; ++j) {
            weight[i * hidden + j] = to_bf16(weight_dist(rnd));
        }
    }

    for (int64_t i = 0; i < n_tokens * hidden; ++i) {
        float rand_val = weight_dist(rnd);
        out[i] = to_bf16(rand_val);
        expected[i] = to_bf16(rand_val);
    }

    for (int64_t i = 0; i < n_tokens; ++i) {
        compute_expected_row(weight.get(), hidden, vocab_start, vocab_end,
                             input[i], expected.get() + i * hidden);
    }

    kutacc::embedding(input.get(), weight.get(), out.get(), sizeof(scalar_t),
                      n_tokens, hidden, vocab_start, vocab_end);

    bool passed = true;
    for (int64_t i = 0; i < n_tokens && passed; ++i) {
        for (int64_t j = 0; j < hidden; ++j) {
            scalar_t got = out[i * hidden + j];
            scalar_t exp = expected[i * hidden + j];
            if (!bf16_near(got, exp, 2)) {
                std::cerr << "Mismatch at token " << i << ", index " << j
                          << ": token_id=" << input[i]
                          << ", got=" << to_float(got)
                          << ", expected=" << to_float(exp) << std::endl;
                passed = false;
                break;
            }
        }
    }

    if (passed) {
        std::cout << "Embedding test passed: n_tokens=" << n_tokens
                  << ", hidden=" << hidden
                  << ", vocab_start=" << vocab_start
                  << ", vocab_end=" << vocab_end << std::endl;
    } else {
        std::exit(1);
    }
}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases = {
            {128, 7168, 0, 8080},
            {128, 7168, 8080, 16160},
            {191, 271, 7, 8081},
            {2077, 459, 2166, 17611}
        };

        read_args(cases, argc, argv);

        for (auto& cas : cases) {
            test_embedding(cas);
        }
    });
    return 0;
}