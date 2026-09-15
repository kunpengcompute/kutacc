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

static uint16_t bf16_to_u16(bfloat16_t val)
{
    uint16_t u;
    std::memcpy(&u, &val, sizeof(u));
    return u;
}

static bool bf16_near(bfloat16_t a, bfloat16_t b, int max_ulp = 2)
{
    int32_t diff = static_cast<int32_t>(bf16_to_u16(a)) - static_cast<int32_t>(bf16_to_u16(b));
    return std::abs(diff) <= max_ulp;
}

void test_mul_scalar_add(std::vector<int64_t> &cas)
{
    int64_t num         = cas[0];
    int64_t load_mode   = cas[1];  // 0 or 1
    bool load_output    = (load_mode != 0);

    std::mt19937 rnd(time(0));
    std::uniform_real_distribution<float> val_dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> alpha_dist(-3.0f, 3.0f);

    float alpha = alpha_dist(rnd);

    std::unique_ptr<bfloat16_t[]> i_ptr(new bfloat16_t[num]);
    std::unique_ptr<bfloat16_t[]> o_ptr(new bfloat16_t[num]);
    std::unique_ptr<bfloat16_t[]> expected(new bfloat16_t[num]);

    for (int64_t j = 0; j < num; ++j) {
        i_ptr[j] = to_bf16(val_dist(rnd));
        o_ptr[j] = to_bf16(val_dist(rnd));   // 总是生成随机原始输出
    }

    for (int64_t j = 0; j < num; ++j) {
        float i_val = to_float(i_ptr[j]);
        float o_val = to_float(o_ptr[j]);
        float expected_val = load_output ? (o_val + alpha * i_val) : (alpha * i_val);
        expected[j] = to_bf16(expected_val);
    }

    kutacc::mul_scalar_add(i_ptr.get(), o_ptr.get(), num, alpha, load_output);

    for (int64_t j = 0; j < num; ++j) {
        if (!bf16_near(o_ptr[j], expected[j], 2)) {
            std::fprintf(stderr,
                         "Mismatch at index %ld: load=%ld alpha=%.6f got=%.6f expected=%.6f\n",
                         j, load_mode, alpha,
                         to_float(o_ptr[j]), to_float(expected[j]));
            std::exit(1);
        }
    }

    std::cout << "mul_scalar_add test passed (num=" << num
              << ", load=" << load_mode << ", alpha=" << alpha << ")\n";
}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases={{4096, 1}, {16384, 0}, {11, 1}, {71, 0}};

        // 每个用例: [num, load_mode (0/1)]
        read_args(cases, 2, argc, argv);
        for (auto &cas : cases) {
            test_mul_scalar_add(cas);
        }
    });
    return 0;
}
