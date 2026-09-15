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
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <arm_neon.h>

#include "kutacc.h"

using scalar_t = __bf16;

inline scalar_t to_bf16(float x) { return vcvth_bf16_f32(x); }
inline float to_float(scalar_t x) { return vcvtah_f32_bf16(x); }

static void read_args(std::vector<std::vector<int64_t>>& cases, int argc, char **argv)
{
    if (argc < 2) {
        return;
    }
    int num_args = argc - 1;
    if (num_args % 3 != 0) {
        std::cerr << "Error: arguments must be multiples of 3 (height width has_residual)\n";
        std::exit(1);
    }
    cases.clear();
    for (int i = 1; i < argc; i += 3) {
        int64_t h = std::atoll(argv[i]);
        int64_t w = std::atoll(argv[i + 1]);
        int64_t r = std::atoll(argv[i + 2]);
        cases.push_back({h, w, r});
    }
}

void test_rmsnorm_quant(int64_t height, int64_t width, bool has_residual)
{
    const float eps = 1e-6f;
    std::unique_ptr<scalar_t[]> act(new scalar_t[height * width]);
    std::unique_ptr<scalar_t[]> weight(new scalar_t[width]);
    std::unique_ptr<int8_t[]> out(new int8_t[height * width]);
    std::unique_ptr<float[]> scale(new float[height]);
    std::unique_ptr<scalar_t[]> residual(new scalar_t[height * width]);
    std::unique_ptr<int8_t[]> expect(new int8_t[height * width]);
    std::unique_ptr<float[]> expect_scale(new float[height]);
    std::unique_ptr<scalar_t[]> expect_residual(new scalar_t[height * width]);

    std::mt19937 rnd(time(0));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int16_t> int_dist(0, 7); // for initial out values

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                act[i * width + j] = rnd() * 1.0 / rnd.max();
            }
        }
        for (int i = 0; i < width; i++) {
            weight[i] = rnd() * 1.0 / rnd.max();
        }
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                out[i * width + j] = expect[i * width + j] = rnd() % 8;
                residual[i * width + j] = expect_residual[i * width + j] = rnd() * 1.0 / rnd.max();
            }
        }
        for (int i = 0; i < height; i++) {
            scale[i] = expect_scale[i] = rnd() * 1.0 / rnd.max();
        }

    // Compute expected
    for (int64_t i = 0; i < height; ++i) {
        double sqrsum = 0.0;
        double absmax = 0.0;
        for (int64_t j = 0; j < width; ++j) {
            float value;
            if (has_residual) {
                value = to_float(act[i * width + j]) + to_float(expect_residual[i * width + j]);
                expect_residual[i * width + j] = to_bf16(value);
            } else {
                value = to_float(act[i * width + j]);
            }
            sqrsum += value * value;
            absmax = std::max(absmax, (double)std::abs(value * to_float(weight[j])));
        }
        double rms = std::sqrt(sqrsum / width + eps);
        double quant_scale = 127.0 / absmax;
        expect_scale[i] = static_cast<float>(absmax / 127.0 / rms);
        for (int64_t j = 0; j < width; ++j) {
            float value;
            if (has_residual)
                value = to_float(expect_residual[i * width + j]);
            else
                value = to_float(act[i * width + j]);
            value = value * quant_scale * to_float(weight[j]);
            value = std::clamp(std::round(value), -128.0f, 127.0f);
            expect[i * width + j] = static_cast<int8_t>(value);
        }
    }

    // Call kernel
    if (has_residual) {
        kutacc::rmsnorm_quant<true>(height, width, act.get(), width, weight.get(), eps,
                                    residual.get(), width, out.get(), width, scale.get(), 1);
    } else {
        kutacc::rmsnorm_quant<false>(height, width, act.get(), width, weight.get(), eps,
                                     nullptr, 0, out.get(), width, scale.get(), 1);
    }

    // Verify
    for (int64_t i = 0; i < height; ++i) {
        for (int64_t j = 0; j < width; ++j) {
            if (std::abs(out[i * width + j] - expect[i * width + j]) > 1) {
                std::cerr << "Mismatch at (" << i << "," << j << "): got=" << (int)out[i * width + j]
                          << ", exp=" << (int)expect[i * width + j] << std::endl;
                std::exit(1);
            }
        }
        if (std::abs(scale[i] - expect_scale[i]) > 0.01f) {
            std::cerr << "Scale mismatch at row " << i << ": got=" << scale[i]
                      << ", exp=" << expect_scale[i] << std::endl;
            std::exit(1);
        }
    }

    std::cout << "RMSNorm+Quant test passed: height=" << height << ", width=" << width
              << ", residual=" << has_residual << std::endl;
}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> default_cases = {
            {128, 7168, 0},
            {128, 7168, 1},
            {3, 11, 0},
            {5, 7, 1}
        };
        std::vector<std::vector<int64_t>> cases = default_cases;
        read_args(cases, argc, argv);
        for (auto &cas : cases) {
            if (cas.size() != 3) {
                std::cerr << "Invalid test case (need height width has_residual)" << std::endl;
                continue;
            }
            test_rmsnorm_quant(cas[0], cas[1], cas[2] != 0);
        }
    });
    return 0;
}