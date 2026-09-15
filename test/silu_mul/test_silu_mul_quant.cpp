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

#include <arm_bf16.h>
#include <arm_neon.h>
#include <arm_sve.h>

#include "kutacc.h"
#include "helper.h"

static float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

void test_silu_mul_quant(std::vector<int64_t> &cas)
{
    int64_t height = cas[0];
    int64_t width = cas[1];  // gateupsize, must be even
    if (width % 2 != 0) {
        std::cerr << "width must be even" << std::endl;
        return;
    }
    int64_t half_width = width / 2;

    std::unique_ptr<bfloat16_t[]> gateup(new bfloat16_t[height * width]);
    std::unique_ptr<int8_t[]> out(new int8_t[height * half_width]);
    std::unique_ptr<float[]> online_scales(new float[height]);
    std::unique_ptr<int8_t[]> expect_out(new int8_t[height * half_width]);
    std::unique_ptr<float[]> expect_online_scales(new float[height]);

    std::mt19937 rnd(time(0));
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    // Randomly initialize gateup data
    for (int64_t i = 0; i < height; i++) {
        for (int64_t j = 0; j < width; j++) {
            gateup[i * width + j] = to_bf16(dist(rnd));
        }
    }

    // Compute expected outputs
    for (int64_t i = 0; i < height; i++) {
        double max_val = 0.0;
        std::vector<float> vals(half_width);

        for (int64_t j = 0; j < half_width; j++) {
            float gate = to_float(gateup[i * width + j]);
            float up   = to_float(gateup[i * width + half_width + j]);
            float silu_g = gate * sigmoid(gate);
            float x = silu_g * up;
            vals[j] = x;
            max_val = std::max(max_val, static_cast<double>(std::abs(x)));
        }

        if (max_val < 1e-12) {
            max_val = 1e-12;   // avoid division by zero
        }
        expect_online_scales[i] = static_cast<float>(max_val / 127.0);
        double scale_inv = 127.0 / max_val;

        for (int64_t j = 0; j < half_width; j++) {
            float quantized = std::round(vals[j] * scale_inv);
            float clamped = std::max(-128.0f, std::min(127.0f, quantized));
            expect_out[i * half_width + j] = static_cast<int8_t>(clamped);
        }
    }

    // Call the function under test
    kutacc::silu_mul_quant(width, height * width, out.get(), gateup.get(), online_scales.get());

    // Compare results
    for (int64_t i = 0; i < height; i++) {
        for (int64_t j = 0; j < half_width; j++) {
            if (std::abs(static_cast<int>(out[i * half_width + j]) -
                         static_cast<int>(expect_out[i * half_width + j])) > 1) {
                std::fprintf(stderr, "diff at (%ld, %ld): got %d, expected %d\n",
                             i, j,
                             static_cast<int>(out[i * half_width + j]),
                             static_cast<int>(expect_out[i * half_width + j]));
                std::exit(1);
            }
        }
        if (std::abs(online_scales[i] - expect_online_scales[i]) > 1e-5f) {
            std::fprintf(stderr, "diff at online_scale %ld: got %f, expected %f\n",
                         i, online_scales[i], expect_online_scales[i]);
            std::exit(1);
        }
    }
}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases={{64, 4096}, {8, 4096}, {1, 11}, {31, 57}};
        read_args(cases, 2, argc, argv);   // each case contains [height, width]
        for (auto &cas : cases) {
            test_silu_mul_quant(cas);
        }
    });
    return 0;
}
