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
#include <arm_neon.h>

#include "kutacc.h"

inline __bf16 to_bf16(float x)
{
    return vcvth_bf16_f32(x);
}

inline float to_float(__bf16 x)
{
    return vcvtah_f32_bf16(x);
}

int main(int argc, char **argv)
{
    using scalar_t = __bf16;
    if (argc != 4) {
        fprintf(stderr, "argc incorrect\n");
        exit(1);
    }
    int height = std::atoi(argv[1]);
    int width = std::atoi(argv[2]);
    bool has_residual = std::atoi(argv[3]);
    float eps = 1e-6;
    std::unique_ptr<scalar_t[]> act(new scalar_t[height * width]);
    std::unique_ptr<scalar_t[]> weight(new scalar_t[width]);
    std::unique_ptr<int8_t[]> out(new int8_t[height * width]);
    std::unique_ptr<float[]> scale(new float[height]);
    std::unique_ptr<scalar_t[]> residual(new scalar_t[height * width]);
    std::unique_ptr<int8_t[]> expect(new int8_t[height * width]);
    std::unique_ptr<float[]> expect_scale(new float[height]);
    std::unique_ptr<scalar_t[]> expect_residual(new scalar_t[height * width]);

    std::mt19937 rnd(time(0));

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

    for (int i = 0; i < height; i++) {
        double sqrsum = 0;
        double absmax = 0;
        for (int j = 0; j < width; j++) {
            float value;
            if (has_residual) {
                expect_residual[i * width + j] =
                    to_bf16(to_float(act[i * width + j]) + to_float(expect_residual[i * width + j]));
                value = to_float(expect_residual[i * width + j]);
            } else {
                value = to_float(act[i * width + j]);
            }
            sqrsum += value * value;
            absmax = std::max(absmax, static_cast<double>(std::abs(value * to_float(weight[j]))));
        }
        double rms = sqrt(sqrsum / width + eps);
        double quant_scale = 127 / absmax;
        expect_scale[i] = absmax / 127 / rms;
        for (int j = 0; j < width; j++) {
            float value = to_float(has_residual ? expect_residual[i * width + j] : act[i * width + j]);
            value = std::clamp(std::round(value * quant_scale * to_float(weight[j])), -128.0, 127.0);
            expect[i * width + j] = to_bf16(value);
        }
    }

    if (has_residual) {
        kutacc::rmsnorm_quant<scalar_t, true>(height, width, act.get(), width, weight.get(), eps, residual.get(),
            out.get(), scale.get());
    } else {
        kutacc::rmsnorm_quant<scalar_t, false>(height, width, act.get(), width, weight.get(), eps, nullptr, out.get(),
            scale.get());
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (std::abs(out[i * width + j] - expect[i * width + j]) > 1) {
                fprintf(stderr, "diff: (%d, %d) is %d, expect %d\n", i, j, out[i * width + j], expect[i * width + j]);
                exit(1);
            }
        }
        if (std::abs(scale[i] - expect_scale[i]) > 0.01) {
            fprintf(stderr, "diff: scale (%d) is %f, expect %f\n", i, scale[i], expect_scale[i]);
            exit(1);
        }
    }
}
