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
    std::unique_ptr<scalar_t[]> out(new scalar_t[height * width]);
    std::unique_ptr<scalar_t[]> residual(new scalar_t[height * width]);
    std::unique_ptr<scalar_t[]> expect(new scalar_t[height * width]);
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
            out[i * width + j] = expect[i * width + j] = rnd() * 1.0 / rnd.max();
            residual[i * width + j] = expect_residual[i * width + j] = rnd() * 1.0 / rnd.max();
        }
    }

    for (int i = 0; i < height; i++) {
        double sqrsum = 0;
        for (int j = 0; j < width; j++) {
            if (has_residual) {
                expect_residual[i * width + j] =
                    to_bf16(to_float(act[i * width + j]) + to_float(expect_residual[i * width + j]));
                sqrsum += to_float(expect_residual[i * width + j]) * to_float(expect_residual[i * width + j]);
            } else {
                sqrsum += to_float(act[i * width + j]) * to_float(act[i * width + j]);
            }
        }
        double rms_inv = 1 / std::sqrt(sqrsum / width + eps);
        for (int j = 0; j < width; j++) {
            float value =
                to_float(has_residual ? to_float(expect_residual[i * width + j]) : to_float(act[i * width + j]));
            expect[i * width + j] = to_bf16(value * rms_inv * to_float(weight[j]));
        }
    }

    if (has_residual) {
        kutacc::rmsnorm<scalar_t, true>(height, width, act.get(), width, weight.get(), eps, residual.get(), out.get());
    } else {
        kutacc::rmsnorm<scalar_t, false>(height, width, act.get(), width, weight.get(), eps, nullptr, out.get());
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (std::abs(to_float(out[i * width + j]) - to_float(expect[i * width + j])) > 0.1) {
                fprintf(stderr, "diff: (%d, %d) is %f, expect %f\n", i, j, to_float(out[i * width + j]),
                    to_float(expect[i * width + j]));
                exit(1);
            }
        }
    }
}
