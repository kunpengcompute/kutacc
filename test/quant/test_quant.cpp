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

void test_quant(std::vector<int64_t> &cas)
{
    int height = cas[0];
    int width = cas[1];
    std::unique_ptr<bfloat16_t[]> act(new bfloat16_t[height * width]);
    std::unique_ptr<int8_t[]> out(new int8_t[height * width]);
    std::unique_ptr<float[]> scale(new float[height]);
    std::unique_ptr<int8_t[]> expect(new int8_t[height * width]);
    std::unique_ptr<float[]> expect_scale(new float[height]);

    std::mt19937 rnd(time(0));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            act[i * width + j] = rnd() * 1.0 / rnd.max();
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            out[i * width + j] = expect[i * width + j] = rnd() * 1.0 / rnd.max();
        }
    }
    for (int i = 0; i < height; i++) {
        double absmax = 0;
        for (int j = 0; j < width; j++) {
            float value = to_float(act[i * width + j]);
            absmax = std::max(absmax, static_cast<double>(std::abs(value)));
        }
        expect_scale[i] = absmax / 127;
        double scale_inv = 127 / absmax;
        for (int j = 0; j < width; j++) {
            float value = to_float(act[i * width + j]);
            value = std::clamp(std::round(value * scale_inv), -128.0, 127.0);
            expect[i * width + j] = to_bf16(value);
        }
    }

    kutacc::quant(height, width, act.get(), width, out.get(), width, scale.get());

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

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases={{128, 7168}, {128, 1536}, {13, 711}, {759, 2631}};
        read_args(cases, 2, argc, argv);
        for (auto cas : cases) {
            test_quant(cas);
        }
    });
}
