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
#include <arm_sve.h>

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
    if (argc != 3) {
        fprintf(stderr, "argc incorrect\n");
        exit(1);
    }
    int height = std::atoi(argv[1]);
    int width = std::atoi(argv[2]);
    std::unique_ptr<scalar_t[]> act(new scalar_t[height * width]);
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
