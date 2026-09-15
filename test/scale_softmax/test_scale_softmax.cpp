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

static void compute_expected_row(const bfloat16_t* src, int64_t width, float scale, bfloat16_t* dst)
{
    std::vector<float> buf(width);
    float max_val = -std::numeric_limits<float>::infinity();
    for (int64_t j = 0; j < width; ++j) {
        buf[j] = to_float(src[j]) * scale;
        max_val = std::max(max_val, buf[j]);
    }

    float sum = 0.0f;
    for (int64_t j = 0; j < width; ++j) {
        buf[j] = std::exp(buf[j] - max_val);   // 与实现中的 fast_exp 一致
        sum += buf[j];
    }

    float inv_sum = 1.0f / sum;
    for (int64_t j = 0; j < width; ++j) {
        float val = buf[j] * inv_sum;
        dst[j] = to_bf16(val);
    }
}

void test_scale_softmax(std::vector<int64_t> &cas)
{
    int64_t height = cas[0];
    int64_t width  = cas[1];

    if (width % 2 != 0) {
        std::cerr << "width must be even (bf16 requirement)" << std::endl;
        return;
    }

    int64_t act_stride = width;
    std::unique_ptr<bfloat16_t[]> act(new bfloat16_t[height * act_stride]);
    std::unique_ptr<float[]> scale(new float[height]);

    std::mt19937 rnd(time(0));
    std::uniform_real_distribution<float> val_dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> scale_dist(0.5f, 2.0f);  // scale 正数，避免全零

    for (int64_t i = 0; i < height; ++i) {
        scale[i] = scale_dist(rnd);
        for (int64_t j = 0; j < width; ++j) {
            act[i * act_stride + j] = to_bf16(val_dist(rnd));
        }
    }

    std::unique_ptr<bfloat16_t[]> expected(new bfloat16_t[height * width]);
    for (int64_t i = 0; i < height; ++i) {
        compute_expected_row(act.get() + i * act_stride, width, scale[i], expected.get() + i * act_stride);
    }

    int64_t buf_stride = width;
    constexpr int kMaxThreads = 1024;
    std::unique_ptr<float[]> buf(new float[kMaxThreads * buf_stride]);

    kutacc::scale_softmax(act.get(), act_stride, scale.get(), buf.get(), buf_stride, height, width);

    for (int64_t i = 0; i < height; ++i) {
        for (int64_t j = 0; j < width; ++j) {
            bfloat16_t got = act[i * act_stride + j];
            bfloat16_t exp = expected[i * act_stride + j];
            if (!bf16_near(got, exp, 2)) {
                std::fprintf(stderr,
                             "Mismatch at (%ld, %ld): scale=%f, got=%f, expected=%f\n",
                             i, j, scale[i], to_float(got), to_float(exp));
                std::exit(1);
            }
        }
    }

    std::cout << "scale_softmax test passed (height=" << height << ", width=" << width << ")\n";
}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases = {{8, 7168}, {4, 129280}, {11, 231}, {67, 8511}};
            
        read_args(cases, 2, argc, argv);   // 每个用例: height, width
        for (auto &cas : cases) {
            test_scale_softmax(cas);
        }
    });
    return 0;
}
