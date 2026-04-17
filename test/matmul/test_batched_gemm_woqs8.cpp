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
#include <iostream>
#include <ctime>
#include <cstdint>
#include <cstdlib>

#include <arm_neon.h>

#include "helper.h"
#include "kutacc.h"

void test_batched_gemm_woqs8(std::vector<int64_t> &cas)
{
    int64_t bs = cas[0];
    int64_t m = cas[1];
    int64_t n = cas[2];
    int64_t k = cas[3];
    for (auto rowQuant : {0, 1}) {
        std::unique_ptr<bfloat16_t[]> a(new bfloat16_t[bs * m * k]);
        std::unique_ptr<bfloat16_t[]> pa(new bfloat16_t[bs * m * k]);
        std::unique_ptr<int8_t[]> b(new int8_t[bs * n * k]);
        std::unique_ptr<int8_t[]> pb(new int8_t[bs * n * k]);
        std::unique_ptr<float[]> scale;
        scale.reset(new float[bs * (rowQuant ? n : k)]);
        float *quantArgs[2] = {};
        quantArgs[rowQuant ? 0 : 1] = scale.get();
        std::unique_ptr<bfloat16_t[]> c(new bfloat16_t[bs * m * n]);
        std::unique_ptr<bfloat16_t[]> expect(new bfloat16_t[bs * m * n]);

        std::mt19937 rnd(time(0));
        for (int64_t i = 0; i < bs * m; i++) {
            for (int64_t j = 0; j < k; j++) {
                a[i * k + j] = ((int64_t)rnd() % 4 - 2) * 1.0 / k;
            }
        }
        for (int64_t i = 0; i < bs * n; i++) {
            for (int64_t j = 0; j < k; j++) {
                b[i * k + j] = (int64_t)rnd() % 4 - 2;
            }
        }
        for (int64_t j = 0; j < bs * (rowQuant ? n : k); j++) {
            scale[j] = (rnd() % 5 + 1);
        }
        for (int64_t i = 0; i < bs * m; i++) {
            for (int64_t j = 0; j < n; j++) {
                c[i * n + j] = expect[i * n + j] = rnd() % 4;
            }
        }
        kutacc::parallel_for((int64_t)0, bs, (int64_t)1, [&](int64_t start, int64_t end) {
            for (int64_t l = start; l < end; l++) {
                int64_t s_offset = l * (rowQuant ? n : k);
                for (int64_t i = 0; i < m; i++)
                    for (int64_t j = 0; j < n; j++) {
                        float temp = 0;
                        for (int64_t ki = 0; ki < k; ki++) {
                            temp += to_float(a[l * m * k + i * k + ki]) *
                                    ((float)b[l * n * k + j * k + ki] * scale[s_offset + (rowQuant ? j : ki)]);
                        }
                        expect[l * m * n + i * n + j] = to_bf16(temp);
                    }
            }
        });

        kutacc::batched_gemm_pack(bs, m, k, m * k, k, a.get(), pa.get(), 2);
        kutacc::batched_gemm_pack(bs, n, k, n * k, k, b.get(), pb.get(), 1);
        kutacc::batched_gemm_woqs8(bs, m, n, k, m * n, n, pa.get(), pb.get(), c.get(), quantArgs[0], quantArgs[1]);
        for (int64_t i = 0; i < bs * m; i++) {
            for (int64_t j = 0; j < n; j++) {
                if (std::abs(to_float(c[i * n + j]) - expect[i * n + j]) > 0.01) {
                    std::cerr << "(" << i << ", " << j << ") is " << to_float(c[i * n + j]) << ", expect "
                              << to_float(expect[i * n + j]) << ", "
                              << "shape: (" << m << ", " << n << ", " << k << "), "
                              << "rowQuant " << rowQuant << "\n";
                    exit(1);
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    std::vector<std::vector<int64_t>> cases;
    read_args(cases, 4, argc, argv);
    for (auto cas : cases) {
        test_batched_gemm_woqs8(cas);
    }
}
