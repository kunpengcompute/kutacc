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

#include <arm_neon.h>
#include <arm_sve.h>

#include "helper.h"
#include "kutacc.h"

void test_bgemm(std::vector<int64_t> &cas)
{
    int64_t m = cas[0];
    int64_t n = cas[1];
    int64_t k = cas[2];
    kutacc::MatrixTilingBlock t(cas[3], cas[4], cas[5]);
    std::unique_ptr<bfloat16_t[]> a(new bfloat16_t[m * k]);
    std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[m * k]);
    std::unique_ptr<bfloat16_t[]> b(new bfloat16_t[k * n]);
    std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[k * n]);
    std::unique_ptr<bfloat16_t[]> c(new bfloat16_t[m * n]);
    std::unique_ptr<bfloat16_t[]> tmpc(new bfloat16_t[m * n]);
    std::unique_ptr<bfloat16_t[]> expect(new bfloat16_t[m * n]);

    std::mt19937 rnd(time(0));
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < k; j++) {
            a[i * k + j] = ((int64_t)rnd() % 4 - 2) / sqrt(k);
        }
    }
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < k; j++) {
            b[i * k + j] = ((int64_t)rnd() % 4 - 2) / sqrt(k);
        }
    }

    kutacc::bgemm_pack(m, k, std::get<0>(t), std::get<2>(t), a.get(), pack_a.get());
    kutacc::bgemm_pack(n, k, std::get<1>(t), std::get<2>(t), b.get(), pack_b.get());
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            c[i * n + j] = expect[i * n + j] = rnd() % 2;
        }
    }
    kutacc::parallel_for((int64_t)0, m, (int64_t)1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            for (int64_t j = 0; j < n; j++) {
                float temp = 0;
                for (int64_t ki = 0; ki < k; ki++) {
                    temp += (float)a[i * k + ki] * b[j * k + ki];
                }
                expect[i * n + j] = to_bf16(temp);
            }
        }
    });
    kutacc::bgemm(m, n, k, t, pack_a.get(), pack_b.get(), c.get(), tmpc.get());

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float cans = to_float(c[i * n + j]);
            float eans = to_float(expect[i * n + j]);
            if (std::abs(cans - eans) > 0.1) {
                std::cerr << "(" << i << ", " << j << ") is " << cans << ", expect " << eans << ", "
                          << "shape: (" << m << ", " << n << ", " << k << ")"
                          << "\n";
                exit(1);
            }
        }
    }
}

int main(int argc, char **argv)
{
    std::vector<std::vector<int64_t>> cases = {{128, 8080, 7168, 32, 1010, 7168}};
    read_args(cases, 6, argc, argv);
    for (auto cas : cases) {
        test_bgemm(cas);
    }
}
