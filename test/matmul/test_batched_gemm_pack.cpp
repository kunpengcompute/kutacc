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
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <arm_bf16.h>

#include "helper.h"
#include "kutacc.h"

void igemm_depack(int64_t m, int64_t n, int64_t tm, int64_t tn, int8_t *src, int8_t *dst)
{
    for (int64_t i = 0; i < m; i += tm) {
        for (int64_t j = 0; j < n; j += tn) {
            int64_t offset = i * n + j * tm;
            for (int64_t x = 0, l = 0; x < tm; x += 32) {
                for (int64_t y = 0; y < tn; y += 2) {
                    for (int64_t z = 0; z < 32 && x + z < tm; ++z, l += 2) {
                        *(int16_t *)(&dst[(i + x + z) * n + (j + y)]) = *(int16_t *)(&src[offset + l]);
                    }
                }
            }
        }
    }
}

void bgemm_depack(int64_t m, int64_t n, int64_t tm, int64_t tn, bfloat16_t *src, bfloat16_t *dst)
{
    for (int64_t i = 0; i < m; i += tm) {
        for (int64_t j = 0; j < n; j += tn) {
            int64_t offset = i * n + j * tm;
            for (int64_t x = 0, l = 0; x < tm; x += 16) {
                for (int64_t y = 0; y < tn; y += 2) {
                    for (int64_t z = 0; z < 16 && x + z < tm; ++z, l += 2) {
                        *(int *)(&dst[(i + x + z) * n + (j + y)]) = *(int *)(&src[offset + l]);
                    }
                }
            }
        }
    }
}

void test_batched_gemm_pack(std::vector<int64_t> &cas)
{
    int64_t bs = cas[0];
    int64_t m = cas[1];
    int64_t n = cas[2];
    int64_t k = cas[3];
    assert(m % 32 == 0);
    std::unique_ptr<int8_t[]> a(new int8_t[bs * m * k]);
    std::unique_ptr<int8_t[]> pa(new int8_t[bs * m * k]);
    std::unique_ptr<int8_t[]> tmpa(new int8_t[bs * m * k]);
    std::unique_ptr<bfloat16_t[]> b(new bfloat16_t[bs * k * n]);
    std::unique_ptr<bfloat16_t[]> pb(new bfloat16_t[bs * k * n]);
    std::unique_ptr<bfloat16_t[]> tmpb(new bfloat16_t[bs * k * n]);

    std::mt19937 rnd(time(0));
    for (int i = 0; i < bs * m; i++) {
        for (int j = 0; j < k; j++) {
            a[i * k + j] = ((int64_t)rnd() % 3 - 1);
        }
    }
    for (int i = 0; i < bs * n; i++) {
        for (int j = 0; j < k; j++) {
            b[i * k + j] = to_bf16(((int64_t)rnd() % 4 - 2) * 1.0 / k);
        }
    }

    kutacc::batched_gemm_pack(bs, m, k, m * k, k, a.get(), pa.get(), 1);
    igemm_depack(bs * m, k, bs * m, k, pa.get(), tmpa.get());
    for (int i = 0; i < bs * m; i++) {
        for (int j = 0; j < k; j++) {
            if (tmpa[i * k + j] != a[i * k + j]) {
                fprintf(stderr, "diff pos in batched_gemm_pack A (%d, %d)\n", i, j);
                exit(1);
            }
        }
    }

    kutacc::batched_gemm_pack(bs, n, k, n * k, k, b.get(), pb.get(), 2);
    bgemm_depack(bs * n, k, bs * n, k, pb.get(), tmpb.get());
    for (int i = 0; i < bs * n; i++) {
        for (int j = 0; j < k; j++) {
            if (tmpb[i * k + j] != b[i * k + j]) {
                fprintf(stderr, "diff pos in batched_gemm_pack B (%d, %d)\n", i, j);
                exit(1);
            }
        }
    }
}

int main(int argc, char **argv)
{
    std::vector<std::vector<int64_t>> cases;
    read_args(cases, 4, argc, argv);
    for (auto cas : cases) {
        test_batched_gemm_pack(cas);
    }
}
