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
#include <vector>
#include <cstdio>

#include "helper.h"
#include "kutacc.h"

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

void test_bgemm_pack(std::vector<int64_t> &cas)
{
    int64_t m = cas[0];
    int64_t n = cas[1];
    int64_t tm = cas[2];
    int64_t tn = cas[3];

    std::unique_ptr<bfloat16_t[]> act(new bfloat16_t[m * n]);
    std::unique_ptr<bfloat16_t[]> tmp(new bfloat16_t[m * n]);
    std::unique_ptr<bfloat16_t[]> out(new bfloat16_t[m * n]);

    // generate data
    std::mt19937 rnd(time(0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            act[i * n + j] = to_bf16(rnd() * 1.0 / rnd.max());
        }
    }

    // bgemm_pack cas
    kutacc::bgemm_pack(m, n, tm, tn, act.get(), out.get());
    bgemm_depack(m, n, tm, tn, out.get(), tmp.get());
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (tmp[i * n + j] != act[i * n + j]) {
                fprintf(stderr, "diff pos in bgemm_pack (%d, %d)\n", i, j);
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
        test_bgemm_pack(cas);
    }
}
