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

#include "helper.h"
#include "kutacc.h"

void igemm_depack(int64_t m, int64_t n, int64_t tm, int64_t tn, int8_t *src, int8_t *dst)
{
    for (int64_t i = 0; i < m; i += tm) {
        for (int64_t j = 0; j < n; j += tn) {
            int64_t offset = i * n + j * tm;
            for (int64_t x = 0, l = 0; x < tm; x += 16) {
                for (int64_t y = 0; y < tn; y += 4) {
                    for (int64_t z = 0; z < 16 && x + z < tm; ++z, l += 4) {
                        *(int *)(&dst[(i + x + z) * n + (j + y)]) = *(int *)(&src[offset + l]);
                    }
                }
            }
        }
    }
}

void igemm_vertical_shift(int64_t m, int64_t n, int64_t shift_size, int8_t *src, int8_t *dst)
{
    if (shift_size < 0) {
        shift_size = (shift_size % m + m) % m;
    }
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            dst[(i + shift_size) % m * n + j] = src[i * n + j];
        }
    }
}

void test_igemm_pack(std::vector<int64_t> &cas)
{
    int64_t m = cas[0];
    int64_t n = cas[1];
    int64_t tm = cas[2];
    int64_t tn = cas[3];

    std::unique_ptr<int8_t[]> act(new int8_t[m * n]);
    std::unique_ptr<int8_t[]> tmp(new int8_t[m * n]);
    std::unique_ptr<int8_t[]> out(new int8_t[m * n]);
    std::unique_ptr<int[]> idx(new int[m]);

    // generate data
    std::mt19937 rnd(time(0));
    int shift_size = rnd() % n;
    for (int i = 0; i < m; i++) {
        idx[i] = (i + shift_size) % m;
        for (int j = 0; j < n; j++) {
            act[i * n + j] = rnd() % 256;
        }
    }

    // normal cas
    kutacc::igemm_pack(m, n, tm, tn, act.get(), out.get());
    igemm_depack(m, n, tm, tn, out.get(), tmp.get());

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (tmp[i * n + j] != act[i * n + j]) {
                fprintf(stderr, "diff pos in igemm_pack (%d, %d)\n", i, j);
                exit(1);
            }
        }
    }

    // with_idx cas
    kutacc::igemm_pack(m, n, tm, tn, act.get(), out.get(), true, idx.get(), n);
    igemm_depack(m, n, tm, tn, out.get(), tmp.get());
    igemm_vertical_shift(m, n, shift_size, tmp.get(), out.get());

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (out[i * n + j] != act[i * n + j]) {
                fprintf(stderr, "diff pos in igemm_pack_with_idx (%d, %d)\n", i, j);
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
        test_igemm_pack(cas);
    }
}
