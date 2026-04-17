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
    if (argc != 5) {
        fprintf(stderr, "argc incorrect\n");
        exit(1);
    }
    int n_tokens = std::atoi(argv[1]);
    int hidden = std::atoi(argv[2]);
    int vocab_start = std::atoi(argv[3]);
    int vocab_end = std::atoi(argv[4]);
    std::unique_ptr<int64_t[]> input(new int64_t[n_tokens]);
    std::unique_ptr<scalar_t[]> weight(new scalar_t[(vocab_end - vocab_start) * hidden]);
    std::unique_ptr<scalar_t[]> out(new scalar_t[n_tokens * hidden]);
    std::unique_ptr<scalar_t[]> expect(new scalar_t[n_tokens * hidden]);

    std::mt19937 rnd(time(0));

    for (int i = 0; i < n_tokens; i++) {
        input[i] = rnd() % 2 == 0 ? rnd() % (vocab_end - vocab_start) + vocab_start : rnd() % 129280;
    }
    for (int i = 0; i < vocab_end - vocab_start; i++) {
        for (int j = 0; j < hidden; j++) {
            weight[i * hidden + j] = rnd() * 1.0 / rnd.max();
        }
    }
    for (int i = 0; i < n_tokens; i++) {
        for (int j = 0; j < hidden; j++) {
            out[i * hidden + j] = expect[i * hidden + j] = rnd() * 1.0 / rnd.max();
        }
    }

    for (int i = 0; i < n_tokens; i++) {
        if (input[i] >= vocab_start && input[i] < vocab_end) {
            memcpy(expect.get() + i * hidden, weight.get() + (input[i] - vocab_start) * hidden,
                hidden * sizeof(scalar_t));
        } else {
            memset(expect.get() + i * hidden, 0, hidden * sizeof(scalar_t));
        }
    }

    kutacc::embedding(input.get(), weight.get(), out.get(), sizeof(scalar_t), n_tokens, hidden, vocab_start,
        vocab_end);

    for (int i = 0; i < n_tokens; i++) {
        if (memcmp(out.get() + i * hidden, expect.get() + i * hidden, hidden * sizeof(scalar_t)) != 0) {
            fprintf(stderr, "token %d diff\n", i);
            exit(1);
        }
    }
}
