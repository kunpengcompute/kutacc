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
#pragma once

#define MATRIX_ON()                                                                                 \
    do {                                                                                            \
        __asm__ volatile("SMSTART" ::: "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",              \
                                       "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",        \
                                       "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",      \
                                       "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",      \
                                       "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",              \
                                       "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15");       \
        __asm__ volatile("ISB");                                                                    \
    } while (0)

#define MATRIX_OFF()                                                                                \
    do {                                                                                            \
        __asm__ volatile("SMSTOP" ::: "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",               \
                                      "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",         \
                                      "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",       \
                                      "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",       \
                                      "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",               \
                                      "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15");        \
        __asm__ volatile("ISB");                                                                    \
    } while (0)

#define SVLDNT1_BF16(pg, base)                                      \
    [&] {                                                           \
        svbfloat16_t res;                                           \
        __asm__ volatile (                                          \
            "ldnt1h { %0.h }, %1/z, [%2]\n"                         \
            : "=w" (res)                                            \
            : "Upl" (pg), "r" (base)                                \
            : "memory"                                              \
        );                                                          \
        return res;                                                 \
    }()

#define SVLDNT1_INDEX_BF16(pg, base, index)                         \
    [&] {                                                           \
        svbfloat16_t res;                                           \
        __asm__ volatile (                                          \
            "ldnt1h { %0.h }, %1/z, [%2, %3, lsl #1]\n"             \
            : "=w" (res)                                            \
            : "Upl" (pg), "r" (base), "r" ((int64_t)(index))          \
            : "memory"                                              \
        );                                                          \
        return res;                                                 \
    }()

#define likely(x) __builtin_expect(!!(x), 1)

#define unlikely(x) __builtin_expect(!!(x), 0)

namespace kutacc {

namespace flash_mla {

template <typename T>
constexpr void check_max(T &x, const T &y)
{
    if (x < y) {
        x = y;
    }
}

template <typename T>
constexpr void check_min(T &x, const T &y)
{
    if (x > y) {
        x = y;
    }
}

inline void kv_back_check(const int *indices, int64_t &begin, int64_t &end)
{
    while (begin < end && indices[end - 1] == -1) {
        --end;
    }
    if (begin >= end) {
        begin = end = 0;
    }
}

template <int head_dim>
void get_kv_token_addr(const bfloat16_t *kvcache, const int *indices, bfloat16_t **kv_token_addr, int64_t len)
{
    for (int64_t i = 0; i < len; i += 16) {
        svbool_t p0 = svwhilelt_b32(i, len);
        svint32_t sve0 = svld1(p0, indices + i);
        sve0 = svmul_m(p0, sve0, head_dim * sizeof(bfloat16_t));
        svint64_t sve1 = svunpklo(sve0);
        svint64_t sve2 = svunpkhi(sve0);
        svbool_t p1 = svwhilelt_b64(i, len);
        svbool_t p2 = svwhilelt_b64(i + 8, len);
        sve1 = svadd_m(p1, sve1, reinterpret_cast<int64_t>(kvcache));
        sve2 = svadd_m(p2, sve2, reinterpret_cast<int64_t>(kvcache));
        svst1(p1, reinterpret_cast<int64_t *>(kv_token_addr) + i, sve1);
        svst1(p2, reinterpret_cast<int64_t *>(kv_token_addr) + i + 8, sve2);
    }
}

} // namespace flash_mla

} // namespace kutacc