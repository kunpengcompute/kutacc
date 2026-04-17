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

#include "../../common.h"

#define MATRIX_ON()                                                                                    \
    do {                                                                                            \
        __asm__ volatile("SMSTART" ::: "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",              \
                                       "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",        \
                                       "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",      \
                                       "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",      \
                                       "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",              \
                                       "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15");       \
        __asm__ volatile("ISB");                                                                    \
    } while (0)

#define MATRIX_OFF()                                                                                   \
    do {                                                                                            \
        __asm__ volatile("SMSTOP" ::: "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",               \
                                      "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",         \
                                      "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",       \
                                      "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",       \
                                      "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",               \
                                      "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15");        \
        __asm__ volatile("ISB");                                                                    \
    } while (0)

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace kutacc {

inline void prefetch_L2(const void *data)
{
    __asm__ __volatile__("prfm PLDL2KEEP, [%[data]]         \n\t"
                         :: [data] "r" (data));
}

inline void prefetch_L1(const void *data)
{
    __asm__ __volatile__("prfm PLDL1STRM, [%[data]]         \n\t"
                         :: [data] "r" (data));
}

template <typename T>
inline void check_max(T &x, const T &y)
{
    if (x < y) {
        x = y;
    }
}

template <typename T>
inline void check_min(T &x, const T &y)
{
    if (x > y) {
        x = y;
    }
}

inline void *alloc_buffer(char *&pool_ptr, int size)
{
    pool_ptr += size;
    return pool_ptr - size;
}

} // namespace kutacc