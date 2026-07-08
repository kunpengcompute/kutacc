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

#include <cstdint>
#include <vector>
#include <cstdlib>
#include <string>

#include <arm_neon.h>

inline bfloat16_t to_bf16(float x)
{
    return vcvth_bf16_f32(x);
}

inline float to_float(bfloat16_t x)
{
    return vcvtah_f32_bf16(x);
}

inline void read_args(std::vector<std::vector<int64_t>> &cases, int64_t args_num, int64_t argc, char **argv)
{
    if (argc == 1) {
        return;
    } else if ((argc - 1) % args_num != 0) {
        puts("Wrong args_num");
        std::exit(-1);
    } else {
        cases.clear();
        for (int64_t i = 1; i < argc;) {
            std::vector<int64_t> cas;
            for (int64_t j = 0; j < args_num; ++j, ++i) {
                cas.push_back(std::stoi(argv[i]));
            }
            cases.push_back(cas);
        }
    }
}

void read_args_skip_n(std::vector<std::vector<int64_t>> &cases, int64_t args_num, int64_t argc, char **argv, int64_t n)
{
    int begin_idx = n + 1;
    if (argc <= begin_idx) {
        return;
    } else if ((argc - begin_idx) % args_num != 0) {
        puts("Wrong args_num");
        std::exit(-1);
    } else {
        cases.clear();
        for (int64_t i = begin_idx; i < argc;) {
            std::vector<int64_t> cas;
            for (int64_t j = 0; j < args_num; ++j, ++i) {
                cas.push_back(std::stoi(argv[i]));
            }
            cases.push_back(cas);
        }
    }
}

inline void print(bfloat16_t *c, int64_t m, int64_t n, std::string name)
{
    printf("%s:\n", name.c_str());
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.3f%c", to_float(c[i * n + j]), " \n"[j == n - 1]);
        }
    }
}

#define VTC_FACTOR  1000
#define S2U_FACTOR  1000000         // 1s = 1000000us
#define S2N_FACTOR  1000000000      // 1s = 1000000000ns
#define kutacc_always_inline             inline __attribute__((always_inline))

static uint64_t g_ticksPerMicrosecond = 1;

/**
 * @brief ARM architecture hardware monotonic
 */
static kutacc_always_inline uint64_t kutacc_arm_virtual_timer_count()
{
    uint64_t counter;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(counter));
    return counter;
}

static kutacc_always_inline uint32_t kutacc_arm_virtual_timer_freq()
{
    union {
        uint64_t freq;
        struct {
            uint32_t low;
            uint32_t high;
        } bits;
    } counter;

    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(counter.freq));
    return counter.bits.low;    /* Top 32 bits are reserved */
}

static uint64_t kutacc_arm_now_ns()
{
    uint64_t count = kutacc_arm_virtual_timer_count();
    return count * VTC_FACTOR / g_ticksPerMicrosecond;
}

inline void kutacc_time_init()
{
    uint64_t tmp = static_cast<uint64_t>(kutacc_arm_virtual_timer_freq()) / S2U_FACTOR;
    if (tmp != 0) {
        g_ticksPerMicrosecond = tmp;
    }
}

inline uint64_t kutacc_now_ns()
{
    return kutacc_arm_now_ns();
}