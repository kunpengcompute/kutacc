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

#include <ctime>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "kutacc.h"
#include "timer.h"

namespace kutacc {

static uint64_t g_ticksPerMicrosecond = 1;

/**
 * @brief ARM architecture hardware monotonic
 */
static kutacc_always_inline
uint64_t kutacc_arm_virtual_timer_count()
{
    uint64_t counter;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(counter));
    return counter;
}

static kutacc_always_inline
uint32_t kutacc_arm_virtual_timer_freq()
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

void kutacc_time_init()
{
    uint64_t tmp = static_cast<uint64_t>(kutacc_arm_virtual_timer_freq()) / S2U_FACTOR;
    if (tmp != 0) {
        g_ticksPerMicrosecond = tmp;
    }
}

uint64_t kutacc_now_ns()
{
    return kutacc_arm_now_ns();
}

}