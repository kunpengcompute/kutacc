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

#include <iostream>
#include <unistd.h>

#include "utils/check.h"
#include "utils/timer.h"

#include "config.h"

#ifdef FLASH_MLA_ENABLE_CHECK
#define FLASH_MLA_CHECK(condition, ...) KUTACC_CHECK(condition, __VA_ARGS__)
#else
#define FLASH_MLA_CHECK(condition, ...) ((void)(condition))
#endif

#define FLASH_MLA_CHECK_SHAPE(x, ...)                                                                               \
    FLASH_MLA_CHECK(((x).sizes() == decltype((x).sizes()){__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define FLASH_MLA_CHECK_CONTIGUOUS(x)                                                                               \
    FLASH_MLA_CHECK(((x).is_contiguous()), #x " must be contiguous")

#define FLASH_MLA_CHECK_LAST_DIM_CONTIGUOUS(x)                                                                      \
    FLASH_MLA_CHECK(((x).size(-1) == 1 || (x).stride(-1) == 1), #x " must have contiguous last dimension")

#define FLASH_MLA_DEBUG(x)                                                                                          \
    [&] {                                                                                                           \
        std::cerr << #x << " = " << (x) << std::endl;                                                               \
    }()

#define FLASH_MLA_BOOL_SWITCH(COND, NAME, ...)                                                                      \
    [&] {                                                                                                           \
        if (COND) {                                                                                                 \
            static constexpr bool NAME = true;                                                                      \
            return __VA_ARGS__();                                                                                   \
        } else {                                                                                                    \
            static constexpr bool NAME = false;                                                                     \
            return __VA_ARGS__();                                                                                   \
        }                                                                                                           \
    }()

#define FLASH_MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)                                                          \
    [&] {                                                                                                           \
        if (NUM_SPLITS <= 32) {                                                                                     \
            static constexpr int NAME = 32;                                                                         \
            return __VA_ARGS__();                                                                                   \
        } else if (NUM_SPLITS <= 64) {                                                                              \
            static constexpr int NAME = 64;                                                                         \
            return __VA_ARGS__();                                                                                   \
        } else if (NUM_SPLITS <= 96) {                                                                              \
            static constexpr int NAME = 96;                                                                         \
            return __VA_ARGS__();                                                                                   \
        } else if (NUM_SPLITS <= 128) {                                                                             \
            static constexpr int NAME = 128;                                                                        \
            return __VA_ARGS__();                                                                                   \
        } else if (NUM_SPLITS <= 160) {                                                                             \
            static constexpr int NAME = 160;                                                                        \
            return __VA_ARGS__();                                                                                   \
        } else {                                                                                                    \
            FLASH_MLA_CHECK(false, "");                                                                             \
        }                                                                                                           \
    }()

namespace kutacc {

namespace flash_mla {

template <typename T>
constexpr T ceil_div(const T &a, const T &b)
{
    return (a + b - 1) / b;
}

template <typename T>
constexpr T ceil(const T &a, const T &b)
{
    return ceil_div(a, b) * b;
}

template <int64_t N>
constexpr std::array<int64_t, N> get_indices(int64_t idx, const std::array<int64_t, N> &sizes)
{
    std::array<int64_t, N> res;
    for (int64_t i = N - 1; i >= 0; --i) {
        res[i] = idx % sizes[i];
        idx /= sizes[i];
    }
    return res;
}

inline auto get_clock_us()
{
    static bool init = false;
    if (!init) {
        kutacc_time_init();
        init = true;
    }
    return kutacc_now_ns() / 1000.0;
}

} // namespace flash_mla

} // namespace kutacc