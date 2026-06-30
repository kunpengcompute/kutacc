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

#include <memory>
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iostream>
#include <kutacc.h>
#include "helper.h"

#define FLASH_ASSERT(cond)                                                                                      \
    do {                                                                                                        \
        if (__builtin_expect(!(cond), 0)) {                                                                     \
            std::cerr << "Assertion failed (" << __FILE__ << ":" << __LINE__ << "): " << #cond << std::endl;    \
            exit(1);                                                                                            \
        }                                                                                                       \
    } while (0)

#define CHECK_SHAPE(x, ...) FLASH_ASSERT(((x).sizes() == decltype((x).sizes()){__VA_ARGS__}))

#define CHECK_CONTIGUOUS(x) FLASH_ASSERT(((x).is_contiguous()))

#define CHECK_LAST_DIM_CONTIGUOUS(x) FLASH_ASSERT(((x).size(-1) == 1 || (x).stride(-1) == 1))

template <typename T>
constexpr T ceil_div(const T &x, const T &y)
{
    return (x + y - 1) / y;
}

template <typename T>
constexpr T ceil(const T &x, const T &y)
{
    return ceil_div(x, y) * y;
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

constexpr int64_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;

template <bool on_package = true>
void *mmap_huge_page_memory(int64_t size)
{
    FLASH_ASSERT(size % HUGE_PAGE_SIZE == 0);
    void *addr =
        mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS |
            (on_package ? MAP_HUGETLB : 0) | MAP_POPULATE, -1, 0);
    int cpu = sched_getcpu();
    int rnode = numa_node_of_cpu(cpu) + (on_package ? 16 : 0);
    unsigned long mask = 1UL << rnode;
    int success = mbind(addr, size, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_STRICT | MPOL_MF_MOVE);
    FLASH_ASSERT(success == 0);
    return addr;
}

template <typename T>
bfloat16_t *get_token_in_kvcache(const T &kvcache, int64_t idx)
{
    if (idx == -1) {
        return nullptr;
    }
    FLASH_ASSERT(idx < kvcache.size(0) * kvcache.size(1));
    return kvcache.data_ptr() + idx / kvcache.size(1) * kvcache.stride(0) + idx % kvcache.size(1) * kvcache.stride(1);
}

template <typename T, int64_t N>
bool check_is_allclose(
    const kutacc::Tensor<T, N> &ans,
    const kutacc::Tensor<T, N> &ref,
    double abs_tol = 1e-5,
    double rel_tol = 1e-2,
    double cos_diff_tol = 1e-7
)
{
    if (ans.sizes() != ref.sizes()) {
        return false;
    }
    const auto &sizes = ans.sizes();
    int64_t numel = ans.numel();

    double sum_xy = 0;
    double sum_x2_y2 = 0;
    for (int64_t i = 0; i < numel; ++i)  {
        auto indices = get_indices<N>(i, sizes);
        double x = ans.index(indices);
        double y = ref.index(indices);
        if (isnan(x) != isnan(y)) {
            return false;
        }
        if ((x == INFINITY) != (y == INFINITY)) {
            return false;
        }
        if ((x == -INFINITY) != (y == -INFINITY)) {
            return false;
        }
        if (isnan(x) || isinf(x)) {
            continue;
        }
        sum_xy += x * y;
        sum_x2_y2 += x * x + y * y;

        double abs_err = std::abs(x - y);
        double rel_err = abs_err / (std::abs(y) + 1e-6);
        if (abs_err >= abs_tol && rel_err >= rel_tol) {
            return false;
        }
    }
    double cos_diff = 1 - 2 * sum_xy / sum_x2_y2;
    return cos_diff < cos_diff_tol;
}