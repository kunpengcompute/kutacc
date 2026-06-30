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

#include "utils.h"

struct CacheFlush {
    static constexpr int64_t ALIGNMENT = 64;
    std::vector<int8_t> xor_sum;
    int64_t size;
    void *buffer;

    CacheFlush(int64_t size_per_thread)
    {
        size = size_per_thread * kutacc::get_thread_num();
        buffer = mmap_huge_page_memory(size);
        memset(buffer, 0, size);
        xor_sum.resize(kutacc::get_thread_num() * ALIGNMENT, 0);
    }

    ~CacheFlush()
    {
        for (auto x : xor_sum) {
            FLASH_ASSERT(x == 0);
        }
        munmap(buffer, size);
    }

    void operator ()()
    {
        kutacc::parallel_for(0, size, 1, [&](int64_t begin, int64_t end) {
            int tid = kutacc::get_thread_id();
            for (int64_t i = begin; i < end; ++i) {
                xor_sum[tid * ALIGNMENT] ^= ((int8_t *)buffer)[i];
            }
        });
    }
};