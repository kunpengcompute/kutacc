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

constexpr int64_t DEFAULT_ALIGNMENT = 64;

struct Allocator {
    void *begin_addr;
    void *cur_addr;
    int64_t max_size = 0;

    template <bool on_package = true>
    Allocator(int64_t max_size_)
    {
        max_size = max_size_;
        begin_addr = cur_addr = mmap_huge_page_memory<on_package>(max_size);
    }

    ~Allocator()
    {
        munmap(begin_addr, max_size);
    }

    int64_t remain()
    {
        return max_size - (reinterpret_cast<int64_t>(cur_addr) - reinterpret_cast<int64_t>(begin_addr));
    }

    void *alloc(int64_t size, int64_t alignment = DEFAULT_ALIGNMENT)
    {
        void *addr = reinterpret_cast<void *>(ceil(reinterpret_cast<int64_t>(cur_addr), alignment));
        FLASH_ASSERT((int8_t *)addr + size <= (int8_t *)begin_addr + max_size);
        cur_addr = (int8_t *)addr + size;
        return addr;
    }

    void reset()
    {
        cur_addr = begin_addr;
    }
};

template <typename T, int64_t N>
inline kutacc::Tensor<T, N> create_tensor(Allocator &allocator,
    const std::array<int64_t, N> &sizes, int64_t aligment = DEFAULT_ALIGNMENT)
{
    auto tensor = kutacc::Tensor<T, N>((T *)allocator.alloc(0), sizes);
    allocator.alloc(tensor.numel() * sizeof(T), aligment);
    return tensor;
}

template <typename T, int64_t N>
inline kutacc::Tensor<T, N> generate_tensor(Allocator &allocator, const std::array<int64_t, N> &sizes,
    const std::function<T()> &generator, int64_t aligment = DEFAULT_ALIGNMENT)
{
    auto tensor = create_tensor<T, N>(allocator, sizes, aligment);
    int64_t numel = tensor.numel();
    for (int64_t i = 0; i < numel; ++i) {
        tensor.data_ptr()[i] = generator();
    }
    return tensor;
}