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

#include <arm_bf16.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "utils.h"

#define CHECK_CONTIGUOUS(x) FLASH_ASSERT(x.is_contiguous())

constexpr int64_t MAX_DIM = 8;

enum class ScalarType {
    kUndefined,
    kHalf,
    kBFloat16,
    kFloat,
    kChar,
    kByte,
    kInt,
    kLong,
};

inline size_t elementSize(ScalarType dtype)
{
    if (dtype == ScalarType::kChar || dtype == ScalarType::kByte) {
        return 1;
    } else if (dtype == ScalarType::kHalf || dtype == ScalarType::kBFloat16) {
        return 2;
    } else if (dtype == ScalarType::kFloat) {
        return sizeof(float);
    } else if (dtype == ScalarType::kInt) {
        return sizeof(int);
    } else if (dtype == ScalarType::kLong) {
        return sizeof(int64_t);
    } else {
        FLASH_ASSERT(false);
    }
}

struct alignas(64) Tensor {
    void *ptr{nullptr};
    ScalarType _dtype{ScalarType::kUndefined};
    int64_t dim{};
    int64_t sizes[MAX_DIM];
    int64_t strides[MAX_DIM];

    template <typename scalar_t>
    scalar_t *data_ptr() const
    {
        if constexpr (!std::is_same_v<scalar_t, void>) {
            if constexpr (std::is_same_v<scalar_t, __fp16>) {
                FLASH_ASSERT(_dtype == ScalarType::kHalf);
            } else if constexpr (std::is_same_v<scalar_t, __bf16>) {
                FLASH_ASSERT(_dtype == ScalarType::kBFloat16);
            } else if constexpr (std::is_same_v<scalar_t, float>) {
                FLASH_ASSERT(_dtype == ScalarType::kFloat);
            } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
                FLASH_ASSERT(_dtype == ScalarType::kChar);
            } else if constexpr (std::is_same_v<scalar_t, uint8_t>) {
                FLASH_ASSERT(_dtype == ScalarType::kByte);
            } else if constexpr (std::is_same_v<scalar_t, int>) {
                FLASH_ASSERT(_dtype == ScalarType::kInt);
            } else if constexpr (std::is_same_v<scalar_t, int64_t>) {
                FLASH_ASSERT(_dtype == ScalarType::kLong);
            } else {
                FLASH_ASSERT(false);
            }
        }
        return reinterpret_cast<scalar_t *>(ptr);
    }

    ScalarType dtype() const
    {
        return _dtype;
    }
    int64_t numel() const
    {
        int64_t mul = 1;
        for (int64_t i = 0; i < dim; i++) {
            mul *= sizes[i];
        }
        return mul;
    }
    int64_t size(int64_t i) const
    {
        if (i < 0) {
            i += dim;
        }
        return sizes[i];
    }
    int64_t stride(int64_t i) const
    {
        if (i < 0) {
            i += dim;
        }
        return strides[i];
    }
    bool is_contiguous() const
    {
        for (int i = 0; i < dim - 1; ++i) {
            if (strides[i] != strides[i + 1] * sizes[i + 1]) {
                return false;
            }
        }
        return strides[dim - 1] == 1;
    }
    static Tensor inplace_create(ScalarType dtype, const std::vector<int64_t> &sizes, char *&ptr);
    static Tensor view(const Tensor &old, const std::vector<int64_t> &new_sizes);
};

inline Tensor Tensor::inplace_create(ScalarType dtype, const std::vector<int64_t> &sizes, char *&ptr)
{
    Tensor result{};
    result._dtype = dtype;
    result.dim = sizes.size();
    int64_t numel = 1;
    for (int64_t i = result.dim - 1; i >= 0; i--) {
        result.sizes[i] = sizes[i];
        result.strides[i] = numel;
        numel *= sizes[i];
    }
    result.ptr = ptr;
    ptr += (numel * elementSize(dtype) + 63) / 64 * 64;
    return result;
}

inline Tensor Tensor::view(const Tensor &old, const std::vector<int64_t> &new_sizes)
{
    Tensor result{};
    result.ptr = old.ptr;
    result.dim = new_sizes.size();
    result._dtype = old._dtype;
    for (int64_t i = 0; i < result.dim; i++) {
        result.sizes[i] = new_sizes[i];
    }
    int64_t view_d = result.dim - 1;
    int64_t chunk_base_stride = old.strides[old.dim - 1];
    int64_t tensor_numel = 1;
    int64_t view_numel = 1;
    for (int64_t tensor_d = old.dim - 1; tensor_d >= 0; tensor_d--) {
        tensor_numel *= old.sizes[tensor_d];
        if ((tensor_d == 0) ||
            (old.sizes[tensor_d - 1] != 1 && old.strides[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
            while (view_d >= 0 && (view_numel < tensor_numel || new_sizes[view_d] == 1)) {
                result.strides[view_d] = view_numel * chunk_base_stride;
                view_numel *= new_sizes[view_d];
                view_d--;
            }
            FLASH_ASSERT(view_numel == tensor_numel);
            if (tensor_d > 0) {
                chunk_base_stride = old.strides[tensor_d - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    FLASH_ASSERT(view_d == -1);
    return result;
}