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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "scalar_type.h"
#include "check.h"
#include "span.h"

namespace utils {
constexpr int64_t MAX_DIM = 8;

struct alignas(64) Tensor {
    void *ptr{};
    ScalarType _dtype{};
    int64_t dim{};
    int64_t sizes[MAX_DIM];
    int64_t strides[MAX_DIM];

    template <typename scalar_t = void>
    scalar_t *data_ptr() const
    {
        if constexpr (!std::is_same_v<scalar_t, void>) {
            if constexpr (std::is_same_v<scalar_t, __fp16>) {
                PARAMETER_CHECK(_dtype == ScalarType::Float16, int(_dtype));
            } else if constexpr (std::is_same_v<scalar_t, __bf16>) {
                PARAMETER_CHECK(_dtype == ScalarType::BFloat16, int(_dtype));
            } else if constexpr (std::is_same_v<scalar_t, float>) {
                PARAMETER_CHECK(_dtype == ScalarType::Float32, int(_dtype));
            } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
                PARAMETER_CHECK(_dtype == ScalarType::Int8, int(_dtype));
            } else if constexpr (std::is_same_v<scalar_t, uint8_t>) {
                PARAMETER_CHECK(_dtype == ScalarType::UInt8, int(_dtype));
            } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
                PARAMETER_CHECK(_dtype == ScalarType::Int16, int(_dtype));
            } else if constexpr (std::is_same_v<scalar_t, int>) {
                PARAMETER_CHECK(_dtype == ScalarType::Int32, int(_dtype));
            } else if constexpr (std::is_same_v<scalar_t, int64_t>) {
                PARAMETER_CHECK(_dtype == ScalarType::Int64, int(_dtype));
            } else {
                PARAMETER_CHECK(false, int(_dtype));
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

    template <typename scalar_t>
    void fill(scalar_t el)
    {
        auto data = data_ptr<scalar_t>();
        std::fill(data, data + numel(), el);
    }

    static Tensor create(ScalarType dtype, const std::vector<int64_t> &sizes, std::unique_ptr<uint8_t[]> &hold);
    static Tensor alloc_from(ScalarType dtype, const std::vector<int64_t> &sizes, u8span &hold);
    static Tensor from_blob(ScalarType dtype, const std::vector<int64_t> &sizes, void *blob);
    static Tensor transpose(const Tensor &old, int64_t dim1, int64_t dim2);
    static Tensor view(const Tensor &old, const std::vector<int64_t> &new_sizes);
    static Tensor select(const Tensor &old, int64_t dim, int64_t index);
    static Tensor slice(const Tensor &old, int64_t dim, int64_t start, int64_t end, int64_t step = 1);
    static Tensor reinterpret(const Tensor &old, ScalarType dtype);
};

inline Tensor Tensor::create(ScalarType dtype, const std::vector<int64_t> &sizes, std::unique_ptr<uint8_t[]> &hold)
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
    hold.reset(new uint8_t[numel * elementSize(dtype)]);
    result.ptr = hold.get();
    return result;
}

inline Tensor Tensor::alloc_from(ScalarType dtype, const std::vector<int64_t> &sizes, u8span &hold)
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
    int64_t size = numel * elementSize(dtype);
    PARAMETER_CHECK(size > 0, size);
    PARAMETER_CHECK(size <= hold.size, "need ", size, " remain ", hold.size);
    result.ptr = hold.ptr;
    hold = hold.subspan(size);
    return result;
}

inline Tensor Tensor::from_blob(ScalarType dtype, const std::vector<int64_t> &sizes, void *blob)
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
    result.ptr = blob;
    return result;
}

inline Tensor Tensor::transpose(const Tensor &old, int64_t dim1, int64_t dim2)
{
    Tensor result{};
    result.ptr = old.ptr;
    result.dim = old.dim;
    result._dtype = old._dtype;
#pragma unroll
    for (int64_t i = 0; i < MAX_DIM; i++) {
        result.sizes[i] = old.sizes[i];
        result.strides[i] = old.strides[i];
    }
    std::swap(result.sizes[dim1], result.sizes[dim2]);
    std::swap(result.strides[dim1], result.strides[dim2]);
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
            PARAMETER_CHECK(view_numel == tensor_numel, "utils::Tensor::view");
            if (tensor_d > 0) {
                chunk_base_stride = old.strides[tensor_d - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    PARAMETER_CHECK(view_d == -1, -1);
    return result;
}

inline Tensor Tensor::select(const Tensor &old, int64_t dim, int64_t index)
{
    PARAMETER_CHECK(dim < old.dim, dim, " ", old.dim);
    Tensor result{};
    result._dtype = old._dtype;
    result.dim = old.dim - 1;
    result.ptr = (uint8_t *)old.ptr + old.strides[dim] * index * elementSize(old._dtype);
#pragma unroll
    for (int64_t i = 0; i < MAX_DIM - 1; i++) {
        result.sizes[i] = old.sizes[i < dim ? i : i + 1];
        result.strides[i] = old.strides[i < dim ? i : i + 1];
    }
    return result;
}

inline Tensor Tensor::slice(const Tensor &old, int64_t dim, int64_t start, int64_t end, int64_t step)
{
    PARAMETER_CHECK(dim < old.dim, "utils::Tensor::slice dim");
    PARAMETER_CHECK(start <= end, "utils::Tensor::slice start end");
    Tensor result{};
    result._dtype = old._dtype;
    result.dim = old.dim;
    result.ptr = (uint8_t *)old.ptr + old.strides[dim] * start * elementSize(old._dtype);
#pragma unroll
    for (int64_t i = 0; i < MAX_DIM; i++) {
        result.sizes[i] = old.sizes[i];
        result.strides[i] = old.strides[i];
    }
    result.sizes[dim] = (end - start + step - 1) / step;
    result.strides[dim] *= step;
    return result;
}

inline Tensor Tensor::reinterpret(const Tensor &old, ScalarType dtype)
{
    PARAMETER_CHECK(old.dim == 1, old.dim);
    PARAMETER_CHECK(old._dtype == ScalarType::UInt8, int(old._dtype));
    PARAMETER_CHECK(old.sizes[0] % elementSize(dtype) == 0, old.sizes[0], int(old._dtype));
    Tensor result{};
    result._dtype = dtype;
    result.dim = 1;
    result.ptr = old.ptr;
    result.sizes[0] = old.sizes[0] / elementSize(dtype);
    result.strides[0] = 1;
    return result;
}

struct TensorWithMemory {
    Tensor tensor;
    std::unique_ptr<uint8_t[]> memory;
    static TensorWithMemory create(ScalarType dtype, const std::vector<int64_t> &sizes)
    {
        TensorWithMemory result;
        *result = Tensor::create(dtype, sizes, result.memory);
        return result;
    }
    Tensor &operator*()
    {
        return tensor;
    }
    const Tensor &operator*() const
    {
        return tensor;
    }
    Tensor *operator->()
    {
        return &tensor;
    }
    const Tensor *operator->() const
    {
        return &tensor;
    }
};
}  // namespace utils
