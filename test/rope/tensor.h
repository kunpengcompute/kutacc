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

#include "check.h"
#include "scalar_type.h"
#include "span.h"

constexpr int64_t MAX_DIM = 8;

static inline uint8_t* alignup(uint8_t* addr, int64_t alignment)
{
    return reinterpret_cast<uint8_t*>((reinterpret_cast<int64_t>(addr) + alignment - 1) / alignment * alignment);
}

constexpr int64_t ALIGNMENT_2M = 2 * 1024 * 1024;
constexpr int64_t ALIGNMENT_512K = 512 * 1024;
constexpr int64_t ALIGNMENT_1K = 1024;

struct alignas(64) Tensor {
    void* ptr{};
    ScalarType _dtype{};
    int64_t dim{};
    int64_t sizes[MAX_DIM];
    int64_t strides[MAX_DIM];

    template <typename scalar_t = void>
    scalar_t* data_ptr() const
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
        return reinterpret_cast<scalar_t*>(ptr);
    }

    ScalarType dtype() const { return _dtype; }

    int64_t element_size() const { return elementSize(dtype()); }

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
        if (dim == 0) {
            return true;
        }
        for (int i = 0; i < dim - 1; ++i) {
            if (strides[i] != strides[i + 1] * sizes[i + 1]) {
                return false;
            }
        }
        return strides[dim - 1] == 1;
    }

    template <typename scalar_t>
    void fill(scalar_t el)
    {
        auto data = data_ptr<scalar_t>();
        std::fill(data, data + numel(), el);
    }

    template <typename scalar_t>
    std::vector<scalar_t> to_vector() const
    {
        PARAMETER_CHECK(dim == 1, "invalid dim != 1");
        data_ptr<scalar_t>();  // check scalar type
        int64_t size = sizes[0];
        std::vector<scalar_t> res(size, scalar_t{});
        Tensor res_tensor = Tensor::from_blob(dtype(), {size}, res.data());
        copy(*this, res_tensor);
        return res;
    }

    static Tensor create(ScalarType dtype, const std::vector<int64_t>& sizes, std::unique_ptr<uint8_t[]>& hold);
    static Tensor alloc_from(ScalarType dtype, const std::vector<int64_t>& sizes, u8span& hold, int64_t alignment = 1);
    static Tensor from_blob(ScalarType dtype, const std::vector<int64_t>& sizes, void* blob);
    static Tensor transpose(const Tensor& old, int64_t dim1, int64_t dim2);
    static Tensor view(const Tensor& old, const std::vector<int64_t>& new_sizes);
    static Tensor select(const Tensor& old, int64_t dim, int64_t index);
    static Tensor slice(const Tensor& old, int64_t dim, int64_t start, int64_t end, int64_t step = 1);
    static Tensor reinterpret(const Tensor& old, ScalarType dtype);
    static void copy_impl(const Tensor& src, const Tensor& dst);
    static void copy(const Tensor& src, const Tensor& dst);
};

inline Tensor Tensor::create(ScalarType dtype, const std::vector<int64_t>& sizes, std::unique_ptr<uint8_t[]>& hold)
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

inline Tensor Tensor::alloc_from(ScalarType dtype, const std::vector<int64_t>& sizes, u8span& hold, int64_t alignment)
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
    auto align_gap = alignup(hold.ptr, alignment) - hold.ptr;
    size += align_gap;
    result.ptr = alignup(hold.ptr, alignment);
    PARAMETER_CHECK(size > 0, size);
    PARAMETER_CHECK(size <= (int64_t)hold.size, "need ", size, " remain ", hold.size, " end ",
        static_cast<void*>(hold.ptr + hold.size));
    hold = hold.subspan(size);
    return result;
}

inline Tensor Tensor::from_blob(ScalarType dtype, const std::vector<int64_t>& sizes, void* blob)
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

inline Tensor Tensor::transpose(const Tensor& old, int64_t dim1, int64_t dim2)
{
    PARAMETER_CHECK(dim1 >= 0 && dim1 < old.dim && dim2 >= 0 && dim2 < old.dim, " dim1 = ", dim1, " dim2 = ", dim2,
        " old.dim = ", old.dim);
    Tensor result = old;
    std::swap(result.sizes[dim1], result.sizes[dim2]);
    std::swap(result.strides[dim1], result.strides[dim2]);
    return result;
}

inline Tensor Tensor::view(const Tensor& old, const std::vector<int64_t>& new_sizes)
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
            PARAMETER_CHECK(view_numel == tensor_numel, "Tensor::view");
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

inline Tensor Tensor::select(const Tensor& old, int64_t dim, int64_t index)
{
    PARAMETER_CHECK(dim >= 0 && dim < old.dim, dim, " ", old.dim);
    PARAMETER_CHECK(index >= 0 && index < old.sizes[dim], index, " ", old.sizes[dim]);
    Tensor result{};
    result._dtype = old._dtype;
    result.dim = old.dim - 1;
    result.ptr = (uint8_t*)old.ptr + old.strides[dim] * index * elementSize(old._dtype);
#pragma unroll
    for (int64_t i = 0; i < MAX_DIM - 1; i++) {
        result.sizes[i] = old.sizes[i < dim ? i : i + 1];
        result.strides[i] = old.strides[i < dim ? i : i + 1];
    }
    return result;
}

inline Tensor Tensor::slice(const Tensor& old, int64_t dim, int64_t start, int64_t end, int64_t step)
{
    PARAMETER_CHECK(dim >= 0 && dim < old.dim, fmt::format("dim={}, old.dim={}", dim, old.dim));
    PARAMETER_CHECK(0 <= start && start <= end && end <= old.sizes[dim],
        fmt::format("start={}, end={}, old.sizes[dim]={}", start, end, old.sizes[dim]));
    Tensor result{};
    result._dtype = old._dtype;
    result.dim = old.dim;
    result.ptr = (uint8_t*)old.ptr + old.strides[dim] * start * elementSize(old._dtype);
#pragma unroll
    for (int64_t i = 0; i < MAX_DIM; i++) {
        result.sizes[i] = old.sizes[i];
        result.strides[i] = old.strides[i];
    }
    result.sizes[dim] = (end - start + step - 1) / step;
    result.strides[dim] *= step;
    return result;
}

inline Tensor Tensor::reinterpret(const Tensor& old, ScalarType dtype)
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

inline void Tensor::copy_impl(const Tensor& src, const Tensor& dst)
{
    if (src.is_contiguous() && dst.is_contiguous()) {
        std::memcpy(dst.ptr, src.ptr, src.numel() * src.element_size());
    } else {
        for (int64_t i = 0; i < src.sizes[0]; i++) {
            Tensor::copy(select(src, 0, i), select(dst, 0, i));
        }
    }
}

inline void Tensor::copy(const Tensor& src, const Tensor& dst)
{
    PARAMETER_CHECK(src.dtype() == dst.dtype(), "not the same type");
    PARAMETER_CHECK(src.dim == dst.dim, " src.dim = ", src.dim, " dst.dim = ", dst.dim);
    for (int64_t i = 0; i < src.dim; i++) {
        PARAMETER_CHECK(src.sizes[i] == dst.sizes[i], "not the same size");
    }
    copy_impl(src, dst);
}

struct TensorWithMemory {
    Tensor tensor;
    std::unique_ptr<uint8_t[]> memory;
    static TensorWithMemory create(ScalarType dtype, const std::vector<int64_t>& sizes)
    {
        TensorWithMemory result;
        *result = Tensor::create(dtype, sizes, result.memory);
        return result;
    }
    static TensorWithMemory from(Tensor tensor)
    {
        TensorWithMemory result;
        *result = Tensor::create(tensor.dtype(), std::vector<int64_t>(tensor.sizes, tensor.sizes + tensor.dim),
            result.memory);
        Tensor::copy(tensor, *result);
        return result;
    }
    Tensor& operator*() { return tensor; }
    const Tensor& operator*() const { return tensor; }
    Tensor* operator->() { return &tensor; }
    const Tensor* operator->() const { return &tensor; }

    static TensorWithMemory cat(std::vector<Tensor> tensors, int64_t dim)
    {
        PARAMETER_CHECK(tensors.size() > 0, " tensors.size() = ", tensors.size());
        auto t0 = tensors[0];
        PARAMETER_CHECK(dim < t0.dim, " dim = ", dim, " t0.dim = ", t0.dim);
        std::vector<int64_t> sizes(t0.sizes, t0.sizes + t0.dim);
        for (size_t i = 1; i < tensors.size(); i++) {
            PARAMETER_CHECK(tensors[i].dim == t0.dim, " i = ", i, " tensors[i].dim = ", tensors[i].dim,
                " t0.dim = ", t0.dim);
            PARAMETER_CHECK(tensors[i].dtype() == t0.dtype(), " i = ", i, " tensors[i].dtype() = ", tensors[i].dtype(),
                " t0.dtype() = ", t0.dtype());
            for (int64_t d = 0; d < t0.dim; d++) {
                if (d != dim) {
                    PARAMETER_CHECK(tensors[i].sizes[d] == t0.sizes[d], " i = ", i, " d = ", d,
                        " tensors[i].sizes[d] = ", tensors[i].sizes[d], " t0.sizes[d] = ", t0.sizes[d]);
                }
            }
            sizes[dim] += tensors[i].sizes[dim];
        }
        TensorWithMemory result = TensorWithMemory::create(t0.dtype(), sizes);
        int64_t offset = 0;
        for (size_t i = 0; i < tensors.size(); i++) {
            Tensor::copy(tensors[i], Tensor::slice(*result, dim, offset, offset + tensors[i].sizes[dim]));
            offset += tensors[i].sizes[dim];
        }
        return result;
    }
};
