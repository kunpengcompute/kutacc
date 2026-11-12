#pragma once

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>

namespace kutacc {
template <typename T>
struct AlignedMallocDeleter {
    void operator()(T* ptr) const noexcept
    {
        if (ptr) std::free(reinterpret_cast<void**>(ptr)[-1]);
    }
};

template <typename T>
inline std::unique_ptr<T[], AlignedMallocDeleter<T>> malloc_buffer(int64_t count)
{
    constexpr size_t alignment = 64;
    constexpr size_t overhead = alignment - 1 + sizeof(void*);
    if (count < 0 || static_cast<uint64_t>(count) >
            (std::numeric_limits<size_t>::max() - overhead) / sizeof(T)) {
        throw std::bad_array_new_length();
    }
    const size_t bytes = count ? static_cast<size_t>(count) * sizeof(T) : sizeof(T);
    void* allocation = std::malloc(bytes + overhead);
    if (!allocation) throw std::bad_alloc();
    const uintptr_t address = (reinterpret_cast<uintptr_t>(allocation) +
        sizeof(void*) + alignment - 1) & ~uintptr_t(alignment - 1);
    T* ptr = reinterpret_cast<T*>(address);
    reinterpret_cast<void**>(ptr)[-1] = allocation;
    return std::unique_ptr<T[], AlignedMallocDeleter<T>>(ptr);
}
} // namespace kutacc
