/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KuTACC is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef KUTACC_COLLAPSE_H
#define KUTACC_COLLAPSE_H

#include "wrapper/wrapper.h"
#include <arm_neon.h>

namespace kutacc {

static inline __bf16 to_bf16(float x)
{
    return vcvth_bf16_f32(x);
}

static inline float to_float(__bf16 x)
{
    return vcvtah_f32_bf16(x);
}

template <typename T>
inline T data_index_init(T offset)
{
    return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T &x, const T &X, Args &&...args)
{
    offset = data_index_init(offset, std::forward<Args>(args)...);
    x = offset % X;
    return offset / X;
}

inline bool data_index_step()
{
    return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T &x, const T &X, Args &&...args)
{
    if (data_index_step(std::forward<Args>(args)...)) {
        x = ((x + 1) == X) ? 0 : (x + 1);
        return x == 0;
    }
    return false;
}

template <typename F>
inline void collapse_for(int64_t start, int64_t end, const F &f)
{
    for (int64_t i = start; i < end; i++) {
        f(i);
    }
}

template <typename F>
inline void collapse_for(int64_t start, int64_t end, int64_t n0, const F &f)
{
    (void)n0;
    for (int64_t i = start; i < end; i++) {
        f(i);
    }
}

template <typename F>
inline void collapse_for(int64_t start, int64_t end, int64_t n0, int64_t n1, const F &f)
{
    int64_t i0, i1;
    data_index_init(start, i0, n0, i1, n1);
    for ([[maybe_unused]] int64_t i = start; i < end; i++) {
        f(i0, i1);
        data_index_step(i0, n0, i1, n1);
    }
}

template <typename F>
inline void collapse_for(int64_t start, int64_t end, int64_t n0, int64_t n1, int64_t n2, const F &f)
{
    int64_t i0, i1, i2;
    data_index_init(start, i0, n0, i1, n1, i2, n2);
    for ([[maybe_unused]] int64_t i = start; i < end; i++) {
        f(i0, i1, i2);
        data_index_step(i0, n0, i1, n1, i2, n2);
    }
}

template <typename F>
inline void collapse_for(int64_t start, int64_t end, int64_t n0, int64_t n1, int64_t n2, int64_t n3, const F &f)
{
    int64_t i0, i1, i2, i3;
    data_index_init(start, i0, n0, i1, n1, i2, n2, i3, n3);
    for ([[maybe_unused]] int64_t i = start; i < end; i++) {
        f(i0, i1, i2, i3);
        data_index_step(i0, n0, i1, n1, i2, n2, i3, n3);
    }
}

} // namespace kutacc

#endif
