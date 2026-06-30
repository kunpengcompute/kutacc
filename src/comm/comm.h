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

#include <atomic>
#include <arm_sve.h>
#include <functional>
#include <kupl.h>

#include "kutacc.h"
#include "utils/check.h"

#define kutacc_aarch64_dmb(_op) asm volatile("dmb " #_op ::: "memory")
#define kutacc_memory_cpu_load_fence() kutacc_aarch64_dmb(ishld)
#define COMM_MAX_SIZE 16

namespace kutacc {

inline int64_t shm_datatype_size(shm_datatype_t datatype)
{
    if (datatype == SHM_DATATYPE_INT8) {
        return sizeof(int8_t);
    } else if (datatype == SHM_DATATYPE_UINT8) {
        return sizeof(uint8_t);
    } else if (datatype == SHM_DATATYPE_BFLOAT16) {
        return sizeof(bfloat16_t);
    } else {
        return 0;
    }
}

inline int align = 128;

inline void wait_fence(int16_t* fence_base_ptr, int16_t fence_id)
{
    int16_t* fence_ptr = fence_base_ptr + fence_id * align;
    int16_t flag = 0;
    do {
        kutacc_memory_cpu_load_fence();
        flag = *(fence_ptr);
    } while (flag != 1);
    *(fence_ptr) = 0;
}

inline void wait_batch_fence(int16_t* fence_base_ptr, int16_t fence_offset)
{
    int16_t* fence_ptr[4];
    for (int i = 0; i < 4; ++i) {
        fence_ptr[i] = fence_base_ptr + (fence_offset + i) * align;
    }
    while (1) {
        kutacc_memory_cpu_load_fence();
        if (*(fence_ptr[0]) == 1 &&
            *(fence_ptr[1]) == 1 &&
            *(fence_ptr[2]) == 1 &&
            *(fence_ptr[3]) == 1) {
            break;
        }
    }
    for (int i = 0; i < 4; ++i) {
        *(fence_ptr[i]) = 0;
    }
}

inline void signal_fence(int16_t* fence_base_ptr, int16_t fence_id)
{
    int16_t* fence_ptr = fence_base_ptr + fence_id * align;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *(fence_ptr) = 1;
}

template <int k1, int k2, bool ldnt = false, bool stnt = false, int pld_stride = -1, int pst_stride = -1>
inline void reduce(bfloat16_t **a, bfloat16_t **b, int n)
{
    static_assert(k1 > 0);
    static constexpr enum svprfop pld_op = SV_PLDL2KEEP;
    static constexpr enum svprfop pst_op = SV_PSTL2KEEP;

    svbfloat16_t sve_zero = svdup_bf16(0);
    svbool_t ptrue32 = svptrue_b32();

    if constexpr (pld_stride >= 0 || pst_stride >= 0) {
        svbool_t ptrue16 = svptrue_b16();
        for (int i = 0; i + 32 <= std::min(pld_stride, n); i += 32) {
            for (int k = 0; k < k1; ++k) {
                svprfh(ptrue16, a[k] + i, pld_op);
            }
        }
        for (int i = 0; i + 32 <= std::min(pst_stride, n); i += 32) {
            for (int k = 0; k < k2; ++k) {
                svprfh(ptrue16, b[k] + i, pst_op);
            }
        }
    }
    for (int i = 0; i < n; i += 32) {
        svbool_t pred = svwhilelt_b16(i, n);
        svbfloat16_t sve;
        if constexpr (pld_stride >= 0) {
            svprfh(svwhilelt_b16(i + pld_stride, n), a[0] + i + pld_stride, pld_op);
        }
        if constexpr (ldnt) {
            sve = svldnt1(pred, a[0] + i);
        } else {
            sve = svld1(pred, a[0] + i);
        }
        svfloat32_t sve0 = svreinterpret_f32(svzip1(sve_zero, sve));
        svfloat32_t sve1 = svreinterpret_f32(svzip2(sve_zero, sve));
        for (int k = 1; k < k1; ++k) {
            if constexpr (pld_stride >= 0) {
                svprfh(svwhilelt_b16(i + pld_stride, n), a[k] + i + pld_stride, pld_op);
            }
            if constexpr (ldnt) {
                sve = svldnt1(pred, a[k] + i);
            } else {
                sve = svld1(pred, a[k] + i);
            }
            sve0 = svadd_m(ptrue32, sve0, svreinterpret_f32(svzip1(sve_zero, sve)));
            sve1 = svadd_m(ptrue32, sve1, svreinterpret_f32(svzip2(sve_zero, sve)));
        }
        sve = svuzp1(svcvt_bf16_x(ptrue32, sve0), svcvt_bf16_x(ptrue32, sve1));
        for (int k = 0; k < k2; ++k) {
            if constexpr (pst_stride >= 0) {
                svprfh(svwhilelt_b16(i + pst_stride, n), b[k] + i + pst_stride, pst_op);
            }
            if constexpr (stnt) {
                svstnt1(pred, b[k] + i, sve);
            } else {
                svst1(pred, b[k] + i, sve);
            }
        }
    }
}

template <int k1, int k2, bool ldnt = false, bool stnt = false, int pld_stride = -1, int pst_stride = -1>
inline void reduce_weighted(bfloat16_t **a, bfloat16_t **b, float *weights, int n)
{
    static_assert(k1 > 0);
    static constexpr enum svprfop pld_op = SV_PLDL2KEEP;
    static constexpr enum svprfop pst_op = SV_PSTL2KEEP;

    svbfloat16_t sve_zero = svdup_bf16(0);
    svbool_t ptrue32 = svptrue_b16();

    if constexpr (pld_stride >= 0 || pst_stride >= 0) {
        for (int i = 0; i + 32 <= std::min(pld_stride, n); i += 32) {
            for (int k = 0; k < k1; ++k) {
                svprfh(ptrue32, a[k] + i, pld_op);
            }
        }
        for (int i = 0; i + 32 <= std::min(pst_stride, n); i += 32) {
            for (int k = 0; k < k2; ++k) {
                svprfh(ptrue32, b[k] + i, pst_op);
            }
        }
    }

    for (int i = 0; i < n; i += 32) {
        svbool_t pred = svwhilelt_b16(i, n);
        svfloat32_t sve0 = svdup_f32(0);
        svfloat32_t sve1 = svdup_f32(0);
        for (int k = 0; k < k1; ++k) {
            if constexpr (pld_stride >= 0) {
                svprfh(svwhilelt_b16(i + pld_stride, n), a[k] + i + pld_stride, pld_op);
            }
            auto sve = svldnt1(pred, a[k] + i);
            svfloat32_t zip1_val = svreinterpret_f32(svzip1(sve_zero, sve));
            svfloat32_t zip2_val = svreinterpret_f32(svzip2(sve_zero, sve));
            sve0 = svadd_m(pred, sve0, svmul_m(pred, zip1_val, weights[k]));
            sve1 = svadd_m(pred, sve1, svmul_m(pred, zip2_val, weights[k]));
        }
        svbfloat16_t sve_bf16 = svuzp1(svcvt_bf16_x(ptrue32, sve0), svcvt_bf16_x(ptrue32, sve1));
        for (int k = 0; k < k2; ++k) {
            if constexpr (pst_stride >= 0) {
                svprfh(svwhilelt_b16(i + pst_stride, n), b[k] + i + pst_stride, pst_op);
            }
            svstnt1(pred, b[k] + i, sve_bf16);
        }
    }
}

template <int k2, bool ldnt = false, bool stnt = false, int pld_stride = -1, int pst_stride = -1>
inline void reduce_weighted_runtime(bfloat16_t **a, bfloat16_t **b, float *weights, int n, int k1)
{
    switch (k1) {
        case 0: return;
        case 1: reduce_weighted<1, k2, ldnt, stnt, pld_stride, pst_stride>(a, b, weights, n); break;
        case 2: reduce_weighted<2, k2, ldnt, stnt, pld_stride, pst_stride>(a, b, weights, n); break;
        case 3: reduce_weighted<3, k2, ldnt, stnt, pld_stride, pst_stride>(a, b, weights, n); break;
        case 4: reduce_weighted<4, k2, ldnt, stnt, pld_stride, pst_stride>(a, b, weights, n); break;
        case 5: reduce_weighted<5, k2, ldnt, stnt, pld_stride, pst_stride>(a, b, weights, n); break;
        case 6: reduce_weighted<6, k2, ldnt, stnt, pld_stride, pst_stride>(a, b, weights, n); break;
        case 7: reduce_weighted<7, k2, ldnt, stnt, pld_stride, pst_stride>(a, b, weights, n); break;
        case 8: reduce_weighted<8, k2, ldnt, stnt, pld_stride, pst_stride>(a, b, weights, n); break;
        default: KUTACC_CHECK(false, "no implement for K1 > 8"); break;
    }
}

template <int k, bool ldnt = false, bool stnt = false, int pld_stride = -1, int pst_stride = -1>
inline void copy(bfloat16_t **a, const bfloat16_t *b, int n)
{
    static constexpr enum svprfop pld_op = SV_PLDL2KEEP;
    static constexpr enum svprfop pst_op = SV_PSTL2KEEP;

    if constexpr (pld_stride >= 0 || pst_stride >= 0) {
        svbool_t ptrue16 = svptrue_b16();
        for (int i = 0; i + 32 <= std::min(pld_stride, n); i += 32) {
            svprfh(ptrue16, b + i, pld_op);
        }
        for (int i = 0; i + 32 <= std::min(pst_stride, n); i += 32) {
            for (int j = 0; j < k; ++j) {
                svprfh(ptrue16, a[j] + i, pst_op);
            }
        }
    }
    for (int i = 0; i < n; i += 32) {
        svbool_t pred = svwhilelt_b16(i, n);
        if constexpr (pld_stride >= 0) {
            svprfh(svwhilelt_b16(i + pld_stride, n), b + i + pld_stride, pld_op);
        }
        svbfloat16_t sve;
        if constexpr (ldnt) {
            sve = svldnt1(pred, b + i);
        } else {
            sve = svld1(pred, b + i);
        }
        for (int j = 0; j < k; ++j) {
            if constexpr (pst_stride >= 0) {
                svprfh(svwhilelt_b16(i + pst_stride, n), a[j] + i + pst_stride, pst_op);
            }
            if constexpr (stnt) {
                svstnt1(pred, a[j] + i, sve);
            } else {
                svst1(pred, a[j] + i, sve);
            }
        }
    }
}

}