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

#include "blas.h"
#include <cstdlib>

#ifndef BLAS_EXTEND_PARM_DEF
#define BLAS_EXTEND_PARM_DEF

typedef enum {
    BLAS_EXTEND_TYPE_NUM_THREADS,
    BLAS_EXTEND_TYPE_PREPACK,
    BLAS_EXTEND_TYPE_BIAS,
    BLAS_EXTEND_TYPE_ACTIVATION,
} BlasExtendType;
#define BLAS_EXTEND_PREPACK_A_MASK (0x1)
#define BLAS_EXTEND_PREPACK_B_MASK (0x2)
#define BLAS_EXTEND_BIAS_ROW_MODE (0x1)
#define BLAS_EXTEND_BIAS_COL_MODE (0x2)
#define BLAS_EXTEND_ACTIVATION_RELU (0x1)

typedef struct BlasExtendParam_ {
    BlasExtendType type;
    uint64_t extra;
    struct BlasExtendParam_ *next;
} BlasExtendParam;

typedef struct BlasExtendBiasExtra_ {
    uint32_t biasMode;
    const void *bias;
} BlasExtendBiasExtra;
#endif /* BLAS_EXTEND_PARAM_DEFINE */

#define BLAS_EXTEND_PARAM_LIST_DEFINE(name) BlasExtendParam *name = nullptr;

#define BLAS_EXTEND_PARAM_LIST_APPEND(param, type_, extra_)   \
    BlasExtendParam __BLAS_EXTEND_PARAM_##param##_##type_ = { \
        .type = (type_),                                      \
        .extra = (uint64_t)(extra_),                          \
        .next = param,                                        \
    };                                                        \
    param = &__BLAS_EXTEND_PARAM_##param##_##type_;

#define DEF_GEMM(prefix, intype, outtype)                                                                             \
    extern "C" void prefix##gemm_ex(const char transa, char transb, int m, int n, int k, outtype alpha,               \
                                    const intype *a, int lda, const intype *b, int ldb, outtype beta, outtype *c,     \
                                    int ldc, const BlasExtendParam *kutacc_param);                                    \
    extern "C" size_t prefix##gemm_pack_get_size(char identifier, int m, int n, int k);                               \
    extern "C" void prefix##gemm_pack(char identifier, char transa, char transb, int M, int N, int K, int lda,        \
                                      int ldb, const intype *src, intype *dst);                                       \
    template <>                                                                                                       \
    void kutacc::gemm<intype, outtype>(char transa, char transb, int m, int n, int k, outtype alpha, const intype *a, \
                                       int lda, const intype *b, int ldb, outtype beta, outtype *c, int ldc,          \
                                       const kutacc::BlasExtendParams &kutacc_param)                                  \
    {                                                                                                                 \
        BLAS_EXTEND_PARAM_LIST_DEFINE(param);                                                                         \
        BLAS_EXTEND_PARAM_LIST_APPEND(param, BLAS_EXTEND_TYPE_NUM_THREADS, kutacc_param.num_threads);                 \
        uint32_t prepack_mask = uint32_t((kutacc_param.prepack_a ? BLAS_EXTEND_PREPACK_A_MASK : 0) |                  \
                                         (kutacc_param.prepack_b ? BLAS_EXTEND_PREPACK_B_MASK : 0));                  \
        BLAS_EXTEND_PARAM_LIST_APPEND(param, BLAS_EXTEND_TYPE_PREPACK, prepack_mask);                                 \
        BlasExtendBiasExtra bias_extra = {.biasMode = uint32_t(kutacc_param.row_bias ? BLAS_EXTEND_BIAS_ROW_MODE :    \
                                                               kutacc_param.col_bias ? BLAS_EXTEND_BIAS_COL_MODE :    \
                                                                                       0),                            \
                                          .bias = kutacc_param.bias};                                                 \
        BLAS_EXTEND_PARAM_LIST_APPEND(param, BLAS_EXTEND_TYPE_BIAS, &bias_extra);                                     \
        uint32_t activation = uint32_t(kutacc_param.relu ? BLAS_EXTEND_ACTIVATION_RELU : 0);                          \
        BLAS_EXTEND_PARAM_LIST_APPEND(param, BLAS_EXTEND_TYPE_ACTIVATION, activation);                                \
        prefix##gemm_ex(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, param);                         \
    }                                                                                                                 \
    template <>                                                                                                       \
    size_t kutacc::gemm_pack_get_size<intype, outtype>(char identifier, int m, int n, int k)                          \
    {                                                                                                                 \
        return prefix##gemm_pack_get_size(identifier, m, n, k);                                                       \
    }                                                                                                                 \
    template <>                                                                                                       \
    void kutacc::gemm_pack<intype, outtype>(char identifier, char transa, char transb, int M, int N, int K, int lda,  \
                                            int ldb, const intype *src, intype *dst)                                  \
    {                                                                                                                 \
        return prefix##gemm_pack(identifier, transa, transb, M, N, K, lda, ldb, src, dst);                            \
    }

DEF_GEMM(kutacc_core_b, __bf16, __bf16);

kutacc_export size_t kutacc_af2_gemm_pack_get_size(char identifier, char transa, char transb, int m, int n, int k)
{
    (void)transa;
    (void)transb;
    return kutacc::gemm_pack_get_size<__bf16, __bf16>(identifier, m, n, k);
}
