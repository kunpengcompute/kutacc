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

#ifndef KUTACC_BLAS_H
#define KUTACC_BLAS_H

#include <cstdint>
#include <cstdlib>

#include "kutacc.h"

namespace kutacc {

    template <typename intype, typename outtype>
    void gemm(char transa, char transb, int m, int n, int k, outtype alpha, const intype *a, int lda, const intype *b,
        int ldb, outtype beta, outtype *c, int ldc, const BlasExtendParams &param = {});

    template <typename intype, typename outtype>
    void gemm_pack(char identifier, char transa, char transb, int M, int N, int K, int lda, int ldb, const intype *src,
        intype *dst);

    template <typename intype, typename outtype>                                                                                                                     \
    size_t gemm_pack_get_size(char identifier, int m, int n, int k);
}   // namespace kutacc
#endif

