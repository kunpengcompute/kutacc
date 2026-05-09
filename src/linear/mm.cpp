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

#include "mm.h"
#include "utils/blas.h"
#include "utils/check.h"
#include "utils/parallel.h"

#define DIM_2 2

namespace kutacc {
namespace internal {

inline void add_mm_check(const Tensor &a, const Tensor &b, const Tensor &c)
{
    KUTACC_CHECK(a.sizes()[1] == b.sizes()[0] && c.sizes()[0] == a.sizes()[0] && c.sizes()[1] == b.sizes()[1],
                 "shape invalid, a: [", a.sizes()[0], ", ", a.sizes()[1], "], b: [", b.sizes()[0], ", ", b.sizes()[1],
                 "], c: [", c.sizes()[0], ", ", c.sizes()[1], "]");
    KUTACC_CHECK(a.strides()[0] == 1 || a.strides()[1] == 1, "a strides invalid: [", a.strides()[0], ", ",
                 a.strides()[1], "]");
    KUTACC_CHECK(b.strides()[0] == 1 || b.strides()[1] == 1, "b strides invalid: [", b.strides()[0], ", ",
                 b.strides()[1], "]");
    KUTACC_CHECK(c.strides()[1] == 1, "c strides invalid: [", c.strides()[0], ", ", c.strides()[1], "]");
    KUTACC_CHECK(a.dim() == DIM_2 && b.dim() == DIM_2 && c.dim() == DIM_2, "dim invalid, a: [", a.dim(), "], b: [",
                 b.dim(), "], c: [", c.dim(), "]");
}
} // namespace internal

/**
 * @brief c = alpha * matmul(a, b) + beta * c
 * @param alpha
 * @param a shape [m, k], DType=scalr_t
 * @param a shape [k, n], DType=scalr_t
 * @param beta
 * @param c shape [m, n], DType=scalr_out_t
 */
template <typename scalar_t, typename scalar_out_t>
inline void addmm_template(scalar_out_t alpha, const Tensor &a, const Tensor &b, scalar_out_t beta, const Tensor &c,
                           kutacc::BlasExtendParams param = {})
{
    internal::add_mm_check(a, b, c);
    int m = static_cast<int>(a.sizes()[0]);
    int k = static_cast<int>(a.sizes()[1]);
    int n = static_cast<int>(b.sizes()[1]);
    char transa, transb;
    int64_t lda, ldb;
    if (a.strides()[1] == 1) {
        transa = 'N';
        lda = a.strides()[0];
    } else {
        transa = 'T';
        lda = a.strides()[1];
    }
    if (b.strides()[1] == 1) {
        transb = 'N';
        ldb = b.strides()[0];
    } else {
        transb = 'T';
        ldb = b.strides()[1];
    }

    std::swap(param.prepack_a, param.prepack_b);
    kutacc::gemm(transb, transa, n, m, k, alpha, static_cast<const scalar_t *>(b.data_ptr()), (int)ldb,
                 static_cast<const scalar_t *>(a.data_ptr()), (int)lda, beta, static_cast<scalar_t *>(c.data_ptr()),
                 (int)c.strides()[0], param);
}

void addmm(__bf16 alpha, const kutacc::Tensor &a, const kutacc::Tensor &b, __bf16 beta, const kutacc::Tensor &c,
           kutacc::BlasExtendParams param)
{
    KUTACC_CHECK(a.data_ptr() != nullptr && b.data_ptr() != nullptr && c.data_ptr() != nullptr, "addmm input invalid");
    KUTACC_CHECK(a.dtype() == kutacc::kBF16 && b.dtype() == kutacc::kBF16 && c.dtype() == kutacc::kBF16,
                 "addmm dtype invalid, a: [", a.dtype(), "], b: [", b.dtype(), "], c: [", c.dtype(), "]");
    if (!kutacc_check_err_set) {
        addmm_template<__bf16, __bf16>(alpha, a, b, beta, c, param);
    }
}
} // namespace kutacc

/**
* @brief gemm prepack for linear layer
* @brief weight shape [n, k], dtype = __bf16
* @brief shape [len], dtype = __bf16
*/
kutacc_export void kutacc_af2_linear_weight_prepack(const __bf16 *weight, __bf16 *result, int64_t n, int64_t k,
                                                    int64_t ldb, int64_t num_threads)
{
    if (num_threads != 1) {
        int64_t nblocks = (n + 32 - 1) / 32;
        kutacc::parallel_for(0, nblocks, 1, [&](int64_t start, int64_t end) {
            for (int64_t block_i = start; block_i < end; block_i++) {
                int64_t start_i = block_i * 32;
                int64_t end_i = std::min(start_i + 32, n);
                kutacc::gemm_pack<__bf16, __bf16>('A', 'T', 'N', (int)(end_i - start_i), 0, (int)k, (int)ldb, 0,
                                                  weight + start_i * ldb, result + start_i * k);
            }
        });
    } else {
        kutacc::gemm_pack<__bf16, __bf16>('A', 'T', 'N', (int)n, 0, (int)k, (int)ldb, 0, weight, result);
    }
}
