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
#include "kutacc.h"

namespace kutacc {

struct gemm_ex_t {
    int nThreads;
    const void *quantArgs;
    bool prepackA;
    bool prepackB;
    int batchCount;
    int splitK;
};

__arm_new("za") void s8_gemm_pack_kernel(int m, int n, int8_t *src, int ldc, int8_t *dst, int ldd) __arm_streaming;

__arm_new("za") void s8_gemm_pack_with_idx_kernel(int m, int n, int8_t *src, int ldc, int m_off, int *idx, int8_t *dst,
    int ldd) __arm_streaming;

__arm_new("za") void s8_packed_gemm_bf16_dq_alpha1_beta0(int m, int n, int ldc, int k, int8_t *a, int8_t *b,
    bfloat16_t *c, const float *rscales, const float *cscales, int macro_kernel_m = 256, int macro_kernel_n = 512,
    int macro_kernel_k = 1024) __arm_streaming;

__arm_new("za") void s8_packed_gemm_bf16_dq_alpha1_beta0_m_lt_128_k_2048(int m, int n, int ldc, int k, int8_t *a,
    int8_t *b, bfloat16_t *c, const float *rscales, const float *cscales, int macro_kernel_m = 128,
    int macro_kernel_n = 224, int macro_kernel_k = 2048) __arm_streaming;
    
__arm_new("za") void bf16_gemm_pack_kernel(int m, int n, bfloat16_t *src, int ldc, bfloat16_t *dst,
    int ldd) __arm_streaming;

__arm_new("za") void bf16_packed_gemm_alpha1_beta0(int m, int n, int ldc, int k, bfloat16_t *a, bfloat16_t *b,
    bfloat16_t *c, float *bias = nullptr, bool row_bias = true, int macro_kernel_m = 256, int macro_kernel_n = 512, 
    int macro_kernel_k = 1024) __arm_streaming;

extern "C" void batch_bf16_gemm_pack_kernel(const char *matrix_, const char *trans, const int m, const int n,
    const void *src, const int ld, void *dst, const gemm_ex_t *_extra);

void batch_bf16_s8_packed_gemm_bf16_alpha1_beta0(int64_t m, int64_t n, int64_t k, const __bf16* a, int64_t lda,
    const int8_t* b, int64_t ldb, __bf16* c, int64_t ldc, const float* row_quant, const float* col_quant);

extern "C" void gemm_ncopy_matrix(int64_t, int64_t, const void *, int64_t, void *);

} // namespace kutacc
