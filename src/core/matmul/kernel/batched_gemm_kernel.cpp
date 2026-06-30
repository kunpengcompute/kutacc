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
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>

#include "kutacc.h"
#include "kernel.h"

namespace kutacc {

constexpr int64_t DEFAULT_BLOCK_M = 512;
constexpr int64_t DEFAULT_BLOCK_N = 1024;
constexpr int64_t DEFAULT_BLOCK_K = 4096;
constexpr int64_t MAX_CPU_NUMBER = 1000;

template <bool is_row_quant>
__arm_new("za") static void batch_bf16_s8_packed_gemm_bf16_enable_matrix(int64_t m, int64_t n, int64_t k,
    const __bf16* a, [[maybe_unused]] int64_t lda, const int8_t* b, [[maybe_unused]] int64_t ldb, __bf16* c,
    int64_t ldc, const float* scale) __arm_streaming
{
    svbool_t pg8 = svptrue_b8();
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();

    assert(n % 32 == 0 && k % 2 == 0);
    int64_t mi = 0;

    for (; mi + 32 <= m; mi += 32) {
        int64_t prf_b_offset = 8 * 128;
        for (int64_t ni = 0; ni < n; ni += 32) {
            svfloat32_t rscale00;
            svfloat32_t rscale01;
            svfloat32_t rscale10;
            svfloat32_t rscale11;
            if constexpr (is_row_quant) {
                svfloat32_t rscale0 = svld1(pg32, scale + ni);
                svfloat32_t rscale1 = svld1(pg32, scale + ni + 16);
                rscale00 = svzip1(rscale0, rscale0);
                rscale01 = svzip2(rscale0, rscale0);
                rscale10 = svzip1(rscale1, rscale1);
                rscale11 = svzip2(rscale1, rscale1);
            }
            svzero_za();
            for (int64_t ki = 0; ki < k; ki += 2) {
                svbfloat16_t a0 = svld1(pg16, a + mi * k + 16 * ki);
                svbfloat16_t a1 = svld1(pg16, a + (mi + 16) * k + 16 * ki);
                svint8_t b_values = svldnt1(pg8, b + ni * k + 32 * ki);
                svprfb(svwhilelt_b8(prf_b_offset, n * k), b + prf_b_offset, SV_PLDL2KEEP);
                prf_b_offset += 64;
                svint8_t zero_s8 = svdup_s8(0);
                svint16_t zero_s16 = svreinterpret_s16(zero_s8);
                auto b0_s16 = svreinterpret_s16(svzip1(b_values, zero_s8));
                auto b1_s16 = svreinterpret_s16(svzip2(b_values, zero_s8));
                svfloat32_t b00_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip1(b0_s16, zero_s16))));
                svfloat32_t b01_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip2(b0_s16, zero_s16))));
                svfloat32_t b10_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip1(b1_s16, zero_s16))));
                svfloat32_t b11_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip2(b1_s16, zero_s16))));
                if constexpr (is_row_quant) {
                    b00_f32 = svmul_x(pg32, b00_f32, rscale00);
                    b01_f32 = svmul_x(pg32, b01_f32, rscale01);
                    b10_f32 = svmul_x(pg32, b10_f32, rscale10);
                    b11_f32 = svmul_x(pg32, b11_f32, rscale11);
                }
                if constexpr (!is_row_quant) {
                    svfloat32_t cscale = svzip1(svdup_f32(scale[ki]), svdup_f32(scale[ki + 1]));
                    b00_f32 = svmul_x(pg32, b00_f32, cscale);
                    b01_f32 = svmul_x(pg32, b01_f32, cscale);
                    b10_f32 = svmul_x(pg32, b10_f32, cscale);
                    b11_f32 = svmul_x(pg32, b11_f32, cscale);
                }
                svbfloat16_t b00_bf16 = svcvt_bf16_x(pg32, b00_f32);
                svbfloat16_t b01_bf16 = svcvt_bf16_x(pg32, b01_f32);
                svbfloat16_t b10_bf16 = svcvt_bf16_x(pg32, b10_f32);
                svbfloat16_t b11_bf16 = svcvt_bf16_x(pg32, b11_f32);
                svbfloat16_t b0 = svuzp1(b00_bf16, b01_bf16);
                svbfloat16_t b1 = svuzp1(b10_bf16, b11_bf16);

                svmopa_za32_bf16_m(0, pg16, pg16, a0, b0);
                svmopa_za32_bf16_m(1, pg16, pg16, a0, b1);
                svmopa_za32_bf16_m(2, pg16, pg16, a1, b0);
                svmopa_za32_bf16_m(3, pg16, pg16, a1, b1);
            }
            for (int64_t i = 0; i < 16; i++) {
                svfloat32_t o00 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 0, i);
                svfloat32_t o01 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 1, i);
                svfloat32_t o10 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 2, i);
                svfloat32_t o11 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 3, i);
                svbfloat16_t o0 = svuzp1(svcvt_bf16_x(pg32, o00), svcvt_bf16_x(pg32, o01));
                svbfloat16_t o1 = svuzp1(svcvt_bf16_x(pg32, o10), svcvt_bf16_x(pg32, o11));
                svstnt1(pg16, c + (mi + i) * ldc + ni, o0);
                svstnt1(pg16, c + (mi + i + 16) * ldc + ni, o1);
            }
        }
    }

    if (mi + 16 < m) {
        int64_t prf_b_offset = 8 * 128;
        int64_t block_a1 = std::min(m - mi - 16, int64_t{16});
        svbool_t pg16_a1 = svwhilelt_b16(int64_t{0}, block_a1 * 2);
        for (int64_t ni = 0; ni < n; ni += 32) {
            svfloat32_t rscale00;
            svfloat32_t rscale01;
            svfloat32_t rscale10;
            svfloat32_t rscale11;
            if constexpr (is_row_quant) {
                svfloat32_t rscale0 = svld1(pg32, scale + ni);
                svfloat32_t rscale1 = svld1(pg32, scale + ni + 16);
                rscale00 = svzip1(rscale0, rscale0);
                rscale01 = svzip2(rscale0, rscale0);
                rscale10 = svzip1(rscale1, rscale1);
                rscale11 = svzip2(rscale1, rscale1);
            }
            svzero_za();
            for (int64_t ki = 0; ki < k; ki += 2) {
                svbfloat16_t a0 = svld1(pg16, a + mi * k + 16 * ki);
                svbfloat16_t a1 = svld1(pg16_a1, a + (mi + 16) * k + block_a1 * ki);
                svint8_t b_values = svldnt1(pg8, b + ni * k + 32 * ki);
                svprfb(svwhilelt_b8(prf_b_offset, n * k), b + prf_b_offset, SV_PLDL2KEEP);
                prf_b_offset += 64;
                svint8_t zero_s8 = svdup_s8(0);
                svint16_t zero_s16 = svreinterpret_s16(zero_s8);
                auto b0_s16 = svreinterpret_s16(svzip1(b_values, zero_s8));
                auto b1_s16 = svreinterpret_s16(svzip2(b_values, zero_s8));
                svfloat32_t b00_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip1(b0_s16, zero_s16))));
                svfloat32_t b01_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip2(b0_s16, zero_s16))));
                svfloat32_t b10_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip1(b1_s16, zero_s16))));
                svfloat32_t b11_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip2(b1_s16, zero_s16))));
                if constexpr (is_row_quant) {
                    b00_f32 = svmul_x(pg32, b00_f32, rscale00);
                    b01_f32 = svmul_x(pg32, b01_f32, rscale01);
                    b10_f32 = svmul_x(pg32, b10_f32, rscale10);
                    b11_f32 = svmul_x(pg32, b11_f32, rscale11);
                }
                if constexpr (!is_row_quant) {
                    svfloat32_t cscale = svzip1(svdup_f32(scale[ki]), svdup_f32(scale[ki + 1]));
                    b00_f32 = svmul_x(pg32, b00_f32, cscale);
                    b01_f32 = svmul_x(pg32, b01_f32, cscale);
                    b10_f32 = svmul_x(pg32, b10_f32, cscale);
                    b11_f32 = svmul_x(pg32, b11_f32, cscale);
                }
                svbfloat16_t b00_bf16 = svcvt_bf16_x(pg32, b00_f32);
                svbfloat16_t b01_bf16 = svcvt_bf16_x(pg32, b01_f32);
                svbfloat16_t b10_bf16 = svcvt_bf16_x(pg32, b10_f32);
                svbfloat16_t b11_bf16 = svcvt_bf16_x(pg32, b11_f32);
                svbfloat16_t b0 = svuzp1(b00_bf16, b01_bf16);
                svbfloat16_t b1 = svuzp1(b10_bf16, b11_bf16);

                svmopa_za32_bf16_m(0, pg16, pg16, a0, b0);
                svmopa_za32_bf16_m(1, pg16, pg16, a0, b1);
                svmopa_za32_bf16_m(2, pg16, pg16, a1, b0);
                svmopa_za32_bf16_m(3, pg16, pg16, a1, b1);
            }
            for (int64_t i = 0; i < 16; i++) {
                svfloat32_t o00 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 0, i);
                svfloat32_t o01 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 1, i);
                svbfloat16_t o0 = svuzp1(svcvt_bf16_x(pg32, o00), svcvt_bf16_x(pg32, o01));
                svstnt1(pg16, c + (mi + i) * ldc + ni, o0);
            }
            for (int64_t i = 0; i < block_a1; i++) {
                svfloat32_t o10 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 2, i);
                svfloat32_t o11 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 3, i);
                svbfloat16_t o1 = svuzp1(svcvt_bf16_x(pg32, o10), svcvt_bf16_x(pg32, o11));
                svstnt1(pg16, c + (mi + i + 16) * ldc + ni, o1);
            }
        }
    } else if (mi < m) {
        int64_t prf_b_offset = 8 * 128;
        int64_t block_a0 = std::min(m - mi, int64_t{16});
        svbool_t pg16_a0 = svwhilelt_b16(int64_t{0}, block_a0 * 2);
        for (int64_t ni = 0; ni < n; ni += 32) {
            svfloat32_t rscale00;
            svfloat32_t rscale01;
            svfloat32_t rscale10;
            svfloat32_t rscale11;
            if constexpr (is_row_quant) {
                svfloat32_t rscale0 = svld1(pg32, scale + ni);
                svfloat32_t rscale1 = svld1(pg32, scale + ni + 16);
                rscale00 = svzip1(rscale0, rscale0);
                rscale01 = svzip2(rscale0, rscale0);
                rscale10 = svzip1(rscale1, rscale1);
                rscale11 = svzip2(rscale1, rscale1);
            }
            svzero_za();
            for (int64_t ki = 0; ki < k; ki += 2) {
                svbfloat16_t a0 = svld1(pg16_a0, a + mi * k + block_a0 * ki);
                svint8_t b_values = svldnt1(pg8, b + ni * k + 32 * ki);
                svprfb(svwhilelt_b8(prf_b_offset, n * k), b + prf_b_offset, SV_PLDL2KEEP);
                prf_b_offset += 64;
                svint8_t zero_s8 = svdup_s8(0);
                svint16_t zero_s16 = svreinterpret_s16(zero_s8);
                auto b0_s16 = svreinterpret_s16(svzip1(b_values, zero_s8));
                auto b1_s16 = svreinterpret_s16(svzip2(b_values, zero_s8));
                svfloat32_t b00_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip1(b0_s16, zero_s16))));
                svfloat32_t b01_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip2(b0_s16, zero_s16))));
                svfloat32_t b10_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip1(b1_s16, zero_s16))));
                svfloat32_t b11_f32 = svcvt_f32_x(pg32, svextb_x(pg32, svreinterpret_s32(svzip2(b1_s16, zero_s16))));
                if constexpr (is_row_quant) {
                    b00_f32 = svmul_x(pg32, b00_f32, rscale00);
                    b01_f32 = svmul_x(pg32, b01_f32, rscale01);
                    b10_f32 = svmul_x(pg32, b10_f32, rscale10);
                    b11_f32 = svmul_x(pg32, b11_f32, rscale11);
                }
                if constexpr (!is_row_quant) {
                    svfloat32_t cscale = svzip1(svdup_f32(scale[ki]), svdup_f32(scale[ki + 1]));
                    b00_f32 = svmul_x(pg32, b00_f32, cscale);
                    b01_f32 = svmul_x(pg32, b01_f32, cscale);
                    b10_f32 = svmul_x(pg32, b10_f32, cscale);
                    b11_f32 = svmul_x(pg32, b11_f32, cscale);
                }
                svbfloat16_t b00_bf16 = svcvt_bf16_x(pg32, b00_f32);
                svbfloat16_t b01_bf16 = svcvt_bf16_x(pg32, b01_f32);
                svbfloat16_t b10_bf16 = svcvt_bf16_x(pg32, b10_f32);
                svbfloat16_t b11_bf16 = svcvt_bf16_x(pg32, b11_f32);
                svbfloat16_t b0 = svuzp1(b00_bf16, b01_bf16);
                svbfloat16_t b1 = svuzp1(b10_bf16, b11_bf16);

                svmopa_za32_bf16_m(0, pg16, pg16, a0, b0);
                svmopa_za32_bf16_m(1, pg16, pg16, a0, b1);
            }
            for (int64_t i = 0; i < block_a0; i++) {
                svfloat32_t o00 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 0, i);
                svfloat32_t o01 = svread_hor_za32_f32_m(svfloat32_t(), pg32, 1, i);
                svbfloat16_t o0 = svuzp1(svcvt_bf16_x(pg32, o00), svcvt_bf16_x(pg32, o01));
                svstnt1(pg16, c + (mi + i) * ldc + ni, o0);
            }
        }
    }
}

int64_t GetPrepackOffset(int64_t m, int64_t n, int64_t mOff, int64_t nOff, int64_t blockN)
{
    return mOff * std::min(n - nOff, blockN) + nOff * m;
}

void batch_bf16_s8_packed_gemm_bf16_alpha1_beta0(int64_t m, int64_t n, int64_t k, const __bf16* a, int64_t lda,
    const int8_t* b, int64_t ldb, __bf16* c, int64_t ldc, const float* row_quant, const float* col_quant)
{
    if (row_quant) {
        batch_bf16_s8_packed_gemm_bf16_enable_matrix<true>(m, n, k, a, lda, b, ldb, c, ldc, row_quant);
    } else {
        batch_bf16_s8_packed_gemm_bf16_enable_matrix<false>(m, n, k, a, lda, b, ldb, c, ldc, col_quant);
    }
}

enum class Matrix : uint8_t { A, B, C };

template <int64_t blockM = 16, int64_t blockN = 4>
inline void GemmNCopyStub(int64_t m, int64_t n, const void *src_, int64_t ld, void *dst_)
{
    auto src = static_cast<const uint8_t *>(src_);
    auto dst = static_cast<uint8_t *>(dst_);
    int64_t stepM = 0;
    for (int64_t blockMOff = 0; blockMOff < m; blockMOff += stepM) {
        stepM = std::min(m - blockMOff, blockM);
        for (int64_t blockNOff = 0; blockNOff < n; blockNOff += blockN) {
            int64_t stepN = std::min(n - blockNOff, blockN);
            for (int64_t mOff = 0; mOff < stepM; mOff++) {
                for (int64_t nOff = 0; nOff < stepN; nOff++) {
                    *dst = src[(blockMOff + mOff) * ld + blockNOff + nOff];
                    dst++;
                }
            }
        }
    }
}

template <Matrix Mat>
void GemmPackOp(int64_t m, int64_t n, const int64_t *rangeM, const int64_t *rangeN, const void *src, int64_t ld,
    void *dst, int64_t splitK)
{
    int64_t blockN = splitK;
    if (blockN == 0) {
        blockN = DEFAULT_BLOCK_K / std::max(sizeof(int8_t), sizeof(bfloat16_t));
    }
    int64_t mStart = 0;
    int64_t mEnd = m;
    int64_t nStart = 0;
    int64_t nEnd = n;
    if (rangeM != nullptr && rangeN != nullptr) {
        mStart = rangeM[0];
        mEnd = rangeM[1];
        nStart = rangeN[0];
        nEnd = rangeN[1];
    }
    for (int64_t blockNOff = nStart; blockNOff < nEnd; blockNOff += blockN) {
        int64_t stepN = std::min(n - blockNOff, blockN);
        if constexpr (Mat == Matrix::A) {
            const void *srcPtr = reinterpret_cast<const int8_t *>(src) + mStart * ld + blockNOff;
            void *dstPtr = reinterpret_cast<int8_t *>(dst) + GetPrepackOffset(m, n, mStart, blockNOff, blockN);
            GemmNCopyStub<32, 2>(mEnd - mStart, stepN, srcPtr, ld, dstPtr);
        } else if constexpr (Mat == Matrix::B) {
            const void *srcPtr = reinterpret_cast<const bfloat16_t *>(src) + mStart * ld + blockNOff;
            void *dstPtr = reinterpret_cast<bfloat16_t *>(dst) + GetPrepackOffset(m, n, mStart, blockNOff, blockN);
            gemm_ncopy_matrix(mEnd - mStart, stepN * 2, srcPtr, ld * 2, dstPtr);
        }
    }
}

template <Matrix Mat>
void GemmPackThread(int64_t m, int64_t n, const void *src, int64_t ld, void *dst, const gemm_ex_t &extra)
{
    int64_t rangeM[MAX_CPU_NUMBER + 1];
    int64_t rangeN[2] = {0, n};
    int64_t unrollSz = 16;

    int64_t rangeLenM;
    {
        int64_t nThreads = extra.nThreads;
        int64_t part_m = (m + (nThreads * unrollSz) - 1) / (nThreads * unrollSz) * unrollSz;
        rangeLenM = (m + part_m - 1) / part_m;
        for (int64_t m_thread_id = 0; m_thread_id < rangeLenM; m_thread_id++) {
            rangeM[m_thread_id] = std::min(m_thread_id * part_m, m);
        }
        rangeM[rangeLenM] = m;
    }
    int64_t nThreads = rangeLenM;
    kutacc::parallel_for(0, nThreads, 1, [&](int64_t start, int64_t end) {
        for (int64_t threadId = start; threadId < end; threadId++) {
            GemmPackOp<Mat>(m, n, rangeM + threadId, rangeN, src, ld, dst, extra.splitK);
        }
    });
}

extern "C" void batch_bf16_gemm_pack_kernel(const char *matrix_, const char *trans, const int m, const int n,
    const void *src, const int ld, void *dst, const gemm_ex_t *_extra)
{
    auto matrix = std::toupper(*matrix_) == 'A' ? Matrix::A : std::toupper(*matrix_) == 'B' ? Matrix::B : Matrix::C;
    bool is_trans = (std::toupper(*trans) == 'T');
    assert((matrix == Matrix::A && is_trans) || (matrix == Matrix::B && !is_trans));
    gemm_ex_t extra = *_extra;
    if (extra.nThreads <= 0) {
        extra.nThreads = kutacc::get_thread_num();
    }
    if (extra.nThreads == 1) {
        if (matrix == Matrix::A)
            GemmPackOp<Matrix::A>(m, n, nullptr, nullptr, src, ld, dst, extra.splitK);
        else if (matrix == Matrix::B)
            GemmPackOp<Matrix::B>(m, n, nullptr, nullptr, src, ld, dst, extra.splitK);
    } else {
        if (matrix == Matrix::A)
            GemmPackThread<Matrix::A>(m, n, src, ld, dst, extra);
        else if (matrix == Matrix::B)
            GemmPackThread<Matrix::B>(m, n, src, ld, dst, extra);
    }
}

} // namespace kutacc
