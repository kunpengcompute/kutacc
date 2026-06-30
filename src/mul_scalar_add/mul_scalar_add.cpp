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
#include "mul_scalar_add.h"
#include <arm_sve.h>
#include <arm_sve.h>
#include "kutacc.h"
#include "mul_scalar_add.h"
#include "utils/bf16.h"

namespace kutacc {
template <bool load_output>
void mul_scalar_add(bfloat16_t* i_ptr, bfloat16_t* o_ptr, int64_t num, float alpha)
{
    const int64_t STEP = svcnth();
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    svbfloat16_t zero_b = svdup_bf16(0);

    kutacc::parallel_for(0, (num + STEP - 1) / STEP, 1, [&](int64_t start, int64_t end) {
        start *= STEP;
        end = std::min(end * STEP, num);
        for (int64_t j = start; j < end; j += STEP) {
#ifdef ENABLE_EXTRA_PREFETCH
            constexpr int prf_stride = 3 * 1024 / 2;
            svprfh(svwhilelt_b16(j + prf_stride / 2, end), i_ptr + j + prf_stride / 2, SV_PLDL2STRM);
            svprfh(svwhilelt_b16(j + prf_stride / 2, end), o_ptr + j + prf_stride / 2, SV_PLDL2STRM);
#endif
            svbfloat16_t i0 = svldnt1(svwhilelt_b16(j, end), i_ptr + j);
            svfloat32_t i00 = svreinterpret_f32(svzip1(zero_b, i0));
            svfloat32_t i01 = svreinterpret_f32(svzip2(zero_b, i0));

            svfloat32_t o00;
            svfloat32_t o01;
            if constexpr (load_output) {
                svbfloat16_t o0 = svld1(svwhilelt_b16(j, end), o_ptr + j);
                o00 = svreinterpret_f32(svzip1(zero_b, o0));
                o01 = svreinterpret_f32(svzip2(zero_b, o0));
            } else {
                o00 = o01 = svdup_f32(0);
            }

            svfloat32_t r00 = svmla_n_f32_x(pg32, o00, i00, alpha);
            svfloat32_t r01 = svmla_n_f32_x(pg32, o01, i01, alpha);

            svbfloat16_t op0 = svuzp1(svcvt_bf16_f32_x(pg32, r00), svcvt_bf16_f32_x(pg32, r01));

            svstnt1(svwhilelt_b16(j, end), o_ptr + j, op0);
        }
    });
}

void mul_scalar_add(bfloat16_t* i_ptr, bfloat16_t* o_ptr, int64_t num, float alpha, bool load_output)
{
    if (load_output) {
        mul_scalar_add<true>(i_ptr, o_ptr, num, alpha);
    } else {
        mul_scalar_add<false>(i_ptr, o_ptr, num, alpha);
    }
}

}  // namespace kutacc
