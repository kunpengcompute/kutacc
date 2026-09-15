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
#include <limits>
#include <arm_sve.h>
#include "kutacc.h"
#include "math/fast_exp.h"

namespace kutacc {

template <typename scalar_t>
void silu_mul_scales(const int64_t width, scalar_t* gateup, float& quant_scale, float& online_scale)
{
    const int64_t step = svcnth();
    const int64_t half_step = svcntw();
    const int64_t prefetch_dis = 128;
    float max = std::numeric_limits<float>::min();
    if constexpr (std::is_same<scalar_t, bfloat16_t>::value) {
        svfloat32_t absmax = svdup_f32(0);
        svbfloat16_t zero_b = svdup_bf16(0);
        for (int64_t i = 0; i < width; i += step) {
            svbool_t pg = svwhilelt_b16(i, width);
            svprfb(svptrue_b8(), gateup + i + prefetch_dis + ((i + prefetch_dis) > width ? 2 * width : 0),
                SV_PLDL1KEEP);
            svprfb(svptrue_b8(), gateup + i + width + prefetch_dis, SV_PLDL1KEEP);
            svbfloat16_t g = svld1(pg, &gateup[i]);
            svbfloat16_t u = svld1(pg, &gateup[i + width]);
            svbool_t p0 = svwhilelt_b32(i, width);
            svbool_t p1 = svwhilelt_b32(i + half_step, width);
            svfloat32_t g0 = svreinterpret_f32(svzip1(zero_b, g));
            svfloat32_t g1 = svreinterpret_f32(svzip2(zero_b, g));
            svfloat32_t e0 = fast_exp(p0, svneg_x(p0, g0));
            svfloat32_t e1 = fast_exp(p1, svneg_x(p1, g1));
            g0 = svdiv_x(p0, g0, svadd_x(p0, e0, 1));
            g1 = svdiv_x(p1, g1, svadd_x(p1, e1, 1));
            svfloat32_t u0 = svreinterpret_f32(svzip1(zero_b, u));
            svfloat32_t u1 = svreinterpret_f32(svzip2(zero_b, u));
            g0 = svmul_x(p0, g0, u0);
            g1 = svmul_x(p1, g1, u1);
            absmax = svmax_m(p0, absmax, svabs_x(p0, g0));
            absmax = svmax_m(p1, absmax, svabs_x(p1, g1));
            svst1(pg, &gateup[i + width], svuzp1(svcvt_bf16_x(p0, g0), svcvt_bf16_x(p1, g1)));
        }
        max = svmaxv(svptrue_b32(), absmax);
    }
    quant_scale = 127.0f / max;
    online_scale = max / 127.0f;
}

template <typename scalar_t>
void silu_mul_quant_kernel(const int64_t width, scalar_t* gateup, int8_t* out, float& online_scale)
{
    const int64_t step = svcnth();
    float quant_scale;
    silu_mul_scales(width, gateup, quant_scale, online_scale);
    if constexpr (std::is_same<scalar_t, bfloat16_t>::value) {
        svbfloat16_t zero_b = svdup_bf16(0);
        for (int64_t i = 0; i < width; i += step) {
            svbool_t pg = svwhilelt_b16(i, width);
            svbfloat16_t u = svld1(pg, &gateup[i + width]);
            svfloat32_t u0 = svreinterpret_f32(svzip1(zero_b, u));
            svfloat32_t u1 = svreinterpret_f32(svzip2(zero_b, u));
            u0 = svmul_x(svptrue_b32(), u0, quant_scale);
            u0 = svrintn_x(svptrue_b32(), u0);
            u1 = svmul_x(svptrue_b32(), u1, quant_scale);
            u1 = svrintn_x(svptrue_b32(), u1);
            svint32_t r0 = svcvt_s32_x(svptrue_b32(), u0);
            svint32_t r1 = svcvt_s32_x(svptrue_b32(), u1);
            svint16_t results = svuzp1(svreinterpret_s16(r0), svreinterpret_s16(r1));
            results = svmax_x(pg, results, INT8_MIN);
            results = svmin_x(pg, results, INT8_MAX);
            svst1b(pg, &out[i], results);
        }
    }
}

void silu_mul_quant(int64_t gateupsize, int64_t gateupnumel, int8_t* outdata_ptr, bfloat16_t* gateupdata_ptr,
    float* online_scalesdata_ptr)
{
    int64_t width = gateupsize;
    int64_t height = gateupnumel / width;
    auto online_scales_data = online_scalesdata_ptr;

    bfloat16_t* gateup_data = gateupdata_ptr;
    int8_t* out_data = outdata_ptr;
    kutacc::parallel_for(0, height, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
            // warning: width / 2 !
            silu_mul_quant_kernel(width / 2, gateup_data + i * width, out_data + i * width / 2, online_scales_data[i]);
        };
    });
}

}  // namespace kutacc
