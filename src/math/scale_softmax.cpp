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
#include <arm_sve.h>

#include "kutacc.h"
#include "math/fast_exp.h"

namespace kutacc {
void scale_softmax_kernel(__bf16* act, int64_t width, float scale, float* buf)
{
#ifdef ENABLE_EXTRA_PREFETCH
    constexpr int prf_stride = 512;
#endif
    int64_t vl = svcntw();
    // convert to float && reduce max
    svfloat32_t max_v = svdup_f32(std::numeric_limits<float>::lowest());
    for (int64_t i = 0; i < width; i += vl) {
        svbool_t pg = svwhilelt_b32(i, width);
        svbool_t pg16 = svuzp1_b16(pg, svpfalse());
        svbfloat16_t values_b16 = svld1(pg16, act + i);
#ifdef ENABLE_EXTRA_PREFETCH
        svprfh(svwhilelt_b16(i + prf_stride, width), act + i + prf_stride, SV_PLDL2STRM);
#endif
        svfloat32_t values_f32 = svreinterpret_f32(svzip1(svdup_bf16(0), values_b16));
        values_f32 = svmul_x(pg, values_f32, scale);
        max_v = svmax_m(pg, max_v, values_f32);
        svst1(pg, buf + i, values_f32);
#ifdef ENABLE_EXTRA_PREFETCH
        svprfh(svwhilelt_b32(i + prf_stride, width), buf + i + prf_stride, SV_PLDL2STRM);
#endif
    }
    float max = svmaxv(svptrue_b32(), max_v);
    // sub max & exp & reduce sum
    svfloat32_t sum_v = svdup_f32(0.f);
    for (int64_t i = 0; i < width; i += vl) {
        svbool_t pg = svwhilelt_b32(i, width);
        svfloat32_t values = svld1_f32(pg, buf + i);
#ifdef ENABLE_EXTRA_PREFETCH
        svprfh(svwhilelt_b32(i + prf_stride, width), buf + i + prf_stride, SV_PLDL2STRM);
#endif
        values = fast_exp(pg, svsub_x(pg, values, max));
        sum_v = svadd_m(pg, sum_v, values);
        svst1(pg, buf + i, values);
    }
    // mul sum_inv & convert to bf16
    float sum_inv = 1 / svaddv(svptrue_b32(), sum_v);
    for (int64_t i = 0; i < width; i += vl) {
        svbool_t pg = svwhilelt_b32(i, width);
        svbool_t pg16 = svuzp1_b16(pg, svpfalse());
        svfloat32_t values_f32 = svld1(pg, buf + i);
#ifdef ENABLE_EXTRA_PREFETCH
        svprfh(svwhilelt_b32(i + prf_stride, width), buf + i + prf_stride, SV_PLDL2STRM);
#endif
        values_f32 = svmul_x(pg, values_f32, sum_inv);
        svbfloat16_t values_b16 = svcvt_bf16_x(pg, values_f32);
        values_b16 = svuzp1(values_b16, values_b16);
        svst1(pg16, act + i, values_b16);
#ifdef ENABLE_EXTRA_PREFETCH
        svprfh(svwhilelt_b16(i + prf_stride, width), act + i + prf_stride, SV_PLDL2STRM);
#endif
    }
}

void scale_softmax_h8(__bf16* act_, int64_t act_stride, float* scale, float* buf_, int64_t buf_stride,
    [[maybe_unused]] int64_t height, int64_t width)
{
#ifdef ENABLE_EXTRA_PREFETCH
    constexpr int prf_stride = 512;
#endif
    int64_t vl = svcntw();
    float part_max[8][4];
    float part_sum[8][4];
    // convert to float && reduce max
    kutacc::parallel_for(0, 32, 1, [&](int64_t, int64_t) {
        int64_t hi = kutacc::get_thread_id() / 4;
        int64_t wi = kutacc::get_thread_id() % 4;
        int64_t wStart = wi * (width / 4);
        int64_t wEnd = (wi + 1) * (width / 4);
        __bf16* act = act_ + hi * act_stride;
        float* buf = buf_ + kutacc::get_thread_id() * buf_stride;
        svfloat32_t max_v = svdup_f32(std::numeric_limits<float>::lowest());
        float scale_ = scale[hi];
        for (int64_t i = wStart; i < wEnd; i += vl) {
            svbool_t pg = svwhilelt_b32(i, wEnd);
            svbool_t pg16 = svuzp1_b16(pg, svpfalse());
            svbfloat16_t values_b16 = svld1(pg16, act + i);
#ifdef ENABLE_EXTRA_PREFETCH
            svprfh(svwhilelt_b16(i + prf_stride, wEnd), act + i + prf_stride, SV_PLDL2STRM);
#endif
            svfloat32_t values_f32 = svreinterpret_f32(svzip1(svdup_bf16(0), values_b16));
            values_f32 = svmul_x(pg, values_f32, scale_);
            max_v = svmax_m(pg, max_v, values_f32);
            svst1(pg, buf + i, values_f32);
#ifdef ENABLE_EXTRA_PREFETCH
            svprfh(svwhilelt_b32(i + prf_stride, wEnd), buf + i + prf_stride, SV_PLDL2STRM);
#endif
        }
        part_max[hi][wi] = svmaxv(svptrue_b32(), max_v);
    });

    // sub max & exp & reduce sum
    kutacc::parallel_for(0, 32, 1, [&](int64_t, int64_t) {
        int64_t hi = kutacc::get_thread_id() / 4;
        int64_t wi = kutacc::get_thread_id() % 4;
        int64_t wStart = wi * (width / 4);
        int64_t wEnd = (wi + 1) * (width / 4);
        __bf16* act = act_ + hi * act_stride;
        float* buf = buf_ + kutacc::get_thread_id() * buf_stride;

        float max = std::max({part_max[hi][0], part_max[hi][1], part_max[hi][2], part_max[hi][3]});
        svfloat32_t sum_v = svdup_f32(0.f);
        for (int64_t i = wStart; i < wEnd; i += vl) {
            svbool_t pg = svwhilelt_b32(i, wEnd);
            svfloat32_t values = svld1_f32(pg, buf + i);
#ifdef ENABLE_EXTRA_PREFETCH
            svprfh(svwhilelt_b32(i + prf_stride, wEnd), buf + i + prf_stride, SV_PLDL2STRM);
#endif
            values = fast_exp(pg, svsub_x(pg, values, max));
            sum_v = svadd_m(pg, sum_v, values);
            svst1(pg, buf + i, values);
        }
        part_sum[hi][wi] = svaddv(svptrue_b32(), sum_v);
    });

    // mul sum_inv & convert to bf16
    kutacc::parallel_for(0, 32, 1, [&](int64_t, int64_t) {
        int64_t hi = kutacc::get_thread_id() / 4;
        int64_t wi = kutacc::get_thread_id() % 4;
        int64_t wStart = wi * (width / 4);
        int64_t wEnd = (wi + 1) * (width / 4);
        __bf16* act = act_ + hi * act_stride;
        float* buf = buf_ + kutacc::get_thread_id() * buf_stride;

        float sum = part_sum[hi][0] + part_sum[hi][1] + part_sum[hi][2] + part_sum[hi][3];
        float sum_inv = 1 / sum;
        for (int64_t i = wStart; i < wEnd; i += vl) {
            svbool_t pg = svwhilelt_b32(i, wEnd);
            svbool_t pg16 = svuzp1_b16(pg, svpfalse());
            svfloat32_t values_f32 = svld1(pg, buf + i);
#ifdef ENABLE_EXTRA_PREFETCH
            svprfh(svwhilelt_b32(i + prf_stride, wEnd), buf + i + prf_stride, SV_PLDL2STRM);
#endif
            values_f32 = svmul_x(pg, values_f32, sum_inv);
            svbfloat16_t values_b16 = svcvt_bf16_x(pg, values_f32);
            values_b16 = svuzp1(values_b16, values_b16);
            svst1(pg16, act + i, values_b16);
#ifdef ENABLE_EXTRA_PREFETCH
            svprfh(svwhilelt_b16(i + prf_stride, wEnd), act + i + prf_stride, SV_PLDL2STRM);
#endif
        }
    });
}

void scale_softmax(__bf16* act, int64_t act_stride, float* scale, float* buf, int64_t buf_stride, int64_t height,
    int64_t width)
{
    if (height == 8 && width == 129280) {
        scale_softmax_h8(act, act_stride, scale, buf, buf_stride, height, width);
    } else {
        kutacc::parallel_for(0, height, 1, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; i++) {
                scale_softmax_kernel(act + i * act_stride, width, scale[i], buf + kutacc::get_thread_id() * buf_stride);
            }
        });
    }
}
}
