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
#include <arm_sme.h>
#include <cmath>
#include "kutacc.h"
#include "utils/check.h"
#include "utils/small_vector.h"

namespace kutacc {
    
const int64_t STEP_W_QUARTER = svcntw();
const int64_t STEP_W_HALF = svcnth();
const int64_t STEP_W = svcntb();
const int64_t PACK_STEP = 16;

__arm_new("za") void quant_pack_kernel(int64_t h_start, int64_t h_end, int64_t w_start, int64_t w_end, int64_t lda,
    bfloat16_t* acts, float* quant_scales, int64_t ldpo, int8_t* packed_out, int64_t ldo = -1,
    int8_t* out = nullptr) __arm_streaming
{
    svbool_t pg16 = svptrue_b16();
    svbool_t pg32 = svptrue_b32();
    svbfloat16_t zero_bf16 = svdup_bf16(0);
    for (int64_t h = h_start; h < h_end; h += PACK_STEP) {
        int64_t part_height = std::min(h_end - h, PACK_STEP);
        int64_t h_num = part_height * 4;
        bfloat16_t* cur_acts = acts + h * lda;
        int8_t* cur_out = packed_out + h * ldpo + w_start * part_height;
        for (int64_t w = w_start; w + STEP_W <= w_end; w += STEP_W) {
            svbool_t pg_8 = svwhilelt_b8(w, w_end);
            for (int64_t i = 0; i < part_height; i++) {
                float scales_inv = 1 / quant_scales[h + i];
                svbfloat16_t v0 = svld1(pg16, cur_acts + i * lda + w);
                svbfloat16_t v1 = svld1(pg16, cur_acts + i * lda + w + STEP_W_HALF);
                svfloat32_t v00 = svreinterpret_f32(svzip1(zero_bf16, v0));
                svfloat32_t v01 = svreinterpret_f32(svzip2(zero_bf16, v0));
                svfloat32_t v10 = svreinterpret_f32(svzip1(zero_bf16, v1));
                svfloat32_t v11 = svreinterpret_f32(svzip2(zero_bf16, v1));
                v00 = svmul_x(pg32, v00, scales_inv);
                v01 = svmul_x(pg32, v01, scales_inv);
                v10 = svmul_x(pg32, v10, scales_inv);
                v11 = svmul_x(pg32, v11, scales_inv);

                svfloat16_t o0 = svuzp1(svcvt_f16_x(pg32, v00), svcvt_f16_x(pg32, v01));
                svfloat16_t o1 = svuzp1(svcvt_f16_x(pg32, v10), svcvt_f16_x(pg32, v11));
                svint8_t t = svuzp1_s8(svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0))),
                    svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1))));

                if (out)
                    svst1(pg_8, &out[(h + i) * ldo + w], t);
                svwrite_hor_za8_s8_m(0, i * 4, pg_8, t);
            }
            svint32_t out_v = svdup_s32(0);
            svbool_t pg_out = svwhilelt_b32(0l, part_height);
            for (int64_t i = 0; i < STEP_W_QUARTER; i++) {
                out_v = svread_ver_za32_s32_m(out_v, svptrue_b32(), 0, i);
                svst1(pg_out, (int32_t*)&cur_out[i * h_num], out_v);
            }
            cur_out += STEP_W_QUARTER * h_num;
        }
    }
}

inline void get_max_scale_kernel(int64_t h_start, int64_t h_end, int64_t w_start, int64_t w_end, int64_t ldw,
    bfloat16_t* input, float* scale)
{
    svbool_t pg32 = svptrue_b32();
    svbfloat16_t zero_bf16 = svdup_bf16(0);
    for (int64_t i = h_start; i < h_end; ++i) {
        svfloat32_t mx_v = svdup_f32(std::numeric_limits<float>::lowest());
        for (int64_t j = w_start; j < w_end; j += svcnth()) {
            svbool_t pg16 = svwhilelt_b16(j, w_end);
            svbfloat16_t v = svld1(pg16, input + i * ldw + j);
            svfloat32_t v0 = svreinterpret_f32(svzip1(zero_bf16, v));
            svfloat32_t v1 = svreinterpret_f32(svzip2(zero_bf16, v));
            v0 = svabs_f32_x(pg32, v0);
            v1 = svabs_f32_x(pg32, v1);
            mx_v = svmax_x(pg32, v0, mx_v);
            mx_v = svmax_x(pg32, v1, mx_v);
        }
        float max_value = svmaxv_f32(pg32, mx_v);
        scale[i] = max_value / 127.0f;
    }
}

void quant_pack(int64_t height, int64_t width, bfloat16_t* input_data, int8_t* output_data, float* scale_data)
{
    KUTACC_CHECK((width % STEP_W) == 0, "(width % STEP_W) != 0, ", width, " ", STEP_W);
    int64_t num_threads = kutacc::get_thread_num();
    int64_t part_height = (height + num_threads * PACK_STEP - 1) / (num_threads * PACK_STEP) * PACK_STEP;
    int64_t used_thread_nums = (height + part_height - 1) / part_height;
    int64_t left_thread_nums = num_threads / used_thread_nums;
    int64_t part_width = (width + (left_thread_nums * STEP_W) - 1) / (left_thread_nums * STEP_W) * STEP_W;
    KUTACC_CHECK((part_width % STEP_W) == 0, "(part_width % STEP_W) != 0, ", part_width, " ", STEP_W);
    kutacc::SmallVector<float, 32 * 128> tmp_scale(left_thread_nums * height);
    kutacc::parallel_for(0, num_threads, 1, [&](int64_t, int64_t) {
        int64_t thread_id = kutacc::get_thread_id();
        if (thread_id < left_thread_nums * used_thread_nums) {
            int64_t h_thread_id = thread_id % used_thread_nums;
            int64_t w_thread_id = thread_id / used_thread_nums;
            int64_t h_start = h_thread_id * part_height;
            int64_t h_end = std::min((int64_t)height, (h_thread_id + 1) * part_height);
            int64_t w_start = w_thread_id * part_width;
            int64_t w_end = std::min((int64_t)width, (w_thread_id + 1) * part_width);
            float* part_scale = tmp_scale.data() + w_thread_id * height;
            get_max_scale_kernel(h_start, h_end, w_start, w_end, width, input_data, part_scale);
            kutacc::parallel_barrier();
            kutacc::SmallVector<float, 128> result_scale(height);
            for (int64_t h = h_start; h < h_end; ++h) {
                float tmp = 0;
                for (int64_t w = 0; w < left_thread_nums; ++w)
                    tmp = std::max(tmp, tmp_scale.data()[w * height + h]);
                result_scale.data()[h] = tmp;
                if (w_thread_id == 0)
                    scale_data[h] = tmp;
            }
            quant_pack_kernel(h_start, h_end, w_start, w_end, width, input_data, result_scale.data(), width,
                output_data);
        } else {
            kutacc::parallel_barrier();
        }
    });
}

void quant_save_pack(int64_t height, int64_t width, bfloat16_t* input_data, int64_t ldo, int8_t* output_data,
    int64_t lds, float* scale_data, int8_t* packed_output_data, float* packed_scale_data)
{
    KUTACC_CHECK((width % STEP_W) == 0, "(width % STEP_W) != 0, ", width, " ", STEP_W);
    int64_t used_thread_nums = 16;
    int64_t part_width = (width + (used_thread_nums * STEP_W) - 1) / (used_thread_nums * STEP_W) * STEP_W;
    int64_t left_thread_nums = kutacc::get_thread_num() / used_thread_nums;
    int64_t part_height = (height + left_thread_nums * PACK_STEP - 1) / (left_thread_nums * PACK_STEP) * PACK_STEP;
    kutacc::SmallVector<float, 32 * 128> tmp_scale(used_thread_nums * height);
    kutacc::parallel_for(0, left_thread_nums * used_thread_nums, 1, [&](int64_t, int64_t) {
        int64_t thread_id = kutacc::get_thread_id();
        int64_t h_thread_id = thread_id / used_thread_nums;
        int64_t w_thread_id = thread_id % used_thread_nums;
        int64_t h_start = h_thread_id * part_height;
        int64_t h_end = std::min((int64_t)height, (h_thread_id + 1) * part_height);
        int64_t w_start = w_thread_id * part_width;
        int64_t w_end = std::min((int64_t)width, (w_thread_id + 1) * part_width);
        float* part_scale = tmp_scale.data() + w_thread_id * height;
        get_max_scale_kernel(h_start, h_end, w_start, w_end, width, input_data, part_scale);
        kutacc::parallel_barrier();
        kutacc::SmallVector<float, 128> result_scale(height);
        for (int64_t h = h_start; h < h_end; ++h) {
            float tmp = 0;
            for (int64_t w = 0; w < used_thread_nums; ++w)
                tmp = std::max(tmp, tmp_scale.data()[w * height + h]);
            result_scale.data()[h] = tmp;
            if (w_thread_id == 0)
                scale_data[h * lds] = packed_scale_data[h] = tmp;
        }
        quant_pack_kernel(h_start, h_end, 0, part_width, width, input_data + w_start, result_scale.data(), part_width,
            packed_output_data + w_thread_id * part_width * height, ldo, output_data + w_start);
    });
}


void quant(int64_t height, int64_t width, const __bf16 *input, int64_t input_stride, int8_t *out, int64_t out_stride,
    float *scale)
{
    const int64_t STEP_HALF = svcnth();
    const int64_t STEP_32 = svcnth() * 2;
    svbool_t pg16 = svptrue_b16();
    svbool_t pg_32 = svptrue_b32();
    svbool_t pg8 = svptrue_b8();
    svbfloat16_t zero_b = svdup_bf16(0);
    kutacc::parallel_for(0, height, 1, [&](int64_t start, int64_t end) {
        for (int i = start; i < end; ++i) {
            svfloat32_t mx_v = svdup_f32(0);
            int j = 0;
            for (; j + STEP_32 <= width; j += STEP_32) {
                svbfloat16_t i0 = svld1(pg16, input + i * input_stride + j);
                svbfloat16_t i1 = svld1(pg16, input + i * input_stride + j + STEP_HALF);
                svfloat32_t i00 = svreinterpret_f32(svzip1(zero_b, i0));
                svfloat32_t i01 = svreinterpret_f32(svzip2(zero_b, i0));
                svfloat32_t i10 = svreinterpret_f32(svzip1(zero_b, i1));
                svfloat32_t i11 = svreinterpret_f32(svzip2(zero_b, i1));
                i00 = svabs_f32_x(pg_32, i00);
                i01 = svabs_f32_x(pg_32, i01);
                i10 = svabs_f32_x(pg_32, i10);
                i11 = svabs_f32_x(pg_32, i11);
                mx_v = svmax_x(pg_32, i00, mx_v);
                mx_v = svmax_x(pg_32, i01, mx_v);
                mx_v = svmax_x(pg_32, i10, mx_v);
                mx_v = svmax_x(pg_32, i11, mx_v);
            }
            float max_value = svmaxv_f32(pg_32, mx_v);
            for (; j < width; ++j) {
                max_value = std::max(max_value, std::abs((float)input[i * input_stride + j]));
            }
            float scale_val = max_value / 127.0f;
            scale[i] = scale_val;
            float scale_val_inv = 1 / scale_val;
            for (j = 0; j + STEP_32 <= width; j += STEP_32) {
                svbfloat16_t i0 = svld1(pg16, input + i * input_stride + j);
                svbfloat16_t i1 = svld1(pg16, input + i * input_stride + j + STEP_HALF);
                svfloat32_t i00 = svreinterpret_f32(svzip1(zero_b, i0));
                svfloat32_t i01 = svreinterpret_f32(svzip2(zero_b, i0));
                svfloat32_t i10 = svreinterpret_f32(svzip1(zero_b, i1));
                svfloat32_t i11 = svreinterpret_f32(svzip2(zero_b, i1));
                i00 = svmul_x(pg_32, i00, scale_val_inv);
                i01 = svmul_x(pg_32, i01, scale_val_inv);
                i10 = svmul_x(pg_32, i10, scale_val_inv);
                i11 = svmul_x(pg_32, i11, scale_val_inv);

                svfloat16_t o0 = svuzp1(svcvt_f16_x(pg_32, i00), svcvt_f16_x(pg_32, i01));
                svfloat16_t o1 = svuzp1(svcvt_f16_x(pg_32, i10), svcvt_f16_x(pg_32, i11));
                svint8_t t0 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o0)));
                svint8_t t1 = svqxtnb_s16(svcvt_s16_x(pg16, svrintn_x(pg16, o1)));
                svst1(pg8, out + i * out_stride + j, svuzp1(t0, t1));
            }
            for (; j < width; ++j) {
                out[i * out_stride + j] =
                    std::nearbyint(std::clamp(input[i * input_stride + j] / scale_val, -127.0f, 127.0f));
            }
        }
    });
}

}  // namespace kutacc
