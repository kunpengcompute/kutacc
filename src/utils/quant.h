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
#include <algorithm>
#include <arm_sve.h>
#include <arm_neon.h>

namespace kutacc {

void quant(int64_t height, int64_t width, const __bf16 *input, int64_t input_stride, int8_t *out, int64_t out_stride,
    float *scale);

void quant_pack(int64_t height, int64_t width, bfloat16_t* input_data, int8_t* output_data, float* scale_data);

void quant_save_pack(int64_t height, int64_t width, bfloat16_t* input_data, int64_t ldo, int8_t* output_data,
    int64_t lds, float* scale_data, int8_t* packed_output_data, float* packed_scale_data);

} // namespace kutacc