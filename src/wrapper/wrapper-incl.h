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

DEFINE_WRAPPER(8, s8, int8_t, svint8_t);
DEFINE_WRAPPER(8, u8, uint8_t, svuint8_t);
DEFINE_WRAPPER(16, s16, int16_t, svint16_t);
DEFINE_WRAPPER(16, u16, uint16_t, svuint16_t);
DEFINE_WRAPPER(32, s32, int32_t, svint32_t);
DEFINE_WRAPPER(32, u32, uint32_t, svuint32_t);
DEFINE_WRAPPER(64, s64, int64_t, svint64_t);
DEFINE_WRAPPER(64, u64, uint64_t, svuint64_t);
DEFINE_WRAPPER(16, f16, __fp16, svfloat16_t);
DEFINE_WRAPPER(16, bf16, __bf16, svbfloat16_t);
DEFINE_WRAPPER(32, f32, float, svfloat32_t);
DEFINE_WRAPPER(64, f64, double, svfloat64_t);
#undef DEFINE_WRAPPER
