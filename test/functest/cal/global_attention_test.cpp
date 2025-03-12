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

#include <vector>
#include <arm_bf16.h>
#include <gtest/gtest.h>
#include "kutacc.h"

TEST(GlobalAttentionTest, global_attention_nullptr_1)
{
    float a = 0.5f;
    void *a_ptr = (void*)&a;
    constexpr int64_t batch = 2;
    constexpr int64_t seq_len = 2;
    constexpr int64_t nchannels = 2;
    float x[batch][seq_len][nchannels] = {
      // batch=0
        {
            {1.0f, 2.0f},  // seq=0
            {3.0f, 4.0f},  // seq=1
        },
        // batch=1
        {
            {7.0f, 8.0f},  // seq=0
            {9.0f, 10.0f}, // seq=1
        }
    };
    void *x_ptr = (void *)&x;

    kutacc_af2_attention_weights_t *weight_ptr = (kutacc_af2_attention_weights_t*)malloc(sizeof(kutacc_af2_attention_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_attention_weights_t){.nchannels = nchannels, .nheads = 1, .head_size = 2, .query_w = x_ptr,
            .key_w = x_ptr, .value_w = x_ptr, .gating_w = x_ptr, .gating_b = x_ptr, 
            .output_w = x_ptr, .output_b = x_ptr};
    }
    kutacc_af2_global_attention(nullptr, a_ptr, a_ptr, weight_ptr, a_ptr);
    std::cerr << "global_attention 1st param q_based_ptr is nullptr" << std::endl;
}

TEST(GlobalAttentionTest, global_attention_nullptr_2)
{
    float a = 0.5f;
    void *a_ptr = (void*)&a;
    constexpr int64_t batch = 2;
    constexpr int64_t seq_len = 2;
    constexpr int64_t nchannels = 2;
    float x[batch][seq_len][nchannels] = {
      // batch=0
        {
            {1.0f, 2.0f},  // seq=0
            {3.0f, 4.0f},  // seq=1
        },
        // batch=1
        {
            {7.0f, 8.0f},  // seq=0
            {9.0f, 10.0f}, // seq=1
        }
    };
    void *x_ptr = (void *)&x;

   
    kutacc_af2_attention_weights_t *weight_ptr = (kutacc_af2_attention_weights_t*)malloc(sizeof(kutacc_af2_attention_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_attention_weights_t){.nchannels = nchannels, .nheads = 1, .head_size = 2, .query_w = x_ptr,
            .key_w = x_ptr, .value_w = x_ptr, .gating_w = x_ptr, .gating_b = x_ptr, 
            .output_w = x_ptr, .output_b = x_ptr};
    }
    kutacc_af2_attention_inputs_t *q_based_ptr = (kutacc_af2_attention_inputs_t*)malloc(sizeof(kutacc_af2_attention_inputs_t));
    if (q_based_ptr != nullptr) {
        *q_based_ptr = (kutacc_af2_attention_inputs_t){.batch = batch, .seq_len = seq_len, .q = x_ptr,
            .k = x_ptr, .v = x_ptr, .gate = x_ptr, .avg = x_ptr};
    }
    kutacc_af2_global_attention(q_based_ptr, nullptr, a_ptr, weight_ptr, a_ptr);
    std::cerr << "global_attention 2nd param q_data is nullptr" << std::endl;
}

TEST(GlobalAttentionTest, global_attention_nullptr_3)
{
    float a = 0.5f;
    void *a_ptr = (void*)&a;
    constexpr int64_t batch = 2;
    constexpr int64_t seq_len = 2;
    constexpr int64_t nchannels = 2;
    float x[batch][seq_len][nchannels] = {
      // batch=0
        {
            {1.0f, 2.0f},  // seq=0
            {3.0f, 4.0f},  // seq=1
        },
        // batch=1
        {
            {7.0f, 8.0f},  // seq=0
            {9.0f, 10.0f}, // seq=1
        }
    };
    void *x_ptr = (void *)&x;

   
    kutacc_af2_attention_weights_t *weight_ptr = (kutacc_af2_attention_weights_t*)malloc(sizeof(kutacc_af2_attention_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_attention_weights_t){.nchannels = nchannels, .nheads = 1, .head_size = 2, .query_w = x_ptr,
            .key_w = x_ptr, .value_w = x_ptr, .gating_w = x_ptr, .gating_b = x_ptr, 
            .output_w = x_ptr, .output_b = x_ptr};
    }
    kutacc_af2_attention_inputs_t *q_based_ptr = (kutacc_af2_attention_inputs_t*)malloc(sizeof(kutacc_af2_attention_inputs_t));
    if (q_based_ptr != nullptr) {
        *q_based_ptr = (kutacc_af2_attention_inputs_t){.batch = batch, .seq_len = seq_len, .q = x_ptr,
            .k = x_ptr, .v = x_ptr, .gate = x_ptr, .avg = x_ptr};
    }
    kutacc_af2_global_attention(q_based_ptr, a_ptr, nullptr, weight_ptr, a_ptr);
    std::cerr << "global_attention 3rd param q_mask is nullptr" << std::endl;
}

TEST(GlobalAttentionTest, global_attention_nullptr_4)
{
    float a = 0.5f;
    void *a_ptr = (void*)&a;
    constexpr int64_t batch = 2;
    constexpr int64_t seq_len = 2;
    constexpr int64_t nchannels = 2;
    float x[batch][seq_len][nchannels] = {
      // batch=0
        {
            {1.0f, 2.0f},  // seq=0
            {3.0f, 4.0f},  // seq=1
        },
        // batch=1
        {
            {7.0f, 8.0f},  // seq=0
            {9.0f, 10.0f}, // seq=1
        }
    };
    void *x_ptr = (void *)&x;

   
    kutacc_af2_attention_inputs_t *q_based_ptr = (kutacc_af2_attention_inputs_t*)malloc(sizeof(kutacc_af2_attention_inputs_t));
    if (q_based_ptr != nullptr) {
        *q_based_ptr = (kutacc_af2_attention_inputs_t){.batch = batch, .seq_len = seq_len, .q = x_ptr,
            .k = x_ptr, .v = x_ptr, .gate = x_ptr, .avg = x_ptr};
    }
    kutacc_af2_global_attention(q_based_ptr, a_ptr, a_ptr, nullptr, a_ptr);
    std::cerr << "global_attention 4th param weight_ptr is nullptr" << std::endl;
}

TEST(GlobalAttentionTest, global_attention_nullptr_5)
{
    float a = 0.5f;
    void *a_ptr = (void*)&a;
    constexpr int64_t batch = 2;
    constexpr int64_t seq_len = 2;
    constexpr int64_t nchannels = 2;
    float x[batch][seq_len][nchannels] = {
      // batch=0
        {
            {1.0f, 2.0f},  // seq=0
            {3.0f, 4.0f},  // seq=1
        },
        // batch=1
        {
            {7.0f, 8.0f},  // seq=0
            {9.0f, 10.0f}, // seq=1
        }
    };
    void *x_ptr = (void *)&x;

   
    kutacc_af2_attention_weights_t *weight_ptr = (kutacc_af2_attention_weights_t*)malloc(sizeof(kutacc_af2_attention_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_attention_weights_t){.nchannels = nchannels, .nheads = 1, .head_size = 2, .query_w = x_ptr,
            .key_w = x_ptr, .value_w = x_ptr, .gating_w = x_ptr, .gating_b = x_ptr, 
            .output_w = x_ptr, .output_b = x_ptr};
    }
    kutacc_af2_attention_inputs_t *q_based_ptr = (kutacc_af2_attention_inputs_t*)malloc(sizeof(kutacc_af2_attention_inputs_t));
    if (q_based_ptr != nullptr) {
        *q_based_ptr = (kutacc_af2_attention_inputs_t){.batch = batch, .seq_len = seq_len, .q = x_ptr,
            .k = x_ptr, .v = x_ptr, .gate = x_ptr, .avg = x_ptr};
    }
    kutacc_af2_global_attention(q_based_ptr, a_ptr, a_ptr, weight_ptr, nullptr);
    std::cerr << "global_attention 5th param out is nullptr" << std::endl;
}
