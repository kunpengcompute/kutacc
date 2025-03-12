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

TEST(TransitionTest, transition_nullptr_1)
{
    float a = 1.5f;
    void *a_ptr = &a;
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
    kutacc_af2_trans_weights_t *weight_ptr = (kutacc_af2_trans_weights_t*)malloc(sizeof(kutacc_af2_trans_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_trans_weights_t){.c_o = 2, .c_i = 2, .linear1_w = x_ptr, .linear1_b = x_ptr,
            .linear2_w = x_ptr, .linear2_b = x_ptr};
    }
    kutacc_af2_transition(nullptr, weight_ptr, a_ptr);
    std::cerr << "transition 1st param trans_inputs_ptr is nullptr" << std::endl;
}

TEST(TransitionTest, transition_nullptr_2)
{
    float a = 1.5f;
    void *a_ptr = &a;
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
    kutacc_af2_trans_act_inputs_t *input_ptr = (kutacc_af2_trans_act_inputs_t*)malloc(sizeof(kutacc_af2_trans_act_inputs_t));
    if (input_ptr != nullptr) {
        *input_ptr = (kutacc_af2_trans_act_inputs_t){.batch = 2, .n_res = 2, .input_act = x_ptr, .intermediate_act = x_ptr};
    }
    kutacc_af2_transition(input_ptr, nullptr, a_ptr);
    std::cerr << "transition 2nd param trans_weights_ptr is nullptr" << std::endl;
}

TEST(TransitionTest, transition_nullptr_3)
{
    float a = 1.5f;
    void *a_ptr = &a;
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
    kutacc_af2_trans_act_inputs_t *input_ptr = (kutacc_af2_trans_act_inputs_t*)malloc(sizeof(kutacc_af2_trans_act_inputs_t));
    if (input_ptr != nullptr) {
        *input_ptr = (kutacc_af2_trans_act_inputs_t){.batch = 2, .n_res = 2, .input_act = x_ptr, .intermediate_act = x_ptr};
    }
    kutacc_af2_trans_weights_t *weight_ptr = (kutacc_af2_trans_weights_t*)malloc(sizeof(kutacc_af2_trans_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_trans_weights_t){.c_o = 2, .c_i = 2, .linear1_w = x_ptr, .linear1_b = x_ptr,
            .linear2_w = x_ptr, .linear2_b = x_ptr};
    }
    kutacc_af2_transition(input_ptr, weight_ptr, nullptr);
    std::cerr << "transition 3rd param out is nullptr" << std::endl;
}
