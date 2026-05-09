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

TEST(InvariantPointTest, invariant_point_nullptr1)
{
    float data = 0.6f;
    void *data_ptr = (void*)&data;
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

    kutacc_af2_ipa_o_inputs_t *o_ptrs = (kutacc_af2_ipa_o_inputs_t*)malloc(sizeof(kutacc_af2_ipa_o_inputs_t));
    if (o_ptrs != nullptr) {
        *o_ptrs = (kutacc_af2_ipa_o_inputs_t){.o =  x_ptr, .o_pt = x_ptr, .o_pt_norm =  x_ptr,
            .o_pair = x_ptr};
    }
    kutacc_af2_ipa_weights_t *weight_ptr = (kutacc_af2_ipa_weights_t*)malloc(sizeof(kutacc_af2_ipa_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_ipa_weights_t){.c_z = 2, .c_hidden = 2, .no_heads = 2, .no_qk_points = 2,
            .no_v_points = 2, .head_weights = x_ptr, .weights_head_weights = x_ptr,
            .linear_b_w = x_ptr, .linear_b_b = x_ptr};
    }
    kutacc_af2_invariant_point(nullptr, o_ptrs, data_ptr, data_ptr, data_ptr, data_ptr, weight_ptr);
    std::cerr << "invariant_point_attention 1st param ipa_s_ptrs is nullptr" << std::endl;
}

TEST(InvariantPointTest, invariant_point_nullptr2)
{
    float data = 0.6f;
    void *data_ptr = (void*)&data;
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


    kutacc_af2_ipa_s_inputs_t *s_ptrs = (kutacc_af2_ipa_s_inputs_t*)malloc(sizeof(kutacc_af2_ipa_s_inputs_t));
    if (s_ptrs != nullptr) {
        *s_ptrs = (kutacc_af2_ipa_s_inputs_t){.n_res = 2, .a = x_ptr, .b = x_ptr,
            .q = x_ptr, .k = x_ptr, .v = x_ptr,
            .q_pts = x_ptr, .k_pts = x_ptr, .v_pts = x_ptr};
    }

    kutacc_af2_ipa_weights_t *weight_ptr = (kutacc_af2_ipa_weights_t*)malloc(sizeof(kutacc_af2_ipa_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_ipa_weights_t){.c_z = 2, .c_hidden = 2, .no_heads = 2, .no_qk_points = 2,
            .no_v_points = 2, .head_weights = x_ptr, .weights_head_weights = x_ptr,
            .linear_b_w = x_ptr, .linear_b_b = x_ptr};
    }
    kutacc_af2_invariant_point(s_ptrs, nullptr, data_ptr, data_ptr, data_ptr, data_ptr, weight_ptr);
    std::cerr << "invariant_point_attention 2nd param ipa_o_ptrs is nullptr" << std::endl;
}

TEST(InvariantPointTest, invariant_point_nullptr3)
{
    float data = 0.6f;
    void *data_ptr = (void*)&data;
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


    kutacc_af2_ipa_s_inputs_t *s_ptrs = (kutacc_af2_ipa_s_inputs_t*)malloc(sizeof(kutacc_af2_ipa_s_inputs_t));
    if (s_ptrs != nullptr) {
        *s_ptrs = (kutacc_af2_ipa_s_inputs_t){.n_res = 2, .a = x_ptr, .b = x_ptr,
            .q = x_ptr, .k = x_ptr, .v = x_ptr,
            .q_pts = x_ptr, .k_pts = x_ptr, .v_pts = x_ptr};
    }
    kutacc_af2_ipa_o_inputs_t *o_ptrs = (kutacc_af2_ipa_o_inputs_t*)malloc(sizeof(kutacc_af2_ipa_o_inputs_t));
    if (o_ptrs != nullptr) {
        *o_ptrs = (kutacc_af2_ipa_o_inputs_t){.o =  x_ptr, .o_pt = x_ptr, .o_pt_norm =  x_ptr,
            .o_pair = x_ptr};
    }
    kutacc_af2_ipa_weights_t *weight_ptr = (kutacc_af2_ipa_weights_t*)malloc(sizeof(kutacc_af2_ipa_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_ipa_weights_t){.c_z = 2, .c_hidden = 2, .no_heads = 2, .no_qk_points = 2,
            .no_v_points = 2, .head_weights = x_ptr, .weights_head_weights = x_ptr,
            .linear_b_w = x_ptr, .linear_b_b = x_ptr};
    }
    kutacc_af2_invariant_point(s_ptrs, o_ptrs, nullptr, data_ptr, data_ptr, data_ptr, weight_ptr);
    std::cerr << "invariant_point_attention 3rd param z is nullptr" << std::endl;
}

TEST(InvariantPointTest, invariant_point_nullptr4)
{
    float data = 0.6f;
    void *data_ptr = (void*)&data;
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


    kutacc_af2_ipa_s_inputs_t *s_ptrs = (kutacc_af2_ipa_s_inputs_t*)malloc(sizeof(kutacc_af2_ipa_s_inputs_t));
    if (s_ptrs != nullptr) {
        *s_ptrs = (kutacc_af2_ipa_s_inputs_t){.n_res = 2, .a = x_ptr, .b = x_ptr,
            .q = x_ptr, .k = x_ptr, .v = x_ptr,
            .q_pts = x_ptr, .k_pts = x_ptr, .v_pts = x_ptr};
    }
    kutacc_af2_ipa_o_inputs_t *o_ptrs = (kutacc_af2_ipa_o_inputs_t*)malloc(sizeof(kutacc_af2_ipa_o_inputs_t));
    if (o_ptrs != nullptr) {
        *o_ptrs = (kutacc_af2_ipa_o_inputs_t){.o =  x_ptr, .o_pt = x_ptr, .o_pt_norm =  x_ptr,
            .o_pair = x_ptr};
    }
    kutacc_af2_ipa_weights_t *weight_ptr = (kutacc_af2_ipa_weights_t*)malloc(sizeof(kutacc_af2_ipa_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_ipa_weights_t){.c_z = 2, .c_hidden = 2, .no_heads = 2, .no_qk_points = 2,
            .no_v_points = 2, .head_weights = x_ptr, .weights_head_weights = x_ptr,
            .linear_b_w = x_ptr, .linear_b_b = x_ptr};
    }
    kutacc_af2_invariant_point(s_ptrs, o_ptrs, data_ptr, nullptr, data_ptr, data_ptr, weight_ptr);
    std::cerr << "invariant_point_attention 4th param rigid_rot_mats is nullptr" << std::endl;
}

TEST(InvariantPointTest, invariant_point_nullptr5)
{
    float data = 0.6f;
    void *data_ptr = (void*)&data;
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


    kutacc_af2_ipa_s_inputs_t *s_ptrs = (kutacc_af2_ipa_s_inputs_t*)malloc(sizeof(kutacc_af2_ipa_s_inputs_t));
    if (s_ptrs != nullptr) {
        *s_ptrs = (kutacc_af2_ipa_s_inputs_t){.n_res = 2, .a = x_ptr, .b = x_ptr,
            .q = x_ptr, .k = x_ptr, .v = x_ptr,
            .q_pts = x_ptr, .k_pts = x_ptr, .v_pts = x_ptr};
    }
    kutacc_af2_ipa_o_inputs_t *o_ptrs = (kutacc_af2_ipa_o_inputs_t*)malloc(sizeof(kutacc_af2_ipa_o_inputs_t));
    if (o_ptrs != nullptr) {
        *o_ptrs = (kutacc_af2_ipa_o_inputs_t){.o =  x_ptr, .o_pt = x_ptr, .o_pt_norm =  x_ptr,
            .o_pair = x_ptr};
    }
    kutacc_af2_ipa_weights_t *weight_ptr = (kutacc_af2_ipa_weights_t*)malloc(sizeof(kutacc_af2_ipa_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_ipa_weights_t){.c_z = 2, .c_hidden = 2, .no_heads = 2, .no_qk_points = 2,
            .no_v_points = 2, .head_weights = x_ptr, .weights_head_weights = x_ptr,
            .linear_b_w = x_ptr, .linear_b_b = x_ptr};
    }
    kutacc_af2_invariant_point(s_ptrs, o_ptrs, data_ptr, data_ptr, nullptr, data_ptr, weight_ptr);
    std::cerr << "invariant_point_attention 5th param rigid_trans is nullptr" << std::endl;
}

TEST(InvariantPointTest, invariant_point_nullptr6)
{
    float data = 0.6f;
    void *data_ptr = (void*)&data;
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


    kutacc_af2_ipa_s_inputs_t *s_ptrs = (kutacc_af2_ipa_s_inputs_t*)malloc(sizeof(kutacc_af2_ipa_s_inputs_t));
    if (s_ptrs != nullptr) {
        *s_ptrs = (kutacc_af2_ipa_s_inputs_t){.n_res = 2, .a = x_ptr, .b = x_ptr,
            .q = x_ptr, .k = x_ptr, .v = x_ptr,
            .q_pts = x_ptr, .k_pts = x_ptr, .v_pts = x_ptr};
    }
    kutacc_af2_ipa_o_inputs_t *o_ptrs = (kutacc_af2_ipa_o_inputs_t*)malloc(sizeof(kutacc_af2_ipa_o_inputs_t));
    if (o_ptrs != nullptr) {
        *o_ptrs = (kutacc_af2_ipa_o_inputs_t){.o =  x_ptr, .o_pt = x_ptr, .o_pt_norm =  x_ptr,
            .o_pair = x_ptr};
    }
    kutacc_af2_ipa_weights_t *weight_ptr = (kutacc_af2_ipa_weights_t*)malloc(sizeof(kutacc_af2_ipa_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_ipa_weights_t){.c_z = 2, .c_hidden = 2, .no_heads = 2, .no_qk_points = 2,
            .no_v_points = 2, .head_weights = x_ptr, .weights_head_weights = x_ptr,
            .linear_b_w = x_ptr, .linear_b_b = x_ptr};
    }
    kutacc_af2_invariant_point(s_ptrs, o_ptrs, data_ptr, data_ptr, data_ptr, nullptr, weight_ptr);
    std::cerr << "invariant_point_attention 6th param mask is nullptr" << std::endl;
}

TEST(InvariantPointTest, invariant_point_nullptr7)
{
    float data = 0.6f;
    void *data_ptr = (void*)&data;
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


    kutacc_af2_ipa_s_inputs_t *s_ptrs = (kutacc_af2_ipa_s_inputs_t*)malloc(sizeof(kutacc_af2_ipa_s_inputs_t));
    if (s_ptrs != nullptr) {
        *s_ptrs = (kutacc_af2_ipa_s_inputs_t){.n_res = 2, .a = x_ptr, .b = x_ptr,
            .q = x_ptr, .k = x_ptr, .v = x_ptr,
            .q_pts = x_ptr, .k_pts = x_ptr, .v_pts = x_ptr};
    }
    kutacc_af2_ipa_o_inputs_t *o_ptrs = (kutacc_af2_ipa_o_inputs_t*)malloc(sizeof(kutacc_af2_ipa_o_inputs_t));
    if (o_ptrs != nullptr) {
        *o_ptrs = (kutacc_af2_ipa_o_inputs_t){.o =  x_ptr, .o_pt = x_ptr, .o_pt_norm =  x_ptr,
            .o_pair = x_ptr};
    }
    kutacc_af2_invariant_point(s_ptrs, o_ptrs, data_ptr, data_ptr, data_ptr, data_ptr, nullptr);
    std::cerr << "invariant_point_attention 7th param ipa_weights_ptrs is nullptr" << std::endl;
}
