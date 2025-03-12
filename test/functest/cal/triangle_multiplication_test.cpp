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

TEST(TriangleMultiplicationTest, calc_proj_nullptr_1)
{
    float a = 0.8f;
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
    kutacc_af2_tm_proj_weights_t *weight_ptr = (kutacc_af2_tm_proj_weights_t*)malloc(sizeof(kutacc_af2_tm_proj_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_tm_proj_weights_t){.c_o =2, .c_i = 2, .proj_w = x_ptr, .proj_b = x_ptr,
            .gate_w = x_ptr, .gate_b = x_ptr};
    }
    kutacc_af2_triangle_multiplication_calc_proj(nullptr, a_ptr, weight_ptr, true);
    std::cerr << "triangle_multiplication_calc_proj 1st param tm_acts_ptr is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, calc_proj_nullptr_2)
{
    float a = 0.8f;
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
    kutacc_af2_tm_act_inputs_t *act_ptr = (kutacc_af2_tm_act_inputs_t*)malloc(sizeof(kutacc_af2_tm_act_inputs_t));
    if (act_ptr != nullptr) {
        *act_ptr = (kutacc_af2_tm_act_inputs_t){.n_res = 2, .n_res_gather = 2, .proj_act = x_ptr,
            .input_act = x_ptr, .proj_act_gate = x_ptr};
    }
    kutacc_af2_tm_proj_weights_t *weight_ptr = (kutacc_af2_tm_proj_weights_t*)malloc(sizeof(kutacc_af2_tm_proj_weights_t));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_tm_proj_weights_t){.c_o =2, .c_i = 2, .proj_w = x_ptr, .proj_b = x_ptr,
            .gate_w = x_ptr, .gate_b = x_ptr};
    }
    kutacc_af2_triangle_multiplication_calc_proj(act_ptr, nullptr, weight_ptr, true);
    std::cerr << "triangle_multiplication_calc_proj 2nd param mask is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, calc_proj_nullptr_3)
{
    float a = 0.8f;
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
    kutacc_af2_tm_act_inputs_t *act_ptr = (kutacc_af2_tm_act_inputs_t*)malloc(sizeof(kutacc_af2_tm_act_inputs_t));
    if (act_ptr != nullptr) {
        *act_ptr = (kutacc_af2_tm_act_inputs_t){.n_res = 2, .n_res_gather = 2, .proj_act = x_ptr,
            .input_act = x_ptr, .proj_act_gate = x_ptr};
    }
    kutacc_af2_triangle_multiplication_calc_proj(act_ptr, a_ptr, nullptr, true);
    std::cerr << "triangle_multiplication_calc_proj 3rd param tm_weights_ptr is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, equation_nullptr_1)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;
    int64_t n_res_gather = 5;

    kutacc_af2_triangle_multiplication_equation(nullptr, a_ptr, a_ptr, n_res_gather, true);
    std::cerr << "triangle_multiplication_equation 1st param center_act is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, equation_nullptr_2)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;
    int64_t n_res_gather = 5;

    kutacc_af2_triangle_multiplication_equation(a_ptr, nullptr, a_ptr, n_res_gather, true);
    std::cerr << "triangle_multiplication_equation 2nd param left_proj_act is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, equation_nullptr_3)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;
    int64_t n_res_gather = 5;

    kutacc_af2_triangle_multiplication_equation(a_ptr, a_ptr, nullptr, n_res_gather, true);
    std::cerr << "triangle_multiplication_equation 3rd param right_proj_act is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, equation_nullptr_negative_int)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;
    int64_t n_res_gather = -5;
    
    kutacc_af2_triangle_multiplication_equation(a_ptr, a_ptr, a_ptr, n_res_gather, true);
    std::cerr << "triangle_multiplication_equation 4th param n_res_gather is negative" << std::endl;
}

TEST(TriangleMultiplicationTest, gate_and_out_linear_nullptr_1)
{
    float a = 0.8f;
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
    kutacc_af2_tm_act_inputs_t *act_ptr = (kutacc_af2_tm_act_inputs_t*)malloc(sizeof(kutacc_af2_tm_act_inputs_t));
    if (act_ptr != nullptr) {
        *act_ptr = (kutacc_af2_tm_act_inputs_t){.n_res = 2, .n_res_gather = 2, .proj_act = x_ptr,
            .input_act = x_ptr, .proj_act_gate = x_ptr};
    }
    kutacc_af2_tm_linear_weights *weight_ptr = (kutacc_af2_tm_linear_weights*)malloc(sizeof(kutacc_af2_tm_linear_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_tm_linear_weights){.c_o = 2, .c_i = 2, .gating_w = x_ptr,
            .gating_b = x_ptr, .output_proj_w = x_ptr, .output_proj_b = x_ptr};
    }
    kutacc_af2_triangle_multiplication_gate_and_out_linear(nullptr, a_ptr, act_ptr, a_ptr, weight_ptr, true);
    std::cerr << "triangle_multiplication_gate_and_out_linear 1st param gate is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, gate_and_out_linear_nullptr_2)
{
    float a = 0.8f;
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
    
    
    kutacc_af2_tm_act_inputs_t *act_ptr = (kutacc_af2_tm_act_inputs_t*)malloc(sizeof(kutacc_af2_tm_act_inputs_t));
    if (act_ptr != nullptr) {
        *act_ptr = (kutacc_af2_tm_act_inputs_t){.n_res = 2, .n_res_gather = 2, .proj_act = x_ptr,
            .input_act = x_ptr, .proj_act_gate = x_ptr};
    }
    kutacc_af2_tm_linear_weights *weight_ptr = (kutacc_af2_tm_linear_weights*)malloc(sizeof(kutacc_af2_tm_linear_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_tm_linear_weights){.c_o = 2, .c_i = 2, .gating_w = x_ptr,
            .gating_b = x_ptr, .output_proj_w = x_ptr, .output_proj_b = x_ptr};
    }
    kutacc_af2_triangle_multiplication_gate_and_out_linear(a_ptr, nullptr, act_ptr, a_ptr, weight_ptr, true);
    std::cerr << "triangle_multiplication_gate_and_out_linear 2nd param out is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, gate_and_out_linear_nullptr_3)
{
    float a = 0.8f;
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
    kutacc_af2_tm_linear_weights *weight_ptr = (kutacc_af2_tm_linear_weights*)malloc(sizeof(kutacc_af2_tm_linear_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_tm_linear_weights){.c_o = 2, .c_i = 2, .gating_w = x_ptr,
            .gating_b = x_ptr, .output_proj_w = x_ptr, .output_proj_b = x_ptr};
    }
    kutacc_af2_triangle_multiplication_gate_and_out_linear(a_ptr, a_ptr, nullptr, a_ptr, weight_ptr, true);
    std::cerr << "triangle_multiplication_gate_and_out_linear 3rd param tm_acts_ptr is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, gate_and_out_linear_nullptr_4)
{
    float a = 0.8f;
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
    
    
    kutacc_af2_tm_act_inputs_t *act_ptr = (kutacc_af2_tm_act_inputs_t*)malloc(sizeof(kutacc_af2_tm_act_inputs_t));
    if (act_ptr != nullptr) {
        *act_ptr = (kutacc_af2_tm_act_inputs_t){.n_res = 2, .n_res_gather = 2, .proj_act = x_ptr,
            .input_act = x_ptr, .proj_act_gate = x_ptr};
    }
    kutacc_af2_tm_linear_weights *weight_ptr = (kutacc_af2_tm_linear_weights*)malloc(sizeof(kutacc_af2_tm_linear_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_tm_linear_weights){.c_o = 2, .c_i = 2, .gating_w = x_ptr,
            .gating_b = x_ptr, .output_proj_w = x_ptr, .output_proj_b = x_ptr};
    }
    kutacc_af2_triangle_multiplication_gate_and_out_linear(a_ptr, a_ptr, act_ptr, nullptr, weight_ptr, true);
    std::cerr << "triangle_multiplication_gate_and_out_linear 4th param center_act is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, gate_and_out_linear_nullptr_5)
{
    float a = 0.8f;
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
    
    
    kutacc_af2_tm_act_inputs_t *act_ptr = (kutacc_af2_tm_act_inputs_t*)malloc(sizeof(kutacc_af2_tm_act_inputs_t));
    if (act_ptr != nullptr) {
        *act_ptr = (kutacc_af2_tm_act_inputs_t){.n_res = 2, .n_res_gather = 2, .proj_act = x_ptr,
            .input_act = x_ptr, .proj_act_gate = x_ptr};
    }
    kutacc_af2_triangle_multiplication_gate_and_out_linear(a_ptr, a_ptr, act_ptr, a_ptr, nullptr, true);
    std::cerr << "triangle_multiplication_gate_and_out_linear 5th param tm_weights_ptr is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, last_nullptr_1)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;
    kutacc_af2_triangle_multiplication_last(nullptr, a_ptr, 5, 5, 5);
    std::cerr << "triangle_multiplication_last 1st param out is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, last_nullptr_2)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;
    kutacc_af2_triangle_multiplication_last(a_ptr, nullptr, 5, 5, 5);
    std::cerr << "triangle_multiplication_last 2nd param gate is nullptr" << std::endl;
}

TEST(TriangleMultiplicationTest, last_negative_int_1)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;
    kutacc_af2_triangle_multiplication_last(a_ptr, a_ptr, -5, 5, 5);
    std::cerr << "triangle_multiplication_last 3rd param n_res is negative" << std::endl;
}

TEST(TriangleMultiplicationTest, last_negative_int_2)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;
    kutacc_af2_triangle_multiplication_last(a_ptr, a_ptr, 5, -5, 5);
    std::cerr << "triangle_multiplication_last 4th param n_res_gather is negative" << std::endl;
}

TEST(TriangleMultiplicationTest, last_negative_int_3)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;
    kutacc_af2_triangle_multiplication_last(a_ptr, a_ptr, 5, 5, -5);
    std::cerr << "triangle_multiplication_last 5th param c_o is negative" << std::endl;
}
