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

TEST(OuterProductMeanTest, calc_left_and_right_mul_nullptr1)
{
    float a = 1.8f;
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

   
    kutacc_af2_opm_act_inputs *inputs_ptr = (kutacc_af2_opm_act_inputs*)malloc(sizeof(kutacc_af2_opm_act_inputs));
    if (inputs_ptr != nullptr) {
        *inputs_ptr = (kutacc_af2_opm_act_inputs){.n_seq = 2, .n_res = 2, .input_act = x_ptr, .left_proj = x_ptr,
            .right_proj = x_ptr, .left_proj_ = x_ptr, .right_proj_ = x_ptr};
    }
    kutacc_af2_opm_mask_inputs *mask_ptr = (kutacc_af2_opm_mask_inputs*)malloc(sizeof(kutacc_af2_opm_mask_inputs));
    if (mask_ptr != nullptr) {
        *mask_ptr = (kutacc_af2_opm_mask_inputs){.n_res_gather = 2, .mask_bias = 1, .mask = x_ptr, .norm = x_ptr};
    }
    kutacc_af2_opm_weights *weight_ptr = (kutacc_af2_opm_weights*)malloc(sizeof(kutacc_af2_opm_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_opm_weights){.c_m = 2, .c_i = 1, .c_z = 1, .left_proj_w = x_ptr, .left_proj_b = x_ptr,
            .right_proj_w = x_ptr, .right_proj_b = x_ptr, .outer_w = x_ptr, .outer_b = x_ptr};
    }
    kutacc_af2_outer_product_mean_calc_left_and_right_mul(nullptr, mask_ptr, weight_ptr);
    std::cerr << "outer_product_mean_calc_left_and_right_mul 1st param opm_acts_ptr is nullptr" << std::endl;
}

TEST(OuterProductMeanTest, calc_left_and_right_mul_nullptr2)
{
    float a = 1.8f;
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

   
    kutacc_af2_opm_act_inputs *inputs_ptr = (kutacc_af2_opm_act_inputs*)malloc(sizeof(kutacc_af2_opm_act_inputs));
    if (inputs_ptr != nullptr) {
        *inputs_ptr = (kutacc_af2_opm_act_inputs){.n_seq = 2, .n_res = 2, .input_act = x_ptr, .left_proj = x_ptr,
            .right_proj = x_ptr, .left_proj_ = x_ptr, .right_proj_ = x_ptr};
    }
    kutacc_af2_opm_mask_inputs *mask_ptr = (kutacc_af2_opm_mask_inputs*)malloc(sizeof(kutacc_af2_opm_mask_inputs));
    if (mask_ptr != nullptr) {
        *mask_ptr = (kutacc_af2_opm_mask_inputs){.n_res_gather = 2, .mask_bias = 1, .mask = x_ptr, .norm = x_ptr};
    }
    kutacc_af2_opm_weights *weight_ptr = (kutacc_af2_opm_weights*)malloc(sizeof(kutacc_af2_opm_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_opm_weights){.c_m = 2, .c_i = 1, .c_z = 1, .left_proj_w = x_ptr, .left_proj_b = x_ptr,
            .right_proj_w = x_ptr, .right_proj_b = x_ptr, .outer_w = x_ptr, .outer_b = x_ptr};
    }
    kutacc_af2_outer_product_mean_calc_left_and_right_mul(inputs_ptr, nullptr, weight_ptr);
    std::cerr << "outer_product_mean_calc_left_and_right_mul 2nd param opm_masks_ptr is nullptr" << std::endl;
}

TEST(OuterProductMeanTest, calc_left_and_right_mul_nullptr3)
{
    float a = 1.8f;
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

   
    kutacc_af2_opm_act_inputs *inputs_ptr = (kutacc_af2_opm_act_inputs*)malloc(sizeof(kutacc_af2_opm_act_inputs));
    if (inputs_ptr != nullptr) {
        *inputs_ptr = (kutacc_af2_opm_act_inputs){.n_seq = 2, .n_res = 2, .input_act = x_ptr, .left_proj = x_ptr,
            .right_proj = x_ptr, .left_proj_ = x_ptr, .right_proj_ = x_ptr};
    }
    kutacc_af2_opm_mask_inputs *mask_ptr = (kutacc_af2_opm_mask_inputs*)malloc(sizeof(kutacc_af2_opm_mask_inputs));
    if (mask_ptr != nullptr) {
        *mask_ptr = (kutacc_af2_opm_mask_inputs){.n_res_gather = 2, .mask_bias = 1, .mask = x_ptr, .norm = x_ptr};
    }
    kutacc_af2_opm_weights *weight_ptr = (kutacc_af2_opm_weights*)malloc(sizeof(kutacc_af2_opm_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_opm_weights){.c_m = 2, .c_i = 1, .c_z = 1, .left_proj_w = x_ptr, .left_proj_b = x_ptr,
            .right_proj_w = x_ptr, .right_proj_b = x_ptr, .outer_w = x_ptr, .outer_b = x_ptr};
    }
    kutacc_af2_outer_product_mean_calc_left_and_right_mul(inputs_ptr, mask_ptr, nullptr);
    std::cerr << "outer_product_mean_calc_left_and_right_mul 3rd param opm_weights_ptr is nullptr" << std::endl;
}

TEST(OuterProductMeanTest, chunk_nullptr_1)
{
    float a = 0.75f;
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

   
    kutacc_af2_opm_act_inputs *inputs_ptr = (kutacc_af2_opm_act_inputs*)malloc(sizeof(kutacc_af2_opm_act_inputs));
    if (inputs_ptr != nullptr) {
        *inputs_ptr = (kutacc_af2_opm_act_inputs){.n_seq = 2, .n_res = 2, .input_act = x_ptr, .left_proj = x_ptr,
            .right_proj = x_ptr, .left_proj_ = x_ptr, .right_proj_ = x_ptr};
    }
    kutacc_af2_opm_mask_inputs *mask_ptr = (kutacc_af2_opm_mask_inputs*)malloc(sizeof(kutacc_af2_opm_mask_inputs));
    if (mask_ptr != nullptr) {
        *mask_ptr = (kutacc_af2_opm_mask_inputs){.n_res_gather = 2, .mask_bias = 1, .mask = x_ptr, .norm = x_ptr};
    }
    kutacc_af2_opm_weights *weight_ptr = (kutacc_af2_opm_weights*)malloc(sizeof(kutacc_af2_opm_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_opm_weights){.c_m = 2, .c_i = 1, .c_z = 1, .left_proj_w = x_ptr, .left_proj_b = x_ptr,
            .right_proj_w = x_ptr, .right_proj_b = x_ptr, .outer_w = x_ptr, .outer_b = x_ptr};
    }
    kutacc_af2_outer_product_mean_chunk(nullptr, mask_ptr, weight_ptr, a_ptr, 2, 2);
    std::cerr << "outer_product_mean_chunk 1st param opm_acts_ptr is nullptr" << std::endl;
}

TEST(OuterProductMeanTest, chunk_nullptr_2)
{
    float a = 0.75f;
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

   
    kutacc_af2_opm_act_inputs *inputs_ptr = (kutacc_af2_opm_act_inputs*)malloc(sizeof(kutacc_af2_opm_act_inputs));
    if (inputs_ptr != nullptr) {
        *inputs_ptr = (kutacc_af2_opm_act_inputs){.n_seq = 2, .n_res = 2, .input_act = x_ptr, .left_proj = x_ptr,
            .right_proj = x_ptr, .left_proj_ = x_ptr, .right_proj_ = x_ptr};
    }
    kutacc_af2_opm_mask_inputs *mask_ptr = (kutacc_af2_opm_mask_inputs*)malloc(sizeof(kutacc_af2_opm_mask_inputs));
    if (mask_ptr != nullptr) {
        *mask_ptr = (kutacc_af2_opm_mask_inputs){.n_res_gather = 2, .mask_bias = 1, .mask = x_ptr, .norm = x_ptr};
    }
    kutacc_af2_opm_weights *weight_ptr = (kutacc_af2_opm_weights*)malloc(sizeof(kutacc_af2_opm_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_opm_weights){.c_m = 2, .c_i = 1, .c_z = 1, .left_proj_w = x_ptr, .left_proj_b = x_ptr,
            .right_proj_w = x_ptr, .right_proj_b = x_ptr, .outer_w = x_ptr, .outer_b = x_ptr};
    }
    kutacc_af2_outer_product_mean_chunk(inputs_ptr, nullptr, weight_ptr, a_ptr, 2, 2);
    std::cerr << "outer_product_mean_chunk 2nd param opm_masks_ptr is nullptr" << std::endl;
}

TEST(OuterProductMeanTest, chunk_nullptr_3)
{
    float a = 0.75f;
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

   
    kutacc_af2_opm_act_inputs *inputs_ptr = (kutacc_af2_opm_act_inputs*)malloc(sizeof(kutacc_af2_opm_act_inputs));
    if (inputs_ptr != nullptr) {
        *inputs_ptr = (kutacc_af2_opm_act_inputs){.n_seq = 2, .n_res = 2, .input_act = x_ptr, .left_proj = x_ptr,
            .right_proj = x_ptr, .left_proj_ = x_ptr, .right_proj_ = x_ptr};
    }
    kutacc_af2_opm_mask_inputs *mask_ptr = (kutacc_af2_opm_mask_inputs*)malloc(sizeof(kutacc_af2_opm_mask_inputs));
    if (mask_ptr != nullptr) {
        *mask_ptr = (kutacc_af2_opm_mask_inputs){.n_res_gather = 2, .mask_bias = 1, .mask = x_ptr, .norm = x_ptr};
    }
    kutacc_af2_opm_weights *weight_ptr = (kutacc_af2_opm_weights*)malloc(sizeof(kutacc_af2_opm_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_opm_weights){.c_m = 2, .c_i = 1, .c_z = 1, .left_proj_w = x_ptr, .left_proj_b = x_ptr,
            .right_proj_w = x_ptr, .right_proj_b = x_ptr, .outer_w = x_ptr, .outer_b = x_ptr};
    }
    kutacc_af2_outer_product_mean_chunk(inputs_ptr, mask_ptr, nullptr, a_ptr, 2, 2);
    std::cerr << "outer_product_mean_chunk 3rd param opm_weights_ptr is nullptr" << std::endl;
}

TEST(OuterProductMeanTest, chunk_nullptr_4)
{
    float a = 0.75f;
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

   
    kutacc_af2_opm_act_inputs *inputs_ptr = (kutacc_af2_opm_act_inputs*)malloc(sizeof(kutacc_af2_opm_act_inputs));
    if (inputs_ptr != nullptr) {
        *inputs_ptr = (kutacc_af2_opm_act_inputs){.n_seq = 2, .n_res = 2, .input_act = x_ptr, .left_proj = x_ptr,
            .right_proj = x_ptr, .left_proj_ = x_ptr, .right_proj_ = x_ptr};
    }
    kutacc_af2_opm_mask_inputs *mask_ptr = (kutacc_af2_opm_mask_inputs*)malloc(sizeof(kutacc_af2_opm_mask_inputs));
    if (mask_ptr != nullptr) {
        *mask_ptr = (kutacc_af2_opm_mask_inputs){.n_res_gather = 2, .mask_bias = 1, .mask = x_ptr, .norm = x_ptr};
    }
    kutacc_af2_opm_weights *weight_ptr = (kutacc_af2_opm_weights*)malloc(sizeof(kutacc_af2_opm_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_opm_weights){.c_m = 2, .c_i = 1, .c_z = 1, .left_proj_w = x_ptr, .left_proj_b = x_ptr,
            .right_proj_w = x_ptr, .right_proj_b = x_ptr, .outer_w = x_ptr, .outer_b = x_ptr};
    }
    kutacc_af2_outer_product_mean_chunk(inputs_ptr, mask_ptr, weight_ptr, nullptr, 2, 2);
    std::cerr << "outer_product_mean_chunk 4th param out is nullptr" << std::endl;
}

TEST(OuterProductMeanTest, chunk_negative_int_1)
{
    float a = 0.75f;
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

   
    kutacc_af2_opm_act_inputs *inputs_ptr = (kutacc_af2_opm_act_inputs*)malloc(sizeof(kutacc_af2_opm_act_inputs));
    if (inputs_ptr != nullptr) {
        *inputs_ptr = (kutacc_af2_opm_act_inputs){.n_seq = 2, .n_res = 2, .input_act = x_ptr, .left_proj = x_ptr,
            .right_proj = x_ptr, .left_proj_ = x_ptr, .right_proj_ = x_ptr};
    }
    kutacc_af2_opm_mask_inputs *mask_ptr = (kutacc_af2_opm_mask_inputs*)malloc(sizeof(kutacc_af2_opm_mask_inputs));
    if (mask_ptr != nullptr) {
        *mask_ptr = (kutacc_af2_opm_mask_inputs){.n_res_gather = 2, .mask_bias = 1, .mask = x_ptr, .norm = x_ptr};
    }
    kutacc_af2_opm_weights *weight_ptr = (kutacc_af2_opm_weights*)malloc(sizeof(kutacc_af2_opm_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_opm_weights){.c_m = 2, .c_i = 1, .c_z = 1, .left_proj_w = x_ptr, .left_proj_b = x_ptr,
            .right_proj_w = x_ptr, .right_proj_b = x_ptr, .outer_w = x_ptr, .outer_b = x_ptr};
    }
    kutacc_af2_outer_product_mean_chunk(inputs_ptr, mask_ptr, weight_ptr, a_ptr, -10, 2);
    std::cerr << "outer_product_mean_chunk 5th param left_block_size is nullptr" << std::endl;
}

TEST(OuterProductMeanTest, chunk_negative_int_2)
{
    float a = 0.75f;
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

   
    kutacc_af2_opm_act_inputs *inputs_ptr = (kutacc_af2_opm_act_inputs*)malloc(sizeof(kutacc_af2_opm_act_inputs));
    if (inputs_ptr != nullptr) {
        *inputs_ptr = (kutacc_af2_opm_act_inputs){.n_seq = 2, .n_res = 2, .input_act = x_ptr, .left_proj = x_ptr,
            .right_proj = x_ptr, .left_proj_ = x_ptr, .right_proj_ = x_ptr};
    }
    kutacc_af2_opm_mask_inputs *mask_ptr = (kutacc_af2_opm_mask_inputs*)malloc(sizeof(kutacc_af2_opm_mask_inputs));
    if (mask_ptr != nullptr) {
        *mask_ptr = (kutacc_af2_opm_mask_inputs){.n_res_gather = 2, .mask_bias = 1, .mask = x_ptr, .norm = x_ptr};
    }
    kutacc_af2_opm_weights *weight_ptr = (kutacc_af2_opm_weights*)malloc(sizeof(kutacc_af2_opm_weights));
    if (weight_ptr != nullptr) {
        *weight_ptr = (kutacc_af2_opm_weights){.c_m = 2, .c_i = 1, .c_z = 1, .left_proj_w = x_ptr, .left_proj_b = x_ptr,
            .right_proj_w = x_ptr, .right_proj_b = x_ptr, .outer_w = x_ptr, .outer_b = x_ptr};
    }
    kutacc_af2_outer_product_mean_chunk(inputs_ptr, mask_ptr, weight_ptr, a_ptr, 1, -10);
    std::cerr << "outer_product_mean_chunk 6th param right_block_size is nullptr" << std::endl;
}
