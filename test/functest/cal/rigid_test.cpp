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

TEST(RigidTest, rot_vec_mul_nullptr_1)
{
    float a = 0.75f;
    void *a_ptr = (void*)&a; 
    kutacc_af2_rigid_rot_vec_mul(nullptr, a_ptr, a_ptr, a_ptr);
    std::cerr << "rigid_rot_vec_mul 1st param pts is nullptr" << std::endl;
}

TEST(RigidTest, rot_vec_mul_nullptr_2)
{
    float a = 0.75f;
    void *a_ptr = (void*)&a; 
    kutacc_af2_rigid_rot_vec_mul(a_ptr, nullptr, a_ptr, a_ptr);
    std::cerr << "rigid_rot_vec_mul 2nd param rot_mats is nullptr" << std::endl;
}

TEST(RigidTest, rot_vec_mul_nullptr_3)
{
    float a = 0.75f;
    void *a_ptr = (void*)&a; 
    kutacc_af2_rigid_rot_vec_mul(a_ptr, a_ptr, nullptr, a_ptr);
    std::cerr << "rigid_rot_vec_mul 3rd param out is nullptr" << std::endl;
}

TEST(RigidTest, rot_matmul_nullptr_1)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;

    kutacc_af2_rigid_rot_matmul(nullptr, a_ptr, a_ptr);
    std::cerr << "rot_matmul 1st param a is nullptr" << std::endl;
}

TEST(RigidTest, rot_matmul_nullptr_2)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;

    kutacc_af2_rigid_rot_matmul(a_ptr, nullptr, a_ptr);
    std::cerr << "rot_matmul 2nd param b is nullptr" << std::endl;
}

TEST(RigidTest, rot_matmul_nullptr_3)
{
    float a = 0.8f;
    void *a_ptr = (void*)&a;

    kutacc_af2_rigid_rot_matmul(a_ptr, a_ptr, nullptr);
    std::cerr << "rot_matmul 3rd param out is nullptr" << std::endl;
}
