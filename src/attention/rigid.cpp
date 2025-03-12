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

#include "rigid.h"
#include "tensor/tensor.h"
#include "utils/parallel.h"
#include "kutacc.h"
#include "utils/collapse.h"

namespace kutacc {
template <typename scalar_t>
void rigid_rot_vec_mul(Tensor &pts, Tensor &rot_mats, Tensor &out, kutacc_tensor_h trans)
{
    parallel_for(0, pts.numel() / 3, 1024, [&](int64_t start, int64_t end) {
        if (pts.dim() == 2) {
            collapse_for(start, end, pts.sizes()[0], [&](int64_t i0) {
                auto pts_data = (scalar_t *)pts.data_ptr() + pts.strides()[0] * i0;
                auto rot_mats_data = 
                (float *)rot_mats.data_ptr() + rot_mats.strides()[0] * (rot_mats.sizes()[0] != 1 ? i0 : 0);
                auto out_data = (scalar_t *)out.data_ptr() + out.strides()[0] * i0;
                float* trans_data = nullptr;
                if (trans != nullptr) {
                    auto trans_t = *convertKutaccTensor(trans);
                    trans_data = (float *)trans_t.data_ptr() + trans_t.strides()[0] * 
                                        (trans_t.sizes()[0] != 1 ? i0 : 0);
                }
                rigid_rot_vec_mul_kernel(pts_data, rot_mats_data, out_data, trans_data);
        });
        } else if (pts.dim() == 3) {
            collapse_for(start, end, pts.sizes()[0], pts.sizes()[1], [&](int64_t i0, int64_t i1) {
                auto pts_data = (scalar_t *)pts.data_ptr() + pts.strides()[0] * i0 + pts.strides()[1] * i1;
                auto rot_mats_data = (float *)rot_mats.data_ptr() +
                                        rot_mats.strides()[0] * (rot_mats.sizes()[0] != 1 ? i0 : 0) +
                                        rot_mats.strides()[1] * (rot_mats.sizes()[1] != 1 ? i1 : 0);
                auto out_data = (scalar_t *)out.data_ptr() + out.strides()[0] * i0 +
                                    out.strides()[1] * i1;
                float* trans_data = nullptr;
                if (trans != nullptr) {
                    auto trans_t = *convertKutaccTensor(trans);
                    trans_data = (float *)trans_t.data_ptr() + trans_t.strides()[0] * 
                                        (trans_t.sizes()[0] != 1 ? i0 : 0) +
                                        trans_t.strides()[1] * (trans_t.sizes()[1] != 1 ? i1 : 0);
                }
                rigid_rot_vec_mul_kernel(pts_data, rot_mats_data, out_data, trans_data);
            });
        } else if (pts.dim() == 4) {
            collapse_for(start, end, pts.sizes()[0], pts.sizes()[1], pts.sizes()[2],
                [&](int64_t i0, int64_t i1, int64_t i2) {
                auto pts_data = (scalar_t *)pts.data_ptr() + pts.strides()[0] * i0 + pts.strides()[1] * i1 +
                                    pts.strides()[2] * i2;
                auto rot_mats_data = (float *)rot_mats.data_ptr() +
                                        rot_mats.strides()[0] * (rot_mats.sizes()[0] != 1 ? i0 : 0) +
                                        rot_mats.strides()[1] * (rot_mats.sizes()[1] != 1 ? i1 : 0) +
                                        rot_mats.strides()[2] * (rot_mats.sizes()[2] != 1 ? i2 : 0);
                auto out_data = (scalar_t *)out.data_ptr() + out.strides()[0] * i0 +
                                    out.strides()[1] * i1 + out.strides()[2] * i2;
                float* trans_data = nullptr;
                if (trans != nullptr) {
                    auto trans_t = *convertKutaccTensor(trans);
                    trans_data = (float *)trans_t.data_ptr() + trans_t.strides()[0] * 
                                        (trans_t.sizes()[0] != 1 ? i0 : 0) +
                                        trans_t.strides()[1] * (trans_t.sizes()[1] != 1 ? i1 : 0) +
                                        trans_t.strides()[2] * (trans_t.sizes()[2] != 1 ? i2 : 0);
                }
                rigid_rot_vec_mul_kernel(pts_data, rot_mats_data, out_data, trans_data);
            });
        } else {
            KUTACC_CHECK(false, "not implemented");
        }
    });
}

void rigid_rot_matmul(Tensor &a, Tensor &b, Tensor &out)
{
    parallel_for(0, b.numel() / 9, 1024, [&](int64_t start, int64_t end) {
        if (a.dim() == 3) {
            collapse_for(start, end, [&](int64_t i0){
                auto a_data = (float *)a.data_ptr() + a.strides()[0] * (a.sizes()[0] != 1 ? i0 : 0);
                auto b_data = (float *)b.data_ptr() + b.strides()[0] * i0;
                auto out_data = (float *)out.data_ptr() + out.strides()[0] * i0;
                rigid_rot_matmul_kernel(a_data, b_data, out_data);
            });
        } else if (a.dim() == 4) {
            collapse_for(start, end, a.sizes()[0], a.sizes()[1], [&](int64_t i0, int64_t i1) {
                auto a_data = (float *)a.data_ptr() + a.strides()[0] * (a.sizes()[0] != 1 ? i0 : 0) +
                                a.strides()[1] * (a.sizes()[1] != 1 ? i1 : 0);
                auto b_data = (float *)b.data_ptr() + b.strides()[0] * i0 + b.strides()[1] * i1;
                auto out_data = (float *)out.data_ptr() + out.strides()[0] * i0 + out.strides()[1] * i1;
                rigid_rot_matmul_kernel(a_data, b_data, out_data);
            });
        } else {
            KUTACC_CHECK(false, "not implemented");
        }
    });
}
}

void kutacc_af2_rigid_rot_vec_mul(kutacc_tensor_h pts, kutacc_tensor_h rot_mats, kutacc_tensor_h out, kutacc_tensor_h trans)
{
    KUTACC_CHECK(out != nullptr && pts != nullptr && rot_mats != nullptr, "kutacc_af2_rigid_rot_vec_mul: input args nullptr error");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    }
    if ((*kutacc::convertKutaccTensor(pts)).dtype() == kutacc::kBF16) {
        kutacc::rigid_rot_vec_mul<__bf16>(*kutacc::convertKutaccTensor(pts), *kutacc::convertKutaccTensor(rot_mats), *kutacc::convertKutaccTensor(out), trans);
    }
}

void kutacc_af2_rigid_rot_matmul(kutacc_tensor_h a, kutacc_tensor_h b, kutacc_tensor_h out)
{
    KUTACC_CHECK(a != nullptr && b != nullptr && out != nullptr, "kutacc_af2_rigid_rot_matmul: input args nullptr error");
    if (kutacc::kutacc_check_err_set == true) {
        return;
    } 
    kutacc::rigid_rot_matmul(*kutacc::convertKutaccTensor(a), *kutacc::convertKutaccTensor(b), *kutacc::convertKutaccTensor(out));
}