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

#ifndef KUTACC_H
#define KUTACC_H

#include <vector>
#include <cstdint>
#include <optional>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief exposed KuTACC symbol table*/
#define kutacc_export __attribute__((visibility("default")))

/** @brief status in KuTACC library*/
#define KUTACC_OK 0
#define KUTACC_ERROR (-1)

/** @brief KuTACC version info*/
typedef struct kutacc_version {
    const char *product_name;
    const char *product_version;
    const char *component_name;
    const char *component_version;
    const char *component_abi_version;
    const char *component_appendinfo;
} kutacc_version_t;

/*
 * @brief get the kutacc version info
 * @param [out] version     the kutacc version info
 *
 * @return KUTACC_OK for get version info success, other for failed.
 */
kutacc_export int kutacc_get_version(kutacc_version_t *version);

typedef void *kutacc_tensor_h;

kutacc_export extern int64_t world_size;
kutacc_export extern int64_t rank;
extern int64_t buffer_size;

kutacc_export void kutacc_comm_init(int64_t _world_size, int64_t _rank, int64_t _buffer_size);
kutacc_export void kutacc_comm_fini();

/**
 * @brief transpose
 *   chunk(x) = min(block(x), x - rank * block(x))
 *   block(x) = ceil(x / world_size)
 * @param data shape [chunk(m), n, len] or [m, chunk(n), len]
 * @param out shape [m, chunk(n), len] or [chunk(m), n, len]
 */
kutacc_export void kutacc_af2_transpose(kutacc_tensor_h data, kutacc_tensor_h out);

/**
 * @brief all-gather
 *   chunk(x) = min(block(x), x - rank * block(x))
 *   block(x) = ceil(x / world_size)
 * @param data shape [chunk(m), n, len] or [m, chunk(n), len]
 * @param out shape [m, chunk(n), len] or [chunk(m), n, len]
 */
kutacc_export void kutacc_af2_all_gather(kutacc_tensor_h data, kutacc_tensor_h out);

/** @brief used for gating_attention and global_attention op, it's a union of weights params */
typedef struct kutacc_af2_attention_weights {
    int64_t nchannels;
    int64_t nheads;
    int64_t head_size;

    kutacc_tensor_h query_w;
    kutacc_tensor_h key_w;
    kutacc_tensor_h value_w;
    kutacc_tensor_h gating_w;
    kutacc_tensor_h gating_b;
    kutacc_tensor_h output_w;
    kutacc_tensor_h output_b;
} kutacc_af2_attention_weights_t;

/** @brief used for gating_attention and global_attention op,
    it's a union of tensors which shapes like act param in alphafold.py
    and be used as input variables of the intermediate process
 */
typedef struct kutacc_af2_attention_inputs {
    int64_t batch;
    int64_t seq_len;

    kutacc_tensor_h q;
    kutacc_tensor_h k;
    kutacc_tensor_h v;
    kutacc_tensor_h gate;
    kutacc_tensor_h avg;
} kutacc_af2_attention_inputs_t;

/** @brief used for invariant point attention op,
    it's a union of weights params needed by ipa
 */
typedef struct kutacc_af2_ipa_weights {
    int64_t c_z;
    int64_t c_hidden;
    int64_t no_heads;
    int64_t no_qk_points;
    int64_t no_v_points;

    kutacc_tensor_h head_weights;
    kutacc_tensor_h weights_head_weights;
    kutacc_tensor_h linear_b_w;
    kutacc_tensor_h linear_b_b;
} kutacc_af2_ipa_weights_t;

/** @brief used for ipa op,
    it's a union of tensors which shapes like s param in alphafold.py
    and be used as input variables of the intermediate process
 */
typedef struct kutacc_af2_ipa_s_inputs {
    int64_t n_res;

    kutacc_tensor_h a;
    kutacc_tensor_h b;
    kutacc_tensor_h q;
    kutacc_tensor_h k;
    kutacc_tensor_h v;
    kutacc_tensor_h q_pts;
    kutacc_tensor_h k_pts;
    kutacc_tensor_h v_pts;
} kutacc_af2_ipa_s_inputs_t;

/** @brief used for ipa op,
    it's a union of tensors which shapes like o param from kpex
    and be used as input variables of the intermediate process
 */
typedef struct kutacc_af2_ipa_o_inputs {
    kutacc_tensor_h o;
    kutacc_tensor_h o_pt;
    kutacc_tensor_h o_pt_norm;
    kutacc_tensor_h o_pair;
} kutacc_af2_ipa_o_inputs_t;

/** @brief used for triangle multiplication op
    it's a union of tensors and be used as weights
    for op's intermediate process: calc left and right projection
 */
typedef struct kutacc_af2_tm_proj_weights {
    int64_t c_o;
    int64_t c_i;

    kutacc_tensor_h proj_w;
    kutacc_tensor_h proj_b;
    kutacc_tensor_h gate_w;
    kutacc_tensor_h gate_b;
} kutacc_af2_tm_proj_weights_t;

/** @brief used for triangle multiplication op
    it's a union of tensors which shapes like act from alphafold.py
    proj act is the input and also the output of calc left and right projection
 */
typedef struct kutacc_af2_tm_act_inputs {
    int64_t n_res;
    int64_t n_res_gather;

    kutacc_tensor_h proj_act;
    kutacc_tensor_h input_act;
    kutacc_tensor_h proj_act_gate;
} kutacc_af2_tm_act_inputs_t;

/** @brief used for triangle multiplication op
    it's a union of tensors and be used as weights
    for op's intermediate process: calc gate value and out value
 */
typedef struct kutacc_af2_tm_linear_weights {
    int64_t c_o;
    int64_t c_i;

    kutacc_tensor_h gating_w;
    kutacc_tensor_h gating_b;
    kutacc_tensor_h output_proj_w;
    kutacc_tensor_h output_proj_b;
} kutacc_af2_tm_linear_weights_t;

/** @brief used for transition op
    it's a union of tensors and be used as weights of transition op
 */
typedef struct kutacc_af2_trans_weights {
    int64_t c_o;
    int64_t c_i;

    kutacc_tensor_h linear1_w;
    kutacc_tensor_h linear1_b;
    kutacc_tensor_h linear2_w;
    kutacc_tensor_h linear2_b;
} kutacc_af2_trans_weights_t;

/** @brief used for transition op
    it's a union of tensors which shape like act from alphafold.py
    and be used as inputs of transition op
 */
typedef struct kutacc_af2_trans_act_inputs {
    int64_t batch;
    int64_t n_res;

    kutacc_tensor_h input_act;
    kutacc_tensor_h intermediate_act;
} kutacc_af2_trans_act_inputs_t;

/** @brief used for outer product mean op
    it's a union of tensors and be used as weights of outer product mean op
 */
typedef struct kutacc_af2_opm_weights {
    int64_t c_m;
    int64_t c_i;
    int64_t c_z;

    kutacc_tensor_h left_proj_w;
    kutacc_tensor_h left_proj_b;
    kutacc_tensor_h right_proj_w;
    kutacc_tensor_h right_proj_b;
    kutacc_tensor_h outer_w;
    kutacc_tensor_h outer_b;
} kutacc_af2_opm_weights_t;

/** @brief used for outer product mean op
    it's a union of tensors which shape like act from alphafold.py
    and be used as inputs of outer product mean op
 */
typedef struct kutacc_af2_opm_act_inputs {
    int64_t n_seq;
    int64_t n_res;

    kutacc_tensor_h input_act;
    kutacc_tensor_h left_proj;
    kutacc_tensor_h right_proj;
    kutacc_tensor_h left_proj_;
    kutacc_tensor_h right_proj_;
} kutacc_af2_opm_act_inputs_t;

/** @brief used for outer product mean op
    it's a union of tensors which shape like mask from alphafold.py
    and be used as mask inputs of outer product mean op
 */
typedef struct kutacc_af2_opm_mask_inputs {
    int64_t n_res_gather;
    int64_t mask_bias;

    kutacc_tensor_h mask;
    kutacc_tensor_h norm;
} kutacc_af2_opm_mask_inputs_t;

/**
 * @brief outer_product_mean_calc_left_and_right_mu algorithm
 * @param [out] left_proj, right_proj, left_proj_, right_proj_, mask
 * @param [in] left_proj_w, left_proj_b, right_proj_w, right_proj_b
 * @param [in] c_i, c_m, n_res, n_res_gather, n_seq, mask_bias
 * @return Null
 */
kutacc_export void kutacc_af2_outer_product_mean_calc_left_and_right_mul(kutacc_af2_opm_act_inputs_t *opm_acts_ptr,
                                                                         kutacc_af2_opm_mask_inputs_t *opm_masks_ptr,
                                                                         kutacc_af2_opm_weights_t *opm_weights_ptr);

/**
 * @brief outer_product_mean_chunk algorithm
 * @param [out] output_b, output_w out
 * @param [in] left_proj_, right_proj_, norm, left_block_size, right_block_size,
 * @param [in] c_i, c_z, n_res, n_res_gather, n_seq
 * @return Null
 */
kutacc_export void kutacc_af2_outer_product_mean_chunk(kutacc_af2_opm_act_inputs_t *opm_acts_ptr,
                                                       kutacc_af2_opm_mask_inputs_t *opm_masks_ptr,
                                                       kutacc_af2_opm_weights_t *opm_weights_ptr, kutacc_tensor_h out,
                                                       int64_t left_block_size, int64_t right_block_size);

/**
 * @brief gating_attention algorithm
 * @param input prepacked q_data, shape[batch * seq_len, nchannels]
 * @param q q_linear, shape[batch, seq_len, nheads, head_size]
 * @param k k_linear, shape[batch, seq_len, nheads, head_size]
 * @param v v_linear, shape[batch, seq_len, nheads, head_size]
 * @param gate gate_linear, shape[batch, seq_len, nheads, head_size]
 * @param weighted_avg shape[batch, seq_len, nheads, head_size]
 * @param [in] bias
 * @param [in] nobatched_bias
 * @param [in] GatingAttentionWeight
 * @param [out] out
 * @return Null
 * constraint: nchannels = nheads * head_size
 */
kutacc_export void kutacc_af2_gating_attention(kutacc_tensor_h input, kutacc_af2_attention_inputs_t *q_based_ptr,
                                               kutacc_tensor_h bias, kutacc_tensor_h nonbatched_bias,
                                               kutacc_af2_attention_weights_t *weight_ptr, kutacc_tensor_h out,
                                               int64_t block_size);
/**
 * @param q_data shape [batch, seq_len, nchannels], bf16
 * @param q_mask shape [batch, seq_len, 1], bf16
 */
kutacc_export void kutacc_af2_global_attention(kutacc_af2_attention_inputs_t *q_based_ptr, kutacc_tensor_h q_data,
                                               kutacc_tensor_h q_mask, kutacc_af2_attention_weights_t *weight_ptr,
                                               kutacc_tensor_h out);

/**
 * @brief transition algorithm
 * @param [in] input_act
 * @param [in] linear1_w
 * @param [in] linear1_b
 * @param [in] linear2_w
 * @param [in] linear2_b
 * @param [in] intermediate_act
 * @param [in] batch, n_res, c_o, c_i
 * @param [out] out
 * @return Null
 */
kutacc_export void kutacc_af2_transition(kutacc_af2_trans_act_inputs_t *trans_inputs_ptr,
                                         kutacc_af2_trans_weights_t *trans_weights_ptr, kutacc_tensor_h out);

/**
 * @brief af2_layernorm algorithm: layernorm interface for af2 model
 * @param [in] data
 * @param [in] gamma
 * @param [in] beta
 * @param [in] size
 * @param [in] eps
 * @param [out] out
 * @return Null
 */
kutacc_export void kutacc_af2_layernorm(__bf16 *data, float *gamma, float *beta, int64_t size, float eps, __bf16 *out);

/**
 * @brief af2_invariant_point algorithm: invariant_point interface for af2 model
 * @param [in] q
 * @param [in] k
 * @param [in] v
 * @param [in] q_pts
 * @param [in] k_pts
 * @param [in] v_pts
 * @param [in] b
 * @param [in] a
 * @param [in] head_weights
 * @param [in] weights.head_weights
 * @param [in] z
 * @param [in] rigid_rot_mats
 * @param [in] rigid_trans
 * @param [in] mask
 * @param [in] linear_b_w
 * @param [in] linear_b_b
 * @param [in] n_res
 * @param [in] c_z
 * @param [in] c_hidden
 * @param [in] no_heads
 * @param [in] no_qk_points
 * @param [in] no_v_points
 * @param [out] o
 * @param [out] o_pt
 * @param [out] o_pt_norm
 * @param [out] o_pair
 * @param [out] out
 * @return Null
 */
kutacc_export void kutacc_af2_invariant_point(kutacc_af2_ipa_s_inputs_t *ipa_s_ptrs,
                                              kutacc_af2_ipa_o_inputs_t *ipa_o_ptrs, kutacc_tensor_h z,
                                              kutacc_tensor_h rigid_rot_mats, kutacc_tensor_h rigid_trans,
                                              kutacc_tensor_h mask, kutacc_af2_ipa_weights_t *ipa_weight_ptrs);

/**
 * @brief impl of rot_vec_mul
 * @param rot_mats shape [..., 3, 3], fp32
 * @param pts shape [..., 3], bf16 / fp32
 * @param trans shape [..., 3], bf16 / fp32
 * @return shape [..., 3], bf16 / fp32
 */
kutacc_export void kutacc_af2_rigid_rot_vec_mul(kutacc_tensor_h pts, kutacc_tensor_h rot_mats, kutacc_tensor_h out,
                                                kutacc_tensor_h trans);

/**
 * @brief impl of rot_mat_mul
 * @param a shape [..., 3, 3], fp32
 * @param b shape [..., 3, 3], fp32
 * @return shape [..., 3, 3], fp32
 */
kutacc_export void kutacc_af2_rigid_rot_matmul(kutacc_tensor_h a, kutacc_tensor_h b, kutacc_tensor_h out);

/**
 * @brief af2_linear algorithm: linear interface for af2 model
 * @param [in] act
 * @param [in] weight
 * @param [in] bias_data
 * @param [in] beta
 * @param [in/out] result
 * @return Null
 */
kutacc_export void kutacc_af2_linear(kutacc_tensor_h act, kutacc_tensor_h weight, float *bias_data,
                                     kutacc_tensor_h result, int64_t beta);

/**
 * @brief gemm prepack for linear layer
 * @param weight shape [n, k]
 * @return result shape [len]
 */
kutacc_export void kutacc_af2_linear_weight_prepack(const __bf16 *weight, __bf16 *result, int64_t n, int64_t k,
                                                    int64_t ldb, int64_t num_threads = 0);

/**
 * @brief get pack size of A or B for linear layer
 * @param identifier A or B
 * @param transa transpose A or not
 * @param transb transpose B or not
 * @param m,n,k A shape[m, k] B shape[k, n]
 * @return  size： m * k or k * n
 */
kutacc_export size_t kutacc_af2_gemm_pack_get_size(char identifier, char transa, char transb, int m, int n, int k);

/**
 * @brief used for generate proj_act
 * @param input_act shape [n_res, n_res_gather, c_z]
 * @param mask shape [n_res, n_res_gather]
 * @param [out] proj_act
 */
kutacc_export void kutacc_af2_triangle_multiplication_calc_proj(kutacc_af2_tm_act_inputs_t *tm_acts_ptr,
                                                                kutacc_tensor_h mask,
                                                                kutacc_af2_tm_proj_weights_t *tm_weights_ptr,
                                                                bool input_prepack);

/**
 * @brief center_act = left_proj * right_proj
 * @param [in] left_proj_act, right_proj_act
 * @param [out] center_act
 */
kutacc_export void kutacc_af2_triangle_multiplication_equation(kutacc_tensor_h center_act,
                                                               kutacc_tensor_h left_proj_act,
                                                               kutacc_tensor_h right_proj_act, int64_t n_res_gather,
                                                               bool is_incoming);

/**
 * @brief gate = inpur_act * gating_w + gating_b & out = center_act * output_proj_w + output_proj_b
 * @param [in] input_act, center_act, gating_w, gating_b, output_proj_w, output_proj_b
 * @param [out] gate, out
 */
kutacc_export void kutacc_af2_triangle_multiplication_gate_and_out_linear(
    kutacc_tensor_h gate, kutacc_tensor_h out, kutacc_af2_tm_act_inputs_t *tm_acts_ptr, kutacc_tensor_h center_act,
    kutacc_af2_tm_linear_weights_t *tm_weights_ptr, bool input_prepack);

/**
 * @brief out = (out + out_proj_b) * sigmoid(gate + gating_b)
 * @param [in] gate
 * @param [out] out
 */
kutacc_export void kutacc_af2_triangle_multiplication_last(kutacc_tensor_h out, kutacc_tensor_h gate, int64_t n_res,
                                                           int64_t n_res_gather, int64_t c_o);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace kutacc {
/*! \enum DType
 *  \brief datatype of tensor.
 */
typedef enum TensorDataType {
    kInt64 = 0,
    kBF16 = 1,
    kNumTypes
} DType;

/** @brief KuTACC kml extend param */
struct BlasExtendParams {
    int64_t num_threads;
    bool prepack_a;
    bool prepack_b;
    bool row_bias;
    bool col_bias;
    void *bias;
    bool relu;
};

struct kutacc_export TensorWrapper {
private:
    kutacc_tensor_h tensor_;

public:
    TensorWrapper(void *data_ptr, std::vector<int64_t> &&sizes, std::vector<int64_t> &&strides, int64_t dim,
                  DType dtype);
    TensorWrapper();
    kutacc_tensor_h get_tensor()
    {
        return tensor_;
    }
    ~TensorWrapper();
};
} // namespace kutacc

#endif
#endif
