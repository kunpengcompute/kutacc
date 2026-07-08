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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <arm_bf16.h>
#include <arm_neon.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "low_latency/low_latency.h"
#include "utils/tensor.h"
#include "utils/memory_pool.h"
#include "utils/parma.h"
#include "utils/check.h"

int count = 0;
int ans_check;
char* flush_ptr;
int flush_sum[32];
constexpr size_t FLUSH_SIZE = 380 * 1024 * 1024;

static int64_t getenv_int64(std::string name)
{
    auto str = std::getenv(name.c_str());
    KUTACC_CHECK(str != nullptr, name);
    return std::atol(str);
}

void initialize_input_tensors(utils::Tensor& topk_weights, utils::Tensor& topk_idx, int64_t n_tokens,
    int64_t max_token, int64_t n_activated_experts, int64_t world_size, [[maybe_unused]] int64_t world_rank)
{
    // 初始化topk_weights
    for (int i = 0; i < max_token; ++i) {
        for (int j = 0; j < n_activated_experts; ++j) {
            topk_weights.data_ptr<float>()[i * n_activated_experts + j] = 0.2f;
        }
    }
    // 初始化topk_idx
    int fact = max_token / 32;
    for (int k = 0; k < fact; ++k) {
        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < n_activated_experts; ++j) {
                topk_idx.data_ptr<int16_t>()[(k * 32 + i) * n_activated_experts * 2 + j * 2] =
                    (i * n_activated_experts + j) % world_size;
                topk_idx.data_ptr<int16_t>()[(k * 32 + i) * n_activated_experts * 2 + j * 2 + 1] = 0;  // local_expert
            }
        }
    }
#ifdef ENABLE_REREDUNDANT_EXPERTS
    load_topk_to_tensor("topk_rank_dp/" + std::to_string(world_rank / 16) + "_dp.txt", topk_idx);
    printf("load_topk_to_tensor end \n");
#endif
    printf("topk_idx\n");
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < n_activated_experts * 2; ++j) {
            printf("%d, ", topk_idx.data_ptr<int16_t>()[i * n_activated_experts * 2 + j]);
        }
        printf("\n");
    }
}

void initialize_dispatch_send_buf(utils::Tensor& dispatch_send_buf, int64_t n_tokens, int64_t dim)
{
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < dim + 4; ++j) {
            dispatch_send_buf.data_ptr<uint8_t>()[i * (dim + 4) + j] = i % 256;
        }
    }
}

bool verify_results(const utils::Tensor& combined, const utils::Tensor& dispatch_send_buf,
    const utils::Tensor& topk_weights, int64_t n_tokens, int64_t dim, int64_t n_activated_experts, int iter)
{
    // 实际输出 ans
    float ans = 0.0f;
    for (int i = 0; i < n_tokens; ++i) {
        for (int j = 0; j < dim; ++j) {
            ans += vcvtah_f32_bf16(combined.data_ptr<bfloat16_t>()[i * dim + j]);
        }
    }

    // 理论输出 true_ans
    float true_ans = 0.0f;
    for (int i = 0; i < n_tokens; ++i) {
        for (int64_t d = 0; d < dim; ++d) {
            float tmp_val = 0.0f;
            for (int j = 0; j < n_activated_experts; ++j) {
                float weight = topk_weights.data_ptr<float>()[i * n_activated_experts + j];
                uint8_t raw_val = dispatch_send_buf.data_ptr<uint8_t>()[i * (dim + 4) + d];
                bfloat16_t bf_val = vcvth_bf16_f32((float)raw_val);
                tmp_val += bf_val * weight;
            }
            // true_ans += tmp_val;
            true_ans += vcvtah_f32_bf16(vcvth_bf16_f32(tmp_val));
        }
    }

    if (std::fabs(ans - true_ans) > 1e-5) {
        printf("%f, true ans is %f, error iter %d\n", ans, true_ans, iter);
        ans_check = 0;
        return false;
    }
    return true;
}

void load_topk_to_tensor(const std::string& file_path, utils::Tensor& topk_idx)
{
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return;
    }
    printf("%s\n", file_path.c_str());
    std::string line;
    // std::getline(file, line); // 跳过第一行
    int bs = 0 ;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int world_rank;
        int p_rank;
        char comma;
        std::vector<int16_t> topk_values(8);

        iss >> world_rank >> comma >> p_rank >> comma;
        for (int i = 0; i < 8; ++i) {
            iss >> topk_values[i];
        }

        int16_t* data_ptr = topk_idx.data_ptr<int16_t>();
        for (int i = 0; i < 8; ++i) {
            data_ptr[bs * 8 + i] = topk_values[i];
        }
        bs += 1 ;
    }
}

void flush_cache()
{
    kutacc::parallel_for((int64_t)0, (int64_t)FLUSH_SIZE, (int64_t)1, [&](int64_t start, int64_t end) {
        int tid = kutacc::get_thread_id();
        for (int i = start; i < end; ++i) {
            flush_sum[tid] += flush_ptr[i];
        }
    });
}

int main(int argc, char *argv[])
{
    if (argc != 12) {
        fprintf(stderr, "Usage: %s is_prefill enable_async_disp enable_async_comb enable_static_route \
         moe_ep moe_dp moe_tp max_token sub_comm redundant_experts test_rounds\n", argv[0]);
        MPI_Init(&argc, &argv);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    kutacc::global_parallel_launch([&] {
    MPI_Init(&argc, &argv);
    flush_ptr = (char *)malloc(FLUSH_SIZE);
    memset(flush_ptr, 0, FLUSH_SIZE);
    params.is_prefill = std::atoi(argv[1]);
    params.use_async_disp = std::atoi(argv[2]);
    params.use_async_comb = std::atoi(argv[3]);
    params.use_static_route = std::atoi(argv[4]);
    params.moe_ep = std::atoi(argv[5]);
    params.moe_dp = std::atoi(argv[6]);
    params.moe_tp = std::atoi(argv[7]);
    params.max_token = std::atoi(argv[8]);
    params.sub_comm = std::atoi(argv[9]);
    params.redundant_n_routed_experts = std::atoi(argv[10]);
    count = std::atoi(argv[11]);
    params.world_size = getenv_int64("MV2_COMM_WORLD_SIZE");
    params.local_size = getenv_int64("MV2_COMM_WORLD_LOCAL_SIZE");
    params.world_rank = getenv_int64("MV2_COMM_WORLD_RANK");
    params.local_rank = getenv_int64("MV2_COMM_WORLD_LOCAL_RANK");
    int node_id = int(params.world_rank / params.local_size);
    int dtp_size = params.moe_ep / params.moe_dp;
    params.n_tokens = params.max_token;

    setenv("IS_PREFILL",            params.is_prefill ? "1" : "0", 1);
    setenv("ENABLE_ASYNC_DISPATCH", params.use_async_disp ? "1" : "0", 1);
    setenv("ENABLE_ASYNC_COMBINE",  params.use_async_comb ? "1" : "0", 1);
    setenv("ENABLE_STATIC_ROUTING", params.use_static_route ? "1" : "0", 1);
    utils::init_memory_pool();
    params.topk_idx = utils::Tensor::alloc_from(utils::ScalarType::Int16,
        {params.max_token, params.n_activated_experts * 2}, utils::on_package_memory_available);
    params.parallel_policy =
        utils::Tensor::alloc_from(utils::ScalarType::Int16, {3}, utils::on_package_memory_available);
    params.parallel_policy.data_ptr<int16_t>()[0] = params.moe_ep;
    params.parallel_policy.data_ptr<int16_t>()[1] = params.moe_dp;
    params.parallel_policy.data_ptr<int16_t>()[2] = params.moe_tp;
    int64_t num_local_experts = params.redundant_n_routed_experts / params.moe_ep;
    auto dispatch_send_size = params.max_token * (params.dim + 4);
    printf("mode: %s\n", params.is_prefill ? "prefill" : "decode");
    printf("route_policy: %s\n", params.use_static_route ? "static" : "dynamic");
    printf("use_async_thread: %s\n", (params.use_async_disp | params.use_async_comb) ? "y" : "n");
    printf("n_tokens=%ld,ep=%ld,dp=%ld,size=%ld,re_experts=%ld\n",
        params.n_tokens, params.moe_ep, params.moe_dp, params.world_size, params.redundant_n_routed_experts);
    int64_t dispatch_recv_size;
    int64_t combine_send_size;
    int64_t combine_recv_size;
    if (params.is_prefill) {
        params.num_max_dispatch_tokens_per_rank =
            (params.max_token / dtp_size) *
            std::min(params.world_size / params.moe_ep, params.n_activated_experts);
        dispatch_recv_size = num_local_experts * 2 * params.max_token * (params.dim + 4)
             + params.redundant_n_routed_experts * (params.num_max_dispatch_tokens_per_rank * 2 + 1) * 2 * 3;
        combine_send_size = num_local_experts * 2 * params.max_token * params.dim;
    } else {
        params.num_max_dispatch_tokens_per_rank =
            (params.max_token / dtp_size) *
            std::min(params.redundant_n_routed_experts / params.moe_ep, params.n_activated_experts);
        dispatch_recv_size =
            params.redundant_n_routed_experts * params.num_max_dispatch_tokens_per_rank * (params.dim + 4) +
            params.redundant_n_routed_experts * (params.num_max_dispatch_tokens_per_rank * 2 + 1) * 2 * 3;
        combine_send_size =
            params.redundant_n_routed_experts * params.num_max_dispatch_tokens_per_rank * params.dim * 2;
    }
    if (params.use_static_route) {
        combine_recv_size = params.max_token * params.n_activated_experts * params.dim * 2;
    } else {
        combine_recv_size = params.max_token / dtp_size * params.n_activated_experts * params.dim * 2;
    }
    printf("dispatch + combine total on_package_memory size: %ld Byte\n", dispatch_send_size +
        dispatch_recv_size + combine_send_size + combine_recv_size);
    void *dispatch_send_ptr = utils::Tensor::alloc_from(utils::ScalarType::UInt8,
        {dispatch_send_size}, utils::on_package_memory_available).ptr;
    void *dispatch_recv_ptr = utils::Tensor::alloc_from(utils::ScalarType::UInt8,
        {dispatch_recv_size}, utils::on_package_memory_available).ptr;
    void *combine_send_ptr = utils::Tensor::alloc_from(utils::ScalarType::UInt8,
        {combine_send_size}, utils::on_package_memory_available).ptr;
    void *combine_recv_ptr;
    if (params.use_static_route) {
        combine_recv_ptr = utils::Tensor::alloc_from(utils::ScalarType::UInt8,
            {combine_recv_size}, utils::shm_available).ptr;
    } else {
        combine_recv_ptr = utils::Tensor::alloc_from(utils::ScalarType::UInt8,
            {combine_recv_size}, utils::on_package_memory_available).ptr;
    }
    params.dispatch_send_buf = low_latency::create_dispatch_send_buf(dispatch_send_ptr, dispatch_recv_ptr,
        utils::ScalarType::UInt8, {params.max_token, params.dim + 4});
    if (params.is_prefill) {
        params.combine_send_buf = low_latency::create_combine_send_buf(combine_send_ptr, combine_recv_ptr,
            utils::ScalarType::BFloat16,
            {1, num_local_experts * 2 * params.max_token, params.dim});
    } else {
        params.combine_send_buf = low_latency::create_combine_send_buf(combine_send_ptr, combine_recv_ptr,
            utils::ScalarType::BFloat16,
            {1, params.redundant_n_routed_experts * params.num_max_dispatch_tokens_per_rank, params.dim});
    }
    params.ds_moe_comm = (low_latency::moe_comm_h)malloc(sizeof(low_latency::moe_comm_h));
    MPI_Comm oob_comm;
    int color = 0;
    if (params.sub_comm == 1) {
        // color = (params.world_rank % 16 < 8) ? 0 : 1;
        color = (params.world_rank < 256) ? 0 : 1;
    }
    MPI_Comm_split(MPI_COMM_WORLD, color, params.world_rank, &oob_comm);
    int sub_comm_size;
    MPI_Comm_size(oob_comm, &sub_comm_size);
    printf("color:%d\n", color);

    low_latency::moe_comm_create(params.dispatch_send_buf, params.combine_send_buf,
        params.redundant_n_routed_experts, params.n_activated_experts, params.num_max_dispatch_tokens_per_rank,
        params.parallel_policy, oob_comm, params.ds_moe_comm);

    utils::Tensor packed_recv_x;
    if (params.is_prefill) {
        packed_recv_x = utils::Tensor::view(low_latency::packed_recv_x,
            {num_local_experts * 2 * params.max_token, params.dim + 4});
    } else {
        packed_recv_x = utils::Tensor::view(low_latency::packed_recv_x,
            {params.redundant_n_routed_experts * params.num_max_dispatch_tokens_per_rank, params.dim + 4});
    }

    auto recv_src_info = low_latency::recv_src_info;
    auto recv_src_info_bak = low_latency::recv_src_info_bak;
    int64_t recv_src_info_ep_bias = sub_comm_size * (params.num_max_dispatch_tokens_per_rank * 2 + 1);

    utils::Tensor topk_weights = utils::Tensor::alloc_from(utils::ScalarType::Float32,
        {{params.max_token, params.n_activated_experts}}, utils::on_package_memory_available);
    auto act_int8_and_scale = utils::Tensor::slice(params.dispatch_send_buf, 0, 0, params.max_token);
    auto combined = low_latency::combined_x;

    initialize_input_tensors(topk_weights, params.topk_idx, params.n_tokens, params.max_token,
        params.n_activated_experts, params.world_size, params.world_rank);
    int correct_times = 0;
    ans_check = 1;
    for (int iter = 1; iter <= count; ++iter) {
        auto true_recv_src_info = recv_src_info;
        if (iter % 2 != 1) {
            true_recv_src_info = recv_src_info_bak;
        }
        initialize_dispatch_send_buf(params.dispatch_send_buf, params.n_tokens, params.dim);
        low_latency::low_latency_barrier(params.ds_moe_comm);
        low_latency::dispatch_send(act_int8_and_scale, params.topk_idx,
            params.redundant_n_routed_experts, params.parallel_policy, 0, params.ds_moe_comm);
        low_latency::dispatch_recv(0, params.ds_moe_comm);
        // Data layout after simulating MOE calculation
        if (params.is_prefill) {
            int64_t src_bias = 0;
            int64_t dst_bias = 0;
            for (int64_t i = 0; i < num_local_experts; ++i) {
                src_bias = i * 2 * params.max_token * (params.dim + 4);
                for (int64_t j = 0; j < params.world_size; ++j) {
                    int recv_nums = true_recv_src_info.data_ptr<int16_t>()[i * recv_src_info_ep_bias +
                        j * (params.num_max_dispatch_tokens_per_rank * 2 + 1)];
                    if (recv_nums > 0) {
                        for (int k = 0; k < recv_nums; ++k) {
                            for (int l = 0; l < params.dim; ++l) {
                                float bbff = packed_recv_x.data_ptr<uint8_t>()[src_bias];
                                params.combine_send_buf.data_ptr<bfloat16_t>()[dst_bias] = vcvth_bf16_f32(bbff);
                                src_bias++;
                                dst_bias++;
                            }
                            src_bias += 4;
                        }
                    }
                }
            }
        } else {
            int64_t ep_bias = 0;
            for (int64_t i = 0; i < num_local_experts; ++i) {
                for (int64_t j = 0; j < params.world_size; ++j) {
                    int recv_nums = true_recv_src_info.data_ptr<int16_t>()[i * recv_src_info_ep_bias +
                        j * (params.num_max_dispatch_tokens_per_rank * 2 + 1)];
                    if (recv_nums > 0) {
                        for (int k = 0; k < recv_nums; ++k) {
                            for (int l = 0; l < params.dim; ++l) {
                                float bbff = packed_recv_x.data_ptr<uint8_t>()[
                                    i * params.world_size * params.num_max_dispatch_tokens_per_rank * (params.dim + 4) +
                                    j * params.num_max_dispatch_tokens_per_rank * (params.dim + 4) +
                                    k * (params.dim + 4) + l];
                                params.combine_send_buf.data_ptr<bfloat16_t>()[ep_bias] = vcvth_bf16_f32(bbff);
                                ep_bias++;
                            }
                        }
                    }
                }
            }
        }
        flush_cache();
        low_latency::low_latency_barrier(params.ds_moe_comm);
        low_latency::combine_send(params.combine_send_buf, true_recv_src_info, params.redundant_n_routed_experts,
            params.parallel_policy, 0, params.ds_moe_comm, params.topk_idx, topk_weights);
        low_latency::combine_recv(
            params.topk_idx, topk_weights, params.parallel_policy, params.dim, 0, params.ds_moe_comm);
        /* check result */
        if (verify_results(combined, params.dispatch_send_buf, topk_weights,
            params.n_tokens, params.dim, params.n_activated_experts, iter)) {
            correct_times++;
        }
    }
    printf("Test %d rounds, correct %d rounds\n", count, correct_times);
    low_latency::moe_comm_destroy();
    utils::finalize_memory_pool();
    MPI_Finalize();
    if (!ans_check)exit(1);
    });
}