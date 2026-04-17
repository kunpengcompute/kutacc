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
#include <cstdlib>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <arm_bf16.h>
#include <arm_neon.h>

#include "moe/low_latency.h"
#include "utils/tensor.h"
#include "utils/memory_pool.h"
#include "utils/parma.h"
#include "utils/check.h"

int count = 0;
bfloat16_t factor[16] = {0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2, 8.0, 8.8, 9.6, 10.4, 11.2, 12.0};

static int64_t getenv_int64(std::string name)
{
    auto str = std::getenv(name.c_str());
    KUTACC_CHECK(str != nullptr, name);
    return std::atol(str);
}

void printf_perf()
{
    double disp_token_max;
    double disp_token_min;
    double disp_token_ave;
    double disp_meta_max;
    double disp_meta_min;
    double disp_meta_ave;
    double dis_recv_max;
    double dis_recv_min;
    double dis_recv_ave;
    double put_barrier_max;
    double put_barrier_min;
    double put_barrier_ave;
    double put_token_max;
    double put_token_min;
    double put_token_ave;
    double rec_barrier_max;
    double rec_barrier_min;
    double rec_barrier_ave;
    double rec_reduce_max;
    double rec_reduce_min;
    double rec_reduce_ave;
    double rec_allgather_max;
    double rec_allgather_min;
    double rec_allgather_ave;

    double disp_perf[3];
    double comb_perf[5];
    kutacc::moe_dispatch_get_perf(disp_perf);
    kutacc::moe_combine_get_perf(comb_perf);

    MPI_Reduce(&disp_perf[0], &disp_token_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&disp_perf[1], &disp_meta_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&disp_perf[2], &dis_recv_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&disp_perf[0], &disp_token_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&disp_perf[1], &disp_meta_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&disp_perf[2], &dis_recv_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    MPI_Reduce(&disp_perf[0], &disp_token_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&disp_perf[1], &disp_meta_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&disp_perf[2], &dis_recv_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    disp_token_ave = disp_token_ave / params.world_size;
    disp_meta_ave = disp_meta_ave / params.world_size;
    dis_recv_ave = dis_recv_ave / params.world_size;

    MPI_Reduce(&comb_perf[0], &put_barrier_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[1], &put_token_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[2], &rec_barrier_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[3], &rec_reduce_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[4], &rec_allgather_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&comb_perf[0], &put_barrier_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[1], &put_token_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[2], &rec_barrier_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[3], &rec_reduce_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[4], &rec_allgather_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    MPI_Reduce(&comb_perf[0], &put_barrier_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[1], &put_token_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[2], &rec_barrier_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[3], &rec_reduce_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comb_perf[4], &rec_allgather_ave, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    put_barrier_ave = put_barrier_ave / params.world_size;
    put_token_ave = put_token_ave / params.world_size;
    rec_barrier_ave = rec_barrier_ave / params.world_size;
    rec_reduce_ave = rec_reduce_ave / params.world_size;
    rec_allgather_ave = rec_allgather_ave / params.world_size;

    if (params.world_rank == 0) {
        printf("\nPerformance Metrics (us):\n");
        printf("==============dispatch==============\n");
        printf("%-18s %6s %6s %6s\n", "Operation", "Max", "Min", "Ave");
        printf("%-18s %6.2f %6.2f %6.2f\n", "iov_make",
            disp_token_max / count, disp_token_min / count, disp_token_ave / count);
        printf("%-18s %6.2f %6.2f %6.2f\n", "iov_put",
            disp_meta_max / count, disp_meta_min / count, disp_meta_ave / count);
        printf("%-18s %6.2f %6.2f %6.2f\n", "mesg_recv",
            dis_recv_max / count, dis_recv_min / count, dis_recv_ave / count);
        printf("==============combine===============\n");
        printf("%-18s %6s %6s %6s\n", "Operation", "Max", "Min", "Ave");
        printf("%-18s %6.2f %6.2f %6.2f\n", "iov_make",
            put_barrier_max / count, put_barrier_min / count, put_barrier_ave / count);
        printf("%-18s %6.2f %6.2f %6.2f\n", "iov_put",
            put_token_max / count, put_token_min / count, put_token_ave / count);
        printf("%-18s %6.2f %6.2f %6.2f\n", "mesg_recv",
            rec_barrier_max / count, rec_barrier_min / count, rec_barrier_ave / count);
        printf("%-18s %6.2f %6.2f %6.2f\n", "reduce+allgather",
            rec_reduce_max / count, rec_reduce_min / count, rec_reduce_ave / count);
        printf("%-18s %6.2f %6.2f %6.2f\n", "fence",
            rec_allgather_max / count, rec_allgather_min / count, rec_allgather_ave / count);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    count = atoi(argv[1]);
    printf("start\n");
    utils::init_memory_pool();

    params.world_size = getenv_int64("MV2_COMM_WORLD_SIZE");
    params.local_size = getenv_int64("MV2_COMM_WORLD_LOCAL_SIZE");
    params.world_rank = getenv_int64("MV2_COMM_WORLD_RANK");
    params.local_rank = getenv_int64("MV2_COMM_WORLD_LOCAL_RANK");
    int node_id = int(params.world_rank / params.local_size);
    int moe_balance = getenv_int64("MOE_BALANCE");
    params.max_token = getenv_int64("MAX_TOKEN");
    params.moe_ep = getenv_int64("MOE_EP");
    params.moe_dp = getenv_int64("MOE_DP");
    params.moe_tp = getenv_int64("MOE_TP");
    params.num_max_dispatch_tokens_per_rank =
        (params.max_token / params.world_size) *
        std::min(params.n_routed_experts / params.moe_ep, params.n_activated_experts);
    params.topk_idx = utils::Tensor::alloc_from(utils::ScalarType::Int16,
        {params.n_tokens, params.n_activated_experts}, utils::on_package_memory_available);
    params.parallel_sizes = utils::Tensor::alloc_from(utils::ScalarType::Int16, {3}, utils::on_package_memory_available);
    params.parallel_sizes.data_ptr<int16_t>()[0] = params.moe_ep;
    params.parallel_sizes.data_ptr<int16_t>()[1] = params.moe_dp;
    params.parallel_sizes.data_ptr<int16_t>()[2] = params.moe_tp;

    auto recv_size =
        (params.n_routed_experts / params.moe_ep) * params.world_size * params.num_max_dispatch_tokens_per_rank;
    auto resue_size = recv_size * params.dim * 2;
    auto stay_size = recv_size * params.dim * 2 + 4 * 1024 * 1024;
    printf("routed_experts:%ld, moe_ep:%ld, world_size:%ld, num_max_dispatch_tokens_per_rank:%ld, dim:%ld\n",
        params.n_routed_experts, params.moe_ep, params.world_size, params.num_max_dispatch_tokens_per_rank, params.dim);
    printf("dispatch + combine total on_package_memory size: %ld Byte\n", resue_size + stay_size);
    void *baseptr_reuse = utils::Tensor::alloc_from(
        utils::ScalarType::UInt8, {resue_size}, utils::on_package_memory_available).ptr;
    void *baseptr_stay = utils::Tensor::alloc_from(
        utils::ScalarType::UInt8, {stay_size}, utils::on_package_memory_available).ptr;
    void *dispatch_send_ptr = utils::Tensor::alloc_from(utils::ScalarType::UInt8, {params.max_token * (params.dim + 4)},
        utils::on_package_memory_available).ptr;
    memset(baseptr_reuse, 0, resue_size);
    memset(baseptr_stay, 0, stay_size);
    params.dispatch_send_buf = low_latency::create_dispatch_send_buf(dispatch_send_ptr, utils::ScalarType::UInt8,
        {params.max_token, params.dim + 4});
    params.combine_send_buf = low_latency::create_combine_send_buf(baseptr_reuse, baseptr_stay,
        utils::ScalarType::BFloat16, {1, recv_size, params.dim});
    low_latency::dispatch_init(params.dispatch_send_buf, params.n_routed_experts,
        params.num_max_dispatch_tokens_per_rank, params.dim + 4);
    low_latency::combine_init(params.combine_send_buf, params.max_token, params.n_routed_experts,
        params.n_activated_experts, params.num_max_dispatch_tokens_per_rank, params.dim);

    auto packed_recv_x = utils::Tensor::view(low_latency::packed_recv_x, {recv_size, params.dim + 4});
    auto recv_src_info = low_latency::recv_src_info;
    int64_t num_local_experts = params.n_routed_experts / params.moe_ep;
    int64_t recv_src_info_ep_bias = params.world_size * (params.num_max_dispatch_tokens_per_rank + 1);

    utils::Tensor topk_weights = utils::Tensor::alloc_from(utils::ScalarType::Float32,
        {{params.n_tokens, params.n_activated_experts}}, utils::on_package_memory_available);
    auto act_int8_and_scale = utils::Tensor::slice(params.dispatch_send_buf, 0, 0, 128);
    auto combined = low_latency::combined_x;
    for (int i = 0; i < params.n_tokens; ++i) {
        for (int j = 0; j < params.n_activated_experts; ++j) {
            topk_weights.data_ptr<float>()[i * params.n_activated_experts + j] = 0.1;
        }
    }
    /* init topk_idx for load balance */
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 8; ++j) {
                params.topk_idx.data_ptr<int16_t>()[(k * 32 + i) * 8 + j] = i * 8 + j;
            }
        }
    }
    /* init topk_idx unbalance */
    if (moe_balance != 1) {
        for (int i = 64; i < 128; ++i) {
            for (int j = 0; j < 8; ++j) {
                params.topk_idx.data_ptr<int16_t>()[i * 8 + j] = i / 4 + j * 4;
            }
        }
    }

    int correct_times = 0;
    for (int iter = 0; iter < count; ++iter) {
        /* 2P dispatch input init */
        if (params.moe_ep == 16) {
            for (int i = 0; i < params.max_token; ++i) {
                for (int j = 0; j < params.dim + 4; ++j) {
                    params.dispatch_send_buf.data_ptr<uint8_t>()[i * (params.dim + 4) + j] = params.world_rank;
                }
            }
        } else if (params.moe_ep == 256) {  // 32P dispatch input init
            for (int i = 0; i < params.max_token; ++i) {
                for (int j = 0; j < params.dim + 4; ++j) {
                    params.dispatch_send_buf.data_ptr<uint8_t>()[i * (params.dim + 4) + j] = node_id;
                }
            }
        }
        low_latency::low_latency_barrier();
        low_latency::dispatch_send(act_int8_and_scale, params.topk_idx, params.num_max_dispatch_tokens_per_rank,
            params.n_routed_experts, params.parallel_sizes, 0);
        usleep(150);    // Simulation computing communication hiding
        low_latency::dispatch_recv(params.n_routed_experts, params.num_max_dispatch_tokens_per_rank, 0);
        // Data layout after simulating MOE calculation
        int64_t ep_bias = 0;
        for (int64_t i = 0; i < num_local_experts; ++i) {
            for (int64_t j = 0; j < params.world_size; ++j) {
                int recv_nums = recv_src_info.data_ptr<int16_t>()[i * recv_src_info_ep_bias +
                    j * (params.num_max_dispatch_tokens_per_rank + 1)];
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
        low_latency::low_latency_barrier();
        low_latency::combine_send(params.combine_send_buf, recv_src_info, params.n_tokens,
            params.num_max_dispatch_tokens_per_rank, params.n_routed_experts, params.parallel_sizes, 0);
        usleep(150);    // Simulation computing communication hiding
        low_latency::combine_recv(params.topk_idx, topk_weights, recv_src_info, params.num_max_dispatch_tokens_per_rank,
            params.n_routed_experts, params.parallel_sizes, params.dim, 0);
        /* check result */
        if (params.parallel_sizes.data_ptr<int16_t>()[0] == 16) {
            float ans = 0;
            float true_ans = 5494442.0;
            for (int i = 0; i < params.n_tokens; ++i) {
                for (int j = 0; j < params.dim; ++j) {
                    ans += vcvtah_f32_bf16(combined.data_ptr<bfloat16_t>()[i * params.dim + j]);
                }
            }
            if (ans == true_ans) {
                correct_times++;
            } else {
                printf("%f, error iter %d\n", ans, iter);
            }
        }
        if (params.parallel_sizes.data_ptr<int16_t>()[0] == 256) {
            float ans = 0;
            float true_ans = 0;
            for (int i = 0; i < params.n_tokens; ++i) {
                for (int j = 0; j < params.dim; ++j) {
                    ans += vcvtah_f32_bf16(combined.data_ptr<bfloat16_t>()[i * params.dim + j]);
                }
            }
            for (int i = 0; i < params.n_tokens; ++i) {
                for (int j = 0; j < params.dim; ++j) {
                    true_ans += vcvtah_f32_bf16(factor[node_id]);
                }
            }
            if (ans == true_ans) {
                correct_times++;
            } else {
                printf("%f, error iter %d\n", ans, iter);
            }
        }
    }
    printf("Test %d times, correct %d times\n", count, correct_times);
    printf_perf();
    low_latency::dispatch_finalize();
    low_latency::combine_finalize();
    utils::finalize_memory_pool();
    MPI_Finalize();
}