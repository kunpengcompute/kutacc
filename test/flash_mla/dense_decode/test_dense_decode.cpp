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
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <sys/time.h>
#include <random>

#include "helper.h"

#include "kutacc.h"
#include "utils.h"
#include "mla.h"

template <typename T>
constexpr T ceil_div(const T &x, const T &y)
{
    return (x + y - 1) / y;
}

inline double get_clock_us()
{
    static bool init = false;
    if (!init) {
        kutacc_time_init();
        init = true;
    }
    return kutacc_now_ns() / 1000.0;
}

constexpr int64_t MPOOL_SIZE = 2LL * 1024 * 1024 * 1024;
constexpr int64_t CONTEXT_SIZE = 32 * 1024 * 1024;
constexpr int64_t FLUSH_SIZE = 380 * 1024 * 1024;
constexpr int64_t MAX_NUM_THREADS = 38;
constexpr bool COMPLETE_BENCHMARK = false;

constexpr int block_size = 64;
constexpr int head_dim = 576;
constexpr int head_dim_v = 512;
constexpr bool is_kv_packed = false;

int input_tokens;
int output_tokens;
int batch_size;
int num_heads;
int num_layers;
int mtp;

PersistentBuffer mem_pool;
PersistentBuffer context_buffer;
PersistentBuffer flush_buffer;

void init()
{
    mem_pool.malloc_buffer(MPOOL_SIZE);
    memset(mem_pool.g_buf, 0, MPOOL_SIZE);
    flush_buffer.malloc_buffer(FLUSH_SIZE);
    memset(flush_buffer.g_buf, 0, FLUSH_SIZE);
    context_buffer.malloc_buffer(CONTEXT_SIZE);
}

Tensor block_tables, seq_lens_tensor;
int total_num_tokens;
int num_blocks_per_seq;

void update(int seq_len)
{
    total_num_tokens = input_tokens + output_tokens;
    num_blocks_per_seq = ceil_div(total_num_tokens, block_size);
    char *ptr = (char *)context_buffer.g_buf;
    block_tables = make_tensor("int32", {batch_size, num_blocks_per_seq}, ptr);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_blocks_per_seq; ++j) {
            block_tables.data_ptr<int32_t>()[i * num_blocks_per_seq + j] = i * num_blocks_per_seq + j;
        }
    }
    seq_lens_tensor = make_tensor("int32", {batch_size}, ptr);
    for (int i = 0; i < batch_size; ++i) {
        seq_lens_tensor.data_ptr<int32_t>()[i] = seq_len;
    }
    flash_mla_get_metadata(seq_lens_tensor, mtp + 1, num_heads, head_dim, head_dim_v, block_size, is_kv_packed);
}

int flush_sum[MAX_NUM_THREADS];

void flush_cache()
{
    kutacc::parallel_for((int64_t)0, (int64_t)FLUSH_SIZE, (int64_t)1, [&](int64_t start, int64_t end) {
        int tid = kutacc::get_thread_id();
        for (int i = start; i < end; ++i) {
            flush_sum[tid] += ((char *)flush_buffer.g_buf)[i];
        }
    });
}

void check_flush_sum()
{
    int s = 0;
    int tnum = kutacc::get_thread_num();
    for (int i = 0; i < tnum; ++i) {
        s ^= flush_sum[i];
    }
    if (s) {
        assert(0);
    }
}

double sum_time, max_time, min_time;
int test_times;

void flash_mla_test(int token_id, int layer_id)
{
    char *ptr = (char *)mem_pool.g_buf;
    static constexpr int check_step = 23;
    bool check_diff = token_id % check_step == 0 && layer_id == 0;
    if (layer_id == 0) {
        update(input_tokens + token_id + mtp + 1);
    }

    const auto q = make_tensor("bfloat16", {batch_size, mtp + 1, num_heads, head_dim}, ptr);
    const auto kvcache = make_tensor("bfloat16", {num_blocks_per_seq * batch_size, block_size, head_dim}, ptr);
    const double softmax_scale = 1 / sqrt(head_dim);
    const bool causal = true;
    auto o = make_tensor("bfloat16", {batch_size, mtp + 1, num_heads, head_dim_v}, ptr);

    if (check_diff) {
        std::mt19937 rnd(0);
        std::normal_distribution<> std_norm;
        for (auto x : {q, kvcache}) {
            CHECK_CONTIGUOUS(x);
            for (int i = 0; i < x.numel(); ++i) {
                x.data_ptr<bfloat16_t>()[i] = std_norm(rnd);
            }
        }
    }

    const auto packed_kcache = make_tensor("bfloat16", {num_blocks_per_seq * batch_size, block_size, head_dim}, ptr);
    const auto packed_vcache = make_tensor("bfloat16", {num_blocks_per_seq * batch_size, block_size, head_dim_v}, ptr);
    if (is_kv_packed) {
        mla_pack_kvcache(kvcache, packed_kcache, packed_vcache);
    }
    flush_cache();

    auto start = get_clock_us();

    if (!is_kv_packed) {
        flash_mla_with_kvcache(q, kvcache, Tensor{}, block_tables, seq_lens_tensor,
                               softmax_scale, causal, o);
    } else {
        flash_mla_with_kvcache(q, packed_kcache, packed_vcache, block_tables, seq_lens_tensor,
                               softmax_scale, causal, o);
    }

    auto end = get_clock_us();

    auto time = end - start;
    sum_time += time;
    max_time = std::max<double>(max_time, time);
    min_time = std::min<double>(min_time, time);
    ++test_times;

    if (check_diff) {
        auto o_std = make_tensor("bfloat16", {batch_size, mtp + 1, num_heads, head_dim_v}, ptr);
        naive_mla_with_kvcache(q, kvcache, block_tables, seq_lens_tensor, head_dim_v, softmax_scale, causal, o_std);
        CHECK_CONTIGUOUS(o);
        CHECK_CONTIGUOUS(o_std);
        double sum_xy = 0;
        double sum_x2_y2 = 0;
        double amax_diff = 0;
        for (int i = 0; i < o.numel(); ++i) {
            double x = o.data_ptr<bfloat16_t>()[i];
            double y = o_std.data_ptr<bfloat16_t>()[i];
            sum_xy += x * y;
            sum_x2_y2 += x * x + y * y;
            amax_diff = std::max(amax_diff, abs(x - y));
        }
        constexpr double eps = 1e-12;
        FLASH_ASSERT(!isnan(sum_x2_y2));
        if (sum_x2_y2 > eps) {
            double cos_diff = 1 - 2 * sum_xy / sum_x2_y2;
            // double RMSE = sqrt((sum_x2_y2 - 2 * sum_xy) / o.numel());
            // printf("cos_diff = %lf, RMSE = %lf, amax_diff = %lf\n", cos_diff, RMSE, amax_diff);
            FLASH_ASSERT(cos_diff < 1e-5);
        }
    }
}

void test(int input_tokens_, int output_tokens_, int batch_size_, int num_heads_, int num_layers_, int mtp_ = 0)
{
    input_tokens = input_tokens_;
    output_tokens = output_tokens_;
    batch_size = batch_size_;
    num_heads = num_heads_;
    num_layers = num_layers_;
    mtp = mtp_;

    FLASH_ASSERT(output_tokens % (mtp + 1) == 0);

    init();

    flash_mla_test(0, 0);
    
    sum_time = max_time = 0;
    min_time = INFINITY;
    test_times = 0;

    for (int i = 0; i < output_tokens; i += mtp + 1) {
        for (int j = 0; j < num_layers; ++j) {
            flash_mla_test(i, j);
        }
    }
    double avg_time = sum_time / test_times;
    double avg_num_tokens = input_tokens + (1 + output_tokens) / 2.0;
    double FLOPs = batch_size * avg_num_tokens * num_heads * (head_dim + head_dim_v) * 2 * (mtp + 1);
    double num_bytes =
        batch_size * (avg_num_tokens * (is_kv_packed ? head_dim + head_dim_v : head_dim) +
        num_heads * (head_dim + head_dim_v) * (mtp + 1)) * sizeof(bfloat16_t);
    if (!COMPLETE_BENCHMARK) {
        printf("-------- Inference Parameters --------\n");
        printf("input_tokens  = %d\n", input_tokens);
        printf("output_tokens = %d\n", output_tokens);
        printf("batch_size    = %d\n", batch_size);
        printf("num_heads     = %d\n", num_heads);
        printf("num_layers    = %d\n", num_layers);
        printf("mtp           = %d\n", mtp);
        
        printf("-------- FlashMLA Performance --------\n");
        printf("min_time = %.3lf us\n", min_time);
        printf("max_time = %.3lf us\n", max_time);
        printf("avg_time = %.3lf us\n", avg_time);
        printf("%.3lf ms, %.3lf TFLOPS, %.3lf GB/s\n",
            avg_time / 1e3, FLOPs / 1e6 / avg_time, num_bytes / 1e3 / avg_time);

        printf("--------------------------------------\n");
    } else {
        printf("%d %d %d %.lf %.3lf %.3lf %.3lf\n", mtp, batch_size, num_heads, avg_num_tokens,
               avg_time / 1e3, FLOPs / 1e6 / avg_time, num_bytes / 1e3 / avg_time);
    }

    check_flush_sum();
}

int main()
{
    kutacc::global_parallel_launch([&] {
        flash_mla_init();
        // test: input_tokens, output_tokens, batch_size, num_heads, num_layers, mtp = 0 (default)
        if (!COMPLETE_BENCHMARK) {
            test(1023, 1, 8, 128, 100, 0);
            test(1023, 1, 16, 128, 100, 0);
            test(1022, 2, 4, 128, 100, 1);
            test(1022, 2, 8, 128, 100, 1);
            // test(1024, 64, 8, 128, 61, 0);
            // test(1024, 64, 4, 128, 61, 1);
        } else {
            for (int i = 1; i <= 16; i <<= 1) {
                test(i * 1024 - 1, 1, 8, 128, 61, 0);
            }
            for (int i = 1; i <= 16; i <<= 1) {
                test(i * 1024 - 1, 1, 12, 128, 61, 0);
            }
            for (int i = 1; i <= 16; i <<= 1) {
                test(i * 1024 - 1, 1, 16, 128, 61, 0);
            }
            for (int i = 1; i <= 16; i <<= 1) {
                test(i * 1024 - 2, 2, 4, 128, 61, 1);
            }
            for (int i = 1; i <= 16; i <<= 1) {
                test(i * 1024 - 2, 2, 6, 128, 61, 1);
            }
            for (int i = 1; i <= 16; i <<= 1) {
                test(i * 1024 - 2, 2, 8, 128, 61, 1);
            }
            for (int i = 1; i <= 16; i <<= 1) {
                test(i * 1024 - 2, 2, 16, 128, 61, 1);
            }
            test(1024, 64, 8, 128, 61, 0);
            test(1024, 64, 12, 128, 61, 0);
            test(1024, 64, 16, 128, 61, 0);
            test(1024, 64, 4, 128, 61, 1);
            test(1024, 64, 6, 128, 61, 1);
            test(1024, 64, 8, 128, 61, 1);
            test(1024, 64, 16, 128, 61, 1);
        }
        flash_mla_finalize();
    });
    return 0;
}