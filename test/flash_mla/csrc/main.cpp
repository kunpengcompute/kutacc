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
    struct timeval clock;
    gettimeofday(&clock, NULL);
    return 1000000.0 * clock.tv_sec + clock.tv_usec;
}

constexpr size_t MEM_SIZE = 3LL * 1024 * 1024 * 1024;
constexpr size_t CONTEXT_SIZE = 32 * 1024 * 1024;
constexpr size_t FLUSH_SIZE = 380 * 1024 * 1024;
constexpr size_t MAX_NUM_THREADS = 38;

constexpr int input_tokens = 1024;
constexpr int output_tokens = 64;
constexpr int block_size = 64;
constexpr int batch_size = 16;
constexpr int num_heads = 64;
constexpr int num_layers = 61;
constexpr int head_dim = 576;
constexpr int head_dim_v = 512;

Tensor make_tensor(const std::string &dtype, const std::vector<int64_t> &sizes, char *&ptr)
{
    Tensor tensor;
    if (dtype == "bfloat16") {
        tensor = Tensor::inplace_create(ScalarType::kBFloat16, sizes, ptr);
    } else if (dtype == "float32") {
        tensor = Tensor::inplace_create(ScalarType::kFloat, sizes, ptr);
    } else if (dtype == "int8") {
        tensor = Tensor::inplace_create(ScalarType::kChar, sizes, ptr);
    } else if (dtype == "uint8") {
        tensor = Tensor::inplace_create(ScalarType::kByte, sizes, ptr);
    } else if (dtype == "int32") {
        tensor = Tensor::inplace_create(ScalarType::kInt, sizes, ptr);
    } else if (dtype == "int64") {
        tensor = Tensor::inplace_create(ScalarType::kLong, sizes, ptr);
    } else {
        FLASH_ASSERT(false);
    }
    return tensor;
}

PersistentBuffer mem_pool;
PersistentBuffer context_buffer;
PersistentBuffer flush_buffer;

void init()
{
    mem_pool.malloc_buffer(MEM_SIZE);
    memset(mem_pool.g_buf, 0, MEM_SIZE);
    flush_buffer.malloc_buffer(FLUSH_SIZE);
    memset(flush_buffer.g_buf, 0, FLUSH_SIZE);
    context_buffer.malloc_buffer(CONTEXT_SIZE);
}

Tensor block_tables, seq_lens_tensor, tile_scheduler_metadata, num_splits;
constexpr int total_num_tokens = input_tokens + output_tokens;
constexpr int num_blocks_per_seq = ceil_div(total_num_tokens, block_size);

void update(int seq_len)
{
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
    flash_mla_get_metadata(seq_lens_tensor, num_heads, 1, tile_scheduler_metadata, num_splits, ptr);
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
    static char *ptr;
    static constexpr int check_step = 23;
    bool check_diff = token_id % check_step == 0 && layer_id == 0;
    if (layer_id == 0) {
        ptr = (char *)mem_pool.g_buf;
        update(input_tokens + token_id + 1);
    }

    const auto q = make_tensor("bfloat16", {batch_size, num_heads, head_dim}, ptr);
    const auto kvcache = make_tensor("bfloat16", {num_blocks_per_seq * batch_size, block_size, head_dim}, ptr);
    const double softmax_scale = 1 / sqrt(head_dim);
    const bool causal = true;
    auto o = make_tensor("bfloat16", {batch_size, num_heads, head_dim_v}, ptr);

    if (check_diff) {
        std::mt19937 rnd(time(NULL));
        std::normal_distribution<> std_norm;
        for (auto x : {q, kvcache}) {
            CHECK_CONTIGUOUS(x);
            for (int i = 0; i < x.numel(); ++i) {
                x.data_ptr<bfloat16_t>()[i] = std_norm(rnd);
            }
        }
    }

    flush_cache();

    auto start = get_clock_us();

    flash_mla_with_kvcache(q, kvcache, block_tables, seq_lens_tensor, head_dim_v,
                           tile_scheduler_metadata, num_splits, softmax_scale, causal, o);

    auto end = get_clock_us();

    auto time = end - start;
    sum_time += time;
    max_time = std::max<double>(max_time, time);
    min_time = std::min<double>(min_time, time);
    ++test_times;

    if (check_diff) {
        auto o_std = make_tensor("bfloat16", {batch_size, num_heads, head_dim_v}, ptr);
        naive_mla_with_kvcache(q, kvcache, block_tables, seq_lens_tensor, head_dim_v, softmax_scale, o_std);
        CHECK_CONTIGUOUS(o);
        CHECK_CONTIGUOUS(o_std);
        double sum_xy = 0, sum_x2_y2 = 0;
        for (int i = 0; i < o.numel(); ++i) {
            double x = o.data_ptr<bfloat16_t>()[i];
            double y = o_std.data_ptr<bfloat16_t>()[i];
            sum_xy += x * y;
            sum_x2_y2 += x * x + y * y;
        }
        constexpr double eps = 1e-12;
        if (sum_x2_y2 > eps) {
            double cos_diff = 1 - 2 * sum_xy / sum_x2_y2;
            FLASH_ASSERT(cos_diff < 1e-5);
        }
    }
}

int main(int argc, char *argv[])
{
    init();
    flash_mla_test(0, 0);
    
    sum_time = max_time = 0;
    min_time = INFINITY;
    test_times = 0;

    for (int i = 0; i < output_tokens; ++i) {
        for (int j = 0; j < num_layers; ++j) {
            flash_mla_test(i, j);
        }
    }

    printf("---- Inference Parameters ----\n");
    printf("input_tokens = %d\n", input_tokens);
    printf("output_tokens = %d\n", output_tokens);
    printf("batch_size = %d\n", batch_size);
    printf("num_heads = %d\n", num_heads);
    printf("num_layers = %d\n", num_layers);
    
    printf("---- FlashMLA Performance ----\n");
    double avg_time = sum_time / test_times;
    double avg_num_tokens = input_tokens + (1 + output_tokens) / 2.0;
    double FLOPs = batch_size * avg_num_tokens * num_heads * (head_dim + head_dim_v) * 2;
    double num_bytes =
        batch_size * (avg_num_tokens * head_dim + num_heads * (head_dim + head_dim_v)) * sizeof(bfloat16_t);
    printf("min_time = %.3lf us\n", min_time);
    printf("max_time = %.3lf us\n", max_time);
    printf("avg_time = %.3lf us\n", avg_time);
    printf("%.3lf TFLOPS, %.3lf GB/s\n", FLOPs / 1e6 / avg_time, num_bytes / 1e3 / avg_time);

    printf("------------------------------\n");

    check_flush_sum();

    return 0;
}