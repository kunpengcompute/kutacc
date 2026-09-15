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
#include <random>
#include <iomanip>
#include <kutacc.h>

#include "../utils.h"
#include "../allocator.h"
#include "../cache_flush.h"
#include "ref_mla.h"
#include "testcase.h"

kutacc::FlashMLAMetaHandle flash_mla_meta;

constexpr int64_t MAX_ALLOC_SIZE = 3600LL * 1024 * 1024;
constexpr int64_t CACHE_FLUSH_SIZE = 10 * 1024 * 1024;

std::vector<TestcaseParams> gen_testcase()
{
    std::vector<TestcaseParams> correctness_cases;
    std::vector<TestcaseParams> corner_cases;
    std::vector<TestcaseParams> performance_cases;

    constexpr int64_t head_dim = 512;
    constexpr int64_t head_dim_v = 512;

    for (auto have_topk_len : {false, true}) {
        for (auto have_extra_k : {false, true}) {
            for (auto have_extra_topk_len : {false, true}) {
                if (!have_extra_k && have_extra_topk_len) {
                    continue;
                }
                for (auto num_heads_q : {64, 128}) {
                    std::vector<std::tuple<int64_t, int64_t, int64_t>> kvcache_params = {
                        {512, 64, 2},
                        {512, 64, 64},
                        {512, 64, 69},
                        {1024, 576, 2},
                        {1024, 576, 61},
                        {2046, 2048, 2},
                        {2046, 2048, 64},
                        {2046, 2048, 576},
                    };
                    auto extra_kvcache_params = kvcache_params;
                    if (!have_extra_k) {
                        extra_kvcache_params = {{0, 0, 0}};
                    }
                    for (auto [seqlen_kv, topk, page_block_size] : kvcache_params) {
                        for (auto [extra_seqlen_kv, extra_topk, extra_page_block_size] : extra_kvcache_params) {
                            for (auto batch_size : {4, 74, 321}) {
                                for (auto seqlen_q : {1, 3}) {
                                    correctness_cases.push_back({
                                        batch_size, seqlen_q, num_heads_q,
                                        head_dim, head_dim_v,
                                        seqlen_kv, page_block_size, topk,
                                        extra_seqlen_kv, extra_page_block_size, extra_topk,
                                        have_topk_len, have_extra_topk_len, true,
                                        true, true, 0
                                    });
                                }
                            }
                        }
                    }
                    kvcache_params = extra_kvcache_params = {
                        {512, 64, 61},
                        {650, 576, 53},
                    };
                    if (!have_extra_k) {
                        extra_kvcache_params = {{0, 0, 0}};
                    }
                    for (auto [seqlen_kv, topk, page_block_size] : kvcache_params) {
                        for (auto [extra_seqlen_kv, extra_topk, extra_page_block_size] : extra_kvcache_params) {
                            for (auto batch_size : {4, 74, 321}) {
                                for (auto seqlen_q : {3}) {
                                    for (auto enable_attn_sink : {true, false}) {
                                        corner_cases.push_back({
                                            batch_size, seqlen_q, num_heads_q,
                                            head_dim, head_dim_v,
                                            seqlen_kv, page_block_size, topk,
                                            extra_seqlen_kv, extra_page_block_size, extra_topk,
                                            have_topk_len, have_extra_topk_len, enable_attn_sink,
                                            true, true, 0
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    performance_cases = {
        {8, 1, 128, 576, 512, 1024, 64, 1024, 0, 0, 0, false, false, false, false, true, 100},
        {16, 1, 128, 576, 512, 1024, 64, 1024, 0, 0, 0, false, false, false, false, true, 100},
        {8, 1, 128, 512, 512, 1024, 64, 1024, 128, 128, 128, false, false, false, false, true, 100},
    };
    std::vector<TestcaseParams> testcases;
    testcases.insert(testcases.end(), performance_cases.begin(), performance_cases.end());
    return testcases;
}

void test_flash_mla(const TestcaseParams &p, int64_t rand_seed = 0)
{
    static Allocator allocator(MAX_ALLOC_SIZE);
    static CacheFlush cache_flush(CACHE_FLUSH_SIZE);

    std::mt19937 rng(rand_seed);

    static int64_t testcase_count = 0;
    std::cout << "Running on Testcase " << (++testcase_count) << ":" << std::endl;

    std::cout << "--------- Parameters ---------- " << std::endl;
    std::cout << std::boolalpha;
    std::cout << "batch_size             = " << p.batch_size << std::endl;
    std::cout << "seqlen_q               = " << p.seqlen_q << std::endl;
    std::cout << "num_heads_q            = " << p.num_heads_q << std::endl;
    std::cout << "seqlen_kv              = " << p.seqlen_kv << std::endl;
    std::cout << "page_block_size        = " << p.page_block_size << std::endl;
    std::cout << "topk                   = " << p.topk << std::endl;
    std::cout << "extra_seqlen_kv        = " << p.extra_seqlen_kv << std::endl;
    std::cout << "extra_page_block_size  = " << p.extra_page_block_size << std::endl;
    std::cout << "extra_topk             = " << p.extra_topk << std::endl;
    std::cout << "have_topk_length       = " << p.have_topk_length << std::endl;
    std::cout << "have_extra_topk_length = " << p.have_extra_topk_length << std::endl;
    std::cout << "have_attn_sink         = " << p.have_attn_sink << std::endl;
    std::cout << "is_indices_shuffle     = " << p.is_indices_shuffle << std::endl;

    allocator.reset();

    auto d = generate_testcase_data(p, allocator, rng);

    int64_t flash_mla_extra_size;
    flash_mla_sparse_decode_sched(
        p.batch_size,
        p.seqlen_q,
        p.num_heads_q,
        p.head_dim,
        p.head_dim_v,
        p.topk,
        p.extra_topk,
        d.topk_length,
        d.extra_topk_length,
        flash_mla_extra_size,
        flash_mla_meta);

    void *flash_mla_extra_buffer = allocator.alloc(flash_mla_extra_size);

    auto o = create_tensor<bfloat16_t, 4>(allocator, {p.batch_size, p.seqlen_q, p.num_heads_q, p.head_dim_v});
    auto lse = create_tensor<float, 3>(allocator, {p.batch_size, p.seqlen_q, p.num_heads_q});

    auto run_flash_mla = [&]() {
        flash_mla_sparse_decode(
            d.q,
            d.kvcache,
            d.indices,
            d.topk_length,
            d.extra_kvcache,
            d.extra_indices,
            d.extra_topk_length,
            d.attn_sink,
            o,
            lse,
            d.softmax_scale,
            flash_mla_extra_buffer,
            flash_mla_meta);
    };

    run_flash_mla();

    double total_time_usage_us = 0;
    for (int i = 0; i < p.num_runs; ++i) {
        cache_flush();
        auto begin = get_clock_us();
        run_flash_mla();
        auto end = get_clock_us();
        total_time_usage_us += end - begin;
    }

    if (p.check_correctness) {
        auto ref_o = create_tensor<bfloat16_t, 4>(allocator, {p.batch_size, p.seqlen_q, p.num_heads_q, p.head_dim_v});
        auto ref_lse = create_tensor<float, 3>(allocator, {p.batch_size, p.seqlen_q, p.num_heads_q});
        ref_mla_sparse_decode(p, d, ref_o, ref_lse, allocator);

        bool is_o_correct = check_is_allclose(o, ref_o, 1e-3, 2.01 / 128, 5e-6);
        bool is_lse_correct = check_is_allclose(lse, ref_lse, 1e-6, 8.01 / 65536);

        FLASH_ASSERT(is_o_correct && is_lse_correct);
    }

    if (p.num_runs > 0) {
        auto avg_time_usage_us = total_time_usage_us / p.num_runs;
        auto [flop, mem_vol] = count_flop_and_mem_vol(p, d);
        std::cout << "--------- Performance ---------" << std::endl;
        std::cout << std::right;
        std::cout << "Time (us):" << std::setw(21) << avg_time_usage_us << std::endl;
        std::cout << "TFLOPS:   " << std::setw(21) << flop / avg_time_usage_us / 1e6 << std::endl;
        std::cout << "GB/s:     " << std::setw(21) << mem_vol / avg_time_usage_us / 1e3 << std::endl;
    }

    std::cout << "===============================" << std::endl;
}

int main()
{
    kutacc::global_parallel_launch([&]() {
        flash_mla_meta_create(flash_mla_meta);

        auto test_cases = gen_testcase();

        std::cout << "Total Number of Testcases: " << test_cases.size() << std::endl;
        std::cout << "===============================" << std::endl;
        for (const auto &p : test_cases) {
            test_flash_mla(p);
        }

        flash_mla_meta_destory(flash_mla_meta);
    });
    return 0;
}