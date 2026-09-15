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
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "check.h"
#include "kutacc.h"
#include "scalar_type.h"
#include "span.h"
#include "tensor.h"

inline int get_idx(int token_idx, int head_idx, int dim_idx, int num_heads, int head_dim)
{
    return token_idx * (num_heads * head_dim) + head_idx * head_dim + dim_idx;
}

void check_tensor_shape(const Tensor& loaded_tensor, const std::vector<int64_t>& expected_shape,
    const std::string& tensor_name)
{
    std::vector<int64_t> loaded_shape(loaded_tensor.sizes, loaded_tensor.sizes + loaded_tensor.dim);
    if (loaded_shape != expected_shape) {
        std::cerr << "Error: " << tensor_name << " shape mismatch!" << std::endl;
        std::cerr << "Expected: ";
        for (auto s : expected_shape)
            std::cerr << s << " ";
        std::cerr << "\nLoaded: ";
        for (auto s : loaded_shape)
            std::cerr << s << " ";
        std::cerr << std::endl;
        exit(1);
    }
}

template <typename T>
void copy_tensor_data(const Tensor& src, Tensor& dst)
{
    PARAMETER_CHECK(src.numel() == dst.numel(), "Tensor element count mismatch!");
    PARAMETER_CHECK(src.dtype() == dst.dtype(), "Tensor dtype mismatch!");
    std::memcpy(dst.data_ptr<T>(), src.data_ptr<T>(), src.numel() * sizeof(T));
}

__attribute__((noinline)) void multi_head_attention_ref(const __bf16* query, const __bf16* key, const __bf16* value,
    __bf16* output, int batch_size, int seq_len, int qk_head_dim, int vo_head_dim, int num_heads, float scaling_factor,
    bool is_causal)
{
    int n_token = batch_size * seq_len;

    kutacc::parallel_for(0, n_token, 1, [&](uint64_t start, uint64_t end) {
        for (uint64_t token_idx = start; token_idx < end; ++token_idx) {
            int b = token_idx / seq_len;
            int i = token_idx % seq_len;

            for (int h = 0; h < num_heads; ++h) {
                std::vector<float> scores(seq_len, 0.0f);
                float max_score = -std::numeric_limits<float>::infinity();

                for (int j = 0; j < seq_len; ++j) {
                    int key_token_idx = b * seq_len + j;
                    float dot = 0.0f;

                    for (int d = 0; d < qk_head_dim; ++d) {
                        int q_idx = get_idx(token_idx, h, d, num_heads, qk_head_dim);
                        int k_idx = get_idx(key_token_idx, h, d, num_heads, qk_head_dim);
                        dot += static_cast<float>(query[q_idx]) * static_cast<float>(key[k_idx]);
                    }

                    dot *= scaling_factor;

                    if (is_causal && j > i) {
                        scores[j] = -std::numeric_limits<float>::infinity();
                    } else {
                        scores[j] = dot;
                    }

                    if (scores[j] > max_score) {
                        max_score = scores[j];
                    }
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    if (scores[j] == -std::numeric_limits<float>::infinity()) {
                        scores[j] = 0.0f;
                    } else {
                        scores[j] = std::exp(scores[j] - max_score);
                    }
                    sum_exp += scores[j];
                }

                for (int j = 0; j < seq_len; ++j) {
                    scores[j] /= sum_exp;
                }

                for (int d = 0; d < vo_head_dim; ++d) {
                    float val_acc = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        int key_token_idx = b * seq_len + j;
                        int v_idx = get_idx(key_token_idx, h, d, num_heads, vo_head_dim);
                        val_acc += scores[j] * static_cast<float>(value[v_idx]);
                    }

                    int out_idx = get_idx(token_idx, h, d, num_heads, vo_head_dim);
                    output[out_idx] = static_cast<__bf16>(val_acc);
                }
            }
        }
    });
}

static int seed = 42;
std::mt19937 gen(seed);
template <typename T>
void random_initialize(T* data, size_t length, float min = -1.0f, float max = 1.0f)
{
    std::uniform_real_distribution<float> dis(min, max);
    for (size_t i = 0; i < length; ++i) {
        float rand_val = dis(gen);
        data[i] = static_cast<T>(rand_val);
    }
}

template <typename T>
void neg_one_initialize(T* data, size_t length)
{
    for (size_t i = 0; i < length; ++i) {
        data[i] = static_cast<T>(-1.0);
    }
}

void start_loc_init(int32_t* arr, int64_t batch_size, int64_t seq_len)
{
    for (int64_t bi = 0; bi < batch_size + 1; ++bi) {
        arr[bi] = bi * seq_len;
    }
}

bool is_close(float a, float b, float rel_tol = 1e-6, float abs_tol = 1e-6)
{
    return std::abs(a - b) <= std::max(rel_tol * std::max(std::abs(a), std::abs(b)), abs_tol);
}

void chunked_qo_init(std::vector<Tensor>& q_chunked_list, std::vector<Tensor>& q_loc_list,
    std::vector<Tensor>& o_chunked_list, int chunked_size, int64_t num_heads, int64_t qk_head_dim, int64_t vo_head_dim,
    std::vector<int64_t>& seq_lens, Tensor& query, Tensor& query_start_loc, u8span& ddr_available)
{
    auto max_len = *(std::max_element(seq_lens.begin(), seq_lens.end()));
    auto query_start_loc_data = query_start_loc.data_ptr<int32_t>();
    auto query_data = query.data_ptr<bfloat16_t>();

    auto cur_chunk_start = 0;
    for (int chunked_loop = 0; chunked_loop < max_len; chunked_loop += chunked_size, cur_chunk_start += chunked_size) {
        // std::cerr<<"  chunked_loop="<<chunked_loop<<"  chunked_size="<<chunked_size;
        int chunked_size_sum = 0;
        std::vector<int64_t> len_recorder;
        for (int i = 0; i < query_start_loc.size(0) - 1; i++) {
            auto chunked_cur_len = std::max(0,
                std::min(chunked_size, query_start_loc_data[i + 1] - query_start_loc_data[i] - cur_chunk_start));
            // std::cerr<<"  i="<<i<<"   cur_chunk_start="<<cur_chunk_start;
            // std::cerr<<"  i="<<i<<"   query_start_loc_data[i]="<<query_start_loc_data[i];
            // std::cerr<<"  i="<<i<<"   query_start_loc_data[i + 1]="<<query_start_loc_data[i + 1];
            // std::cerr<<"  i="<<i<<"   chunked_cur_len "<<chunked_cur_len<<"\n";
            chunked_size_sum += chunked_cur_len;
            len_recorder.push_back(chunked_cur_len);
        }
        Tensor cur_chunk_output =
            Tensor::alloc_from(ScalarType::BFloat16, {chunked_size_sum, num_heads, vo_head_dim}, ddr_available);
        o_chunked_list.push_back(cur_chunk_output);

        Tensor cur_chunk_query_loc = Tensor::alloc_from(ScalarType::Int32, {query_start_loc.size(0)}, ddr_available);
        auto cur_chunk_query_loc_data = cur_chunk_query_loc.data_ptr<int32_t>();
        cur_chunk_query_loc_data[0] = 0;
        for (int i = 0; i < query_start_loc.size(0) - 1; i++) {
            cur_chunk_query_loc_data[i + 1] = cur_chunk_query_loc_data[i] + len_recorder[i];
            // std::cerr<<"  i="<<i<<"   cur_chunk_query_loc_data[i + 1]="<<cur_chunk_query_loc_data[i + 1]<<"\n";
        }

        Tensor cur_chunk_query =
            Tensor::alloc_from(ScalarType::BFloat16, {chunked_size_sum, num_heads, qk_head_dim}, ddr_available);
        auto cur_chunk_query_data = cur_chunk_query.data_ptr<bfloat16_t>();
        for (int i = 0; i < query_start_loc.size(0) - 1; i++) {
            // std::cerr<<"  i="<<i<<"   to="<<cur_chunk_query_loc_data[i]<<"   from="<<query_start_loc_data[i] +
            // cur_chunk_start<<"   len="<<len_recorder[i]<<"\n";
            memcpy(cur_chunk_query_data + cur_chunk_query_loc_data[i] * num_heads * qk_head_dim,
                query_data + (query_start_loc_data[i] + cur_chunk_start) * num_heads * qk_head_dim,
                len_recorder[i] * num_heads * qk_head_dim * sizeof(bfloat16_t));
        }

        q_chunked_list.push_back(cur_chunk_query);
        q_loc_list.push_back(cur_chunk_query_loc);
    }
}

void output_copy_back(std::vector<Tensor>& o_chunked_list, Tensor& output, std::vector<Tensor>& q_loc_list,
    Tensor& q_loc_full, int chunked_size, int64_t num_heads, int64_t vo_head_dim)
{
    auto output_data = output.data_ptr<bfloat16_t>();
    auto q_loc_full_data = q_loc_full.data_ptr<int32_t>();
    auto cur_chunk_start = 0;
    for (size_t chunk_id = 0; chunk_id < q_loc_list.size(); chunk_id++, cur_chunk_start += chunked_size) {
        // std::cerr<<"  chunk_id="<<chunk_id<<"   q_loc_list.size()="<<q_loc_list.size()<<"\n";
        auto cur_q_loc_data = q_loc_list[chunk_id].data_ptr<int32_t>();
        auto cur_output_data = o_chunked_list[chunk_id].data_ptr<bfloat16_t>();
        for (auto i = 0; i < q_loc_list[chunk_id].size(0) - 1; i++) {
            // std::cerr<<"  batch_id="<<i<<"   q_loc_list[chunk_id].size(0) - 1="<<q_loc_list[chunk_id].size(0) -
            // 1<<"\n"; std::cerr<<"  q_loc_full_data[i] + cur_chunk_start="<<q_loc_full_data[i] + cur_chunk_start<<"
            // cur_q_loc_data[i]="<<cur_q_loc_data[i]<<"   (cur_q_loc_data[i + 1] -
            // cur_q_loc_data[i])="<<(cur_q_loc_data[i + 1] - cur_q_loc_data[i])<<"\n";
            memcpy(output_data + (q_loc_full_data[i] + cur_chunk_start) * num_heads * vo_head_dim,
                cur_output_data + cur_q_loc_data[i] * num_heads * vo_head_dim,
                (cur_q_loc_data[i + 1] - cur_q_loc_data[i]) * num_heads * vo_head_dim * sizeof(bfloat16_t));
        }
    }
}

template <bool on_package = true>
void* mmap_huge_page_memory(int64_t size)
{
    constexpr int64_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;
    PARAMETER_CHECK(size % HUGE_PAGE_SIZE == 0, " size = ", size, " HUGE_PAGE_SIZE = ", HUGE_PAGE_SIZE);
    void* addr =
        mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    int64_t cpu = sched_getcpu();
    int64_t rnode = numa_node_of_cpu(cpu) + (on_package ? 16 : 0);
    unsigned long mask = 1UL << rnode;
    mbind(addr, size, MPOL_BIND, &mask, sizeof(mask) * 8, MPOL_MF_STRICT | MPOL_MF_MOVE);
    return addr;
}

template <typename dtype, int64_t dim>
kutacc::Tensor<dtype, dim> convert_to_kutacc_tensor(const Tensor& x)
{
    return kutacc::Tensor<dtype, dim>(x.data_ptr<dtype>(), x.sizes, x.strides);
}

int64_t num_heads = 16;
int64_t qk_head_dim = 192;
int64_t vo_head_dim = 128;
int64_t seq_len = 32;
int64_t chunked_size = 512;
int64_t batch_size = 1;

int main(int argc, char *argv[])
{
    if (argc < 7 && argc != 1) {
        std::cerr << "用法: " << argv[0]
                  << " <num_heads> <qk_head_dim> <vo_head_dim> <seq_len> <chunked_size> <batch_size>"
                  << std::endl;
        return 1;
    }
    auto parse_int64 = [](const char* str, const char* name) -> int64_t {
        try {
            return std::stoll(str);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("参数 \"") + name + "\" 的值 \"" + str + "\" 不是有效的整数");
        }
    };
    if (argc == 7) {
        try {
            num_heads     = parse_int64(argv[1], "num_heads");
            qk_head_dim   = parse_int64(argv[2], "qk_head_dim");
            vo_head_dim   = parse_int64(argv[3], "vo_head_dim");
            seq_len       = parse_int64(argv[4], "seq_len");
            chunked_size  = parse_int64(argv[5], "chunked_size");
            batch_size    = parse_int64(argv[6], "batch_size");
        } catch (const std::runtime_error& e) {
            std::cerr << "错误: " << e.what() << std::endl;
            return 1;
        }
    }

    kutacc::global_parallel_launch([&] {
        float softmax_scale = 1.0 / std::sqrt(qk_head_dim);
        int64_t max_seq_len = (seq_len + 1023) / 1024 * 1024 + 100;
        bool kv_is_packed = batch_size == 1;
        int64_t n_token = batch_size * seq_len;
        auto threads_num = kutacc::get_thread_num();
        auto br = 128;
        auto bc = 128;

        size_t ddr_size = 32ULL * 1024ULL * 1024ULL * 1024ULL;
        std::unique_ptr<uint8_t[]> mem(new uint8_t[ddr_size]);
        u8span ddr_available{mem.get(), ddr_size};

        Tensor query = Tensor::alloc_from(ScalarType::BFloat16, {n_token, num_heads, qk_head_dim}, ddr_available);
        Tensor key = Tensor::alloc_from(ScalarType::BFloat16, {n_token, num_heads, qk_head_dim}, ddr_available);
        Tensor value = Tensor::alloc_from(ScalarType::BFloat16, {n_token, num_heads, vo_head_dim}, ddr_available);
        Tensor query_start_loc = Tensor::alloc_from(ScalarType::Int32, {batch_size + 1}, ddr_available);
        Tensor key_start_loc = Tensor::alloc_from(ScalarType::Int32, {batch_size + 1}, ddr_available);

        Tensor attn_out_get =
            Tensor::alloc_from(ScalarType::BFloat16, {n_token, num_heads, vo_head_dim}, ddr_available);
        Tensor attn_out_expect =
            Tensor::alloc_from(ScalarType::BFloat16, {n_token, num_heads, vo_head_dim}, ddr_available);

        auto pack_attn_k =
            kv_is_packed
                ? Tensor::alloc_from(ScalarType::BFloat16, {max_seq_len, num_heads, qk_head_dim}, ddr_available)
                : Tensor::alloc_from(ScalarType::BFloat16, {threads_num, n_token, qk_head_dim}, ddr_available);
        auto pack_attn_v =
            kv_is_packed
                ? Tensor::alloc_from(ScalarType::BFloat16, {max_seq_len, num_heads, vo_head_dim}, ddr_available)
                : Tensor::alloc_from(ScalarType::BFloat16, {threads_num, n_token, vo_head_dim}, ddr_available);

        auto pack_attn_q = Tensor::alloc_from(ScalarType::BFloat16, {threads_num, br * qk_head_dim}, ddr_available);

        auto attn_s = Tensor::alloc_from(ScalarType::Float32, {threads_num, bc * br}, ddr_available);
        auto attn_out_block_old =
            Tensor::alloc_from(ScalarType::Float32, {threads_num, br, vo_head_dim}, ddr_available);
        auto attn_out_block_new =
            Tensor::alloc_from(ScalarType::Float32, {threads_num, br, vo_head_dim}, ddr_available);
        auto attn_max_block_old = Tensor::alloc_from(ScalarType::Float32, {threads_num, br}, ddr_available);
        auto attn_max_block_new = Tensor::alloc_from(ScalarType::Float32, {threads_num, br}, ddr_available);
        auto attn_base_block_old = Tensor::alloc_from(ScalarType::Float32, {threads_num, br}, ddr_available);
        auto attn_base_block_new = Tensor::alloc_from(ScalarType::Float32, {threads_num, br}, ddr_available);

        std::vector<int64_t> cur_lens(batch_size, seq_len);
        std::vector<int64_t> seq_lens(batch_size, seq_len);

        random_initialize(query.data_ptr<__bf16>(), query.numel());
        random_initialize(key.data_ptr<__bf16>(), key.numel());
        random_initialize(value.data_ptr<__bf16>(), value.numel());
        neg_one_initialize(attn_out_get.data_ptr<__bf16>(), attn_out_get.numel());
        neg_one_initialize(attn_out_expect.data_ptr<__bf16>(), attn_out_expect.numel());
        start_loc_init(query_start_loc.data_ptr<int32_t>(), batch_size, seq_len);
        start_loc_init(key_start_loc.data_ptr<int32_t>(), batch_size, seq_len);

        std::vector<Tensor> q_chunked_list;
        std::vector<Tensor> o_chunked_list;
        std::vector<Tensor> q_loc_list;
        chunked_qo_init(q_chunked_list, q_loc_list, o_chunked_list, chunked_size, num_heads, qk_head_dim, vo_head_dim,
            seq_lens, query, query_start_loc, ddr_available);

        if (kv_is_packed) {
            kutacc::flash_attention_v_block_pack(n_token, num_heads, vo_head_dim, max_seq_len, value.stride(0),
                value.stride(1), value.data_ptr<bfloat16_t>(), pack_attn_v.data_ptr<bfloat16_t>());
            kutacc::flash_attention_k_block_pack(n_token, num_heads, qk_head_dim, max_seq_len, key.stride(0),
                key.stride(1), key.data_ptr<bfloat16_t>(), pack_attn_k.data_ptr<bfloat16_t>());
        }
        auto& para_k = kv_is_packed ? pack_attn_k : key;
        auto& para_v = kv_is_packed ? pack_attn_v : value;
        for (auto chunked_loop = 0; chunked_loop < seq_len; chunked_loop += chunked_size) {
            auto chunk_id = chunked_loop / chunked_size;
            seq_lens = std::vector<int64_t>(batch_size, chunked_loop + seq_len);
            kutacc::flash_attention(convert_to_kutacc_tensor<bfloat16_t, 3>(q_chunked_list[chunk_id]),
                convert_to_kutacc_tensor<bfloat16_t, 3>(para_k), convert_to_kutacc_tensor<bfloat16_t, 3>(para_v),
                convert_to_kutacc_tensor<bfloat16_t, 3>(o_chunked_list[chunk_id]),
                convert_to_kutacc_tensor<bfloat16_t, 2>(pack_attn_q),
                convert_to_kutacc_tensor<bfloat16_t, 3>(pack_attn_k),
                convert_to_kutacc_tensor<bfloat16_t, 3>(pack_attn_v), convert_to_kutacc_tensor<float, 2>(attn_s),
                convert_to_kutacc_tensor<float, 3>(attn_out_block_old),
                convert_to_kutacc_tensor<float, 3>(attn_out_block_new),
                convert_to_kutacc_tensor<float, 2>(attn_max_block_old),
                convert_to_kutacc_tensor<float, 2>(attn_max_block_new),
                convert_to_kutacc_tensor<float, 2>(attn_base_block_old),
                convert_to_kutacc_tensor<float, 2>(attn_base_block_new), 1, softmax_scale,
                convert_to_kutacc_tensor<int, 1>(q_loc_list[chunk_id]), convert_to_kutacc_tensor<int, 1>(key_start_loc),
                chunked_size, seq_lens, cur_lens, kv_is_packed);
        }
        output_copy_back(o_chunked_list, attn_out_get, q_loc_list, query_start_loc, chunked_size, num_heads,
            vo_head_dim);

        multi_head_attention_ref(query.data_ptr<__bf16>(), key.data_ptr<__bf16>(), value.data_ptr<__bf16>(),
            attn_out_expect.data_ptr<__bf16>(), batch_size, seq_len, qk_head_dim, vo_head_dim, num_heads, softmax_scale,
            true);

        __bf16* get_ptr = attn_out_get.data_ptr<__bf16>();
        __bf16* expect_ptr = attn_out_expect.data_ptr<__bf16>();
        int64_t as0 = attn_out_get.strides[0];
        int64_t as1 = attn_out_get.strides[1];
        int64_t as2 = attn_out_get.strides[2];

        double RTOL = 1e-2;
        double ATOL = 1e-2;
        int errors = 0;

        for (int64_t ti = 0; ti < n_token; ti++) {
            for (int64_t hi = 0; hi < num_heads; hi++) {
                for (int64_t ei = 0; ei < vo_head_dim; ei++) {
                    float ref = float(expect_ptr[ti * as0 + hi * as1 + ei * as2]);
                    float test = float(get_ptr[ti * as0 + hi * as1 + ei * as2]);

                    if (!is_close(ref, test, RTOL, ATOL)) {
                        if (errors < 20) {
                            std::cout << "Mismatch at index ti = " << ti << ", hi = " << hi << ", ei = " << ei
                                      << ": Ref = " << ref << ", Test = " << test << std::endl;
                        }
                        errors++;
                    }
                }
            }
        }

        if (errors == 0) {
            std::cout << "SUCCESS: DeepSeek MHA (qk_head_dim=" << qk_head_dim << ", vo_head_dim=" << vo_head_dim
                      << ") passed." << std::endl;
        } else {
            std::cout << "FAILURE: " << errors << " mismatches found." << std::endl;
            exit(1);
        }
    });
    return 0;
}
