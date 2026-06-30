#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>
#include <cstring>
#include <random>

#include "helper.h"
#include "kutacc.h"
#include "operator_norm.h"

#define SVL_fp32 16
#define SVL_bf16 32
#define SME_ON()                                                                                \
    __asm__ volatile("SMSTART":::"z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",                \
                                 "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",          \
                                 "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",        \
                                 "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",        \
                                 "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",                \
                                 "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15");         \
    __asm__ volatile("ISB")
#define SME_OFF()                                                                               \
    __asm__ volatile("SMSTOP":::"z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",                 \
                                "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",           \
                                "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",         \
                                "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",         \
                                "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",                 \
                                "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15");          \
    __asm__ volatile("ISB")

void pack_left_task_bf16(const bfloat16_t *matrix,bfloat16_t *matrix_pack,int M, int N)
{
    svbool_t pt16 = svptrue_b16();
    int max_threads = kutacc::get_thread_num();
    const int row_step = SVL_bf16;
    const int col_step = 2 * SVL_bf16;
    const int one_task = row_step * col_step;
    int all_task_num = (M * N) / one_task;
    int avg_task = all_task_num / max_threads;
    int busy_thread = all_task_num - avg_task * max_threads;
    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
        int tid = kutacc::get_thread_id();
        int task_num, start_idx;
        if(tid < busy_thread)
        {
            task_num = avg_task+1;
            start_idx = tid*task_num;
        }
        else
        {
            task_num = avg_task;
            start_idx = busy_thread*(task_num+1)+(tid-busy_thread)*task_num;
        }

        int row_task_num = N / col_step;
        SME_ON();
        for(int task_idx = start_idx; task_idx < start_idx + task_num; task_idx += 1)
        {
            int outer_row_index = (task_idx / row_task_num) * row_step;
            int outer_col_index = (task_idx % row_task_num) * col_step;
            svzero_za();
            for (int row_index = 0; row_index < SVL_bf16; row_index += 1)
            {
                svld1_hor_za16(0, static_cast<uint32_t>(row_index), pt16, matrix + (outer_row_index + row_index) * N + outer_col_index);
                svld1_hor_za16(1, static_cast<uint32_t>(row_index), pt16, matrix + (outer_row_index + row_index) * N + outer_col_index + SVL_bf16); 

            }
            #pragma unroll(8)
            for (int col_index = 0; col_index < SVL_bf16; col_index += 2) 
            {
                svbfloat16_t unz1,unz2,z1,z2;
                unz1 = svread_ver_za16_bf16_m(unz1, pt16, 0, static_cast<uint32_t>(col_index));  
                unz2 = svread_ver_za16_bf16_m(unz2, pt16, 0, static_cast<uint32_t>(col_index+1)); 

                z1 = svzip1_bf16(unz1, unz2); 
                z2 = svzip2_bf16(unz1, unz2); 
                svst1_bf16(pt16,matrix_pack + outer_row_index*N + (col_index+ outer_col_index)*SVL_fp32, z1); 
                svst1_bf16(pt16,matrix_pack + (outer_row_index + SVL_fp32)*N + (col_index+ outer_col_index)*SVL_fp32, z2);

                unz1 = svread_ver_za16_bf16_m(unz1, pt16, 1, static_cast<uint32_t>(col_index));  
                unz2 = svread_ver_za16_bf16_m(unz2, pt16, 1, static_cast<uint32_t>(col_index+1)); 

                z1 = svzip1_bf16(unz1,unz2);
                z2 = svzip2_bf16(unz1,unz2);
                svst1_bf16(pt16,matrix_pack + outer_row_index *N + (col_index+ outer_col_index + SVL_bf16)*SVL_fp32 , z1); 
                svst1_bf16(pt16,matrix_pack + (outer_row_index + SVL_fp32)*N + (col_index+ outer_col_index + SVL_bf16)*SVL_fp32, z2);
            }
        }
        SME_OFF();
    });
}

void test_bgemm(std::vector<int64_t> &cas)
{

    std::mt19937 rnd(time(0));
    int samples = cas[0];
    int bucket = cas[1];
    int nheads = 16;
    int head_size = 48;
    int num_channels = 768;
    int size_x = samples * bucket * num_channels;
    int size_weight = num_channels * num_channels;

    int T = 50;

    std::cout << "test bgemm of " << samples << " samples " << bucket << " buckets" << std::endl;

    std::unique_ptr<float[]> x_fp32(new float[size_x]);
    std::unique_ptr<float[]> q_proj(new float[size_weight]);
    std::unique_ptr<float[]> q_bias(new float[num_channels]);
    std::unique_ptr<float[]> k_proj(new float[size_weight]);
    std::unique_ptr<float[]> v_proj(new float[size_weight]);
    std::unique_ptr<float[]> g_proj(new float[size_weight]);
    std::unique_ptr<float[]> q(new float[size_x]);
    std::unique_ptr<float[]> k(new float[size_x]);
    std::unique_ptr<float[]> v(new float[size_x]);
    std::unique_ptr<float[]> g(new float[size_x]);
    std::unique_ptr<float[]> q_transpose(new float[size_x]);
    std::unique_ptr<float[]> k_transpose(new float[size_x]);
    std::unique_ptr<float[]> v_transpose(new float[size_x]);
    std::unique_ptr<float[]> q_kutacc(new float[size_x]);
    std::unique_ptr<float[]> k_kutacc(new float[size_x]);
    std::unique_ptr<float[]> v_kutacc(new float[size_x]);
    std::unique_ptr<float[]> g_kutacc(new float[size_x]);


    std::unique_ptr<bfloat16_t[]> x_bf16(new bfloat16_t[size_x]);
    std::unique_ptr<bfloat16_t[]> pack_x_bf16(new bfloat16_t[size_x]);
    std::unique_ptr<bfloat16_t[]> q_proj_bf16(new bfloat16_t[size_weight]);
    std::unique_ptr<bfloat16_t[]> k_proj_bf16(new bfloat16_t[size_weight]);
    std::unique_ptr<bfloat16_t[]> v_proj_bf16(new bfloat16_t[size_weight]);
    std::unique_ptr<bfloat16_t[]> g_proj_bf16(new bfloat16_t[size_weight]);
    std::unique_ptr<bfloat16_t[]> q_bf16(new bfloat16_t[size_x]);
    std::unique_ptr<bfloat16_t[]> k_bf16(new bfloat16_t[size_x]);
    std::unique_ptr<bfloat16_t[]> v_bf16(new bfloat16_t[size_x]);
    std::unique_ptr<bfloat16_t[]> g_bf16(new bfloat16_t[size_x]);

    for(int i = 0; i < size_x; ++i){
        x_bf16[i] = rnd() * 1.0 / rnd.max();
    }

    for(int i = 0; i < size_weight; ++i){
        q_proj_bf16[i] = rnd() * 1.0 / rnd.max();
        k_proj_bf16[i] = rnd() * 1.0 / rnd.max();
        v_proj_bf16[i] = rnd() * 1.0 / rnd.max();
        g_proj_bf16[i] = rnd() * 1.0 / rnd.max();
    }
    for(int i = 0; i < num_channels; ++i){
        q_bias[i] = rnd() * 1.0 / rnd.max();
    }

    cvt_bf16_fp32(x_bf16.get(), x_fp32.get(), size_x);
    cvt_bf16_fp32(q_proj_bf16.get(), q_proj.get(), size_weight);
    cvt_bf16_fp32(k_proj_bf16.get(), k_proj.get(), size_weight);
    cvt_bf16_fp32(v_proj_bf16.get(), v_proj.get(), size_weight);
    cvt_bf16_fp32(g_proj_bf16.get(), g_proj.get(), size_weight);

    matmul_3d_2d(x_fp32.get(),q_proj.get(),q.get(),samples, bucket, num_channels, num_channels);
    add_bias_broadcast(q.get(), q_bias.get(), samples*bucket, num_channels);
    matmul_3d_2d(x_fp32.get(),k_proj.get(),k.get(),samples, bucket, num_channels, num_channels);
    matmul_3d_2d(x_fp32.get(),v_proj.get(),v.get(),samples, bucket, num_channels, num_channels);
    matmul_3d_2d(x_fp32.get(),g_proj.get(),g.get(),samples, bucket, num_channels, num_channels);
    transpose_BNHD_to_BHND_sve(q.get(), q_transpose.get(), samples, bucket, nheads, head_size);
    transpose_BNHD_to_BHND_sve(k.get(), k_transpose.get(), samples, bucket, nheads, head_size);
    transpose_BNHD_to_BHND_sve(v.get(), v_transpose.get(), samples, bucket, nheads, head_size);

    pack_left_task_bf16(x_bf16.get(), pack_x_bf16.get(), samples*bucket, num_channels);
    kutacc::self_attention_linear_transpose(pack_x_bf16.get(), q_proj_bf16.get(), q_bias.get(), k_proj_bf16.get(), v_proj_bf16.get(), g_proj_bf16.get(),\
    q_bf16.get(), k_bf16.get(), v_bf16.get(), g_bf16.get(), samples, bucket, nheads, head_size);

    cvt_bf16_fp32(q_bf16.get(), q_kutacc.get(), size_x);
    cvt_bf16_fp32(k_bf16.get(), k_kutacc.get(), size_x);
    cvt_bf16_fp32(v_bf16.get(), v_kutacc.get(), size_x);
    cvt_bf16_fp32(g_bf16.get(), g_kutacc.get(), size_x);

    for(int i = 0; i < size_x; i++){
        float out = q_kutacc[i];
        float expect = q_transpose[i];
        float max_diff = expect*0.2;
        if (std::abs(out - expect) > max_diff) {
            std::cerr << "q " << "(" << i << ") is " << out << ", expect " << expect << ", "
                    << "\n";
            exit(1);
        }
    }
    for(int i = 0; i < size_x; i++){
        float out = k_kutacc[i];
        float expect = k_transpose[i];
        float max_diff = expect*0.2;
        if (std::abs(out - expect) > max_diff) {
            std::cerr << "k " << "(" << i << ") is " << out << ", expect " << expect << ", "
                    << "\n";
            exit(1);
        }
    }
    for(int i = 0; i < size_x; i++){
        float out = v_kutacc[i];
        float expect = v_transpose[i];
        float max_diff = expect*0.2;
        if (std::abs(out - expect) > max_diff) {
            std::cerr << "v " << "(" << i << ") is " << out << ", expect " << expect << ", "
                    << "\n";
            exit(1);
        }
    }
    for(int i = 0; i < size_x; i++){
        float out = g_kutacc[i];
        float expect = g[i];
        float max_diff = expect*0.2;
        if (std::abs(out - expect) > max_diff) {
            std::cerr << "g " << "(" << i << ") is " << out << ", expect " << expect << ", "
                    << "\n";
            exit(1);
        }
    }

    for(int t = 0; t<20; ++t)
    {
        kutacc::self_attention_linear_transpose(pack_x_bf16.get(), q_proj_bf16.get(), q_bias.get(), k_proj_bf16.get(), v_proj_bf16.get(), g_proj_bf16.get(),\
        q_bf16.get(), k_bf16.get(), v_bf16.get(), g_bf16.get(), samples, bucket, nheads, head_size);
    }
    auto start_o_st = std::chrono::steady_clock::now();
    for(int t = 0; t<T; ++t)
    {
        kutacc::self_attention_linear_transpose(pack_x_bf16.get(), q_proj_bf16.get(), q_bias.get(), k_proj_bf16.get(), v_proj_bf16.get(), g_proj_bf16.get(),\
        q_bf16.get(), k_bf16.get(), v_bf16.get(), g_bf16.get(), samples, bucket, nheads, head_size);
    }
    auto end_o_st = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapse_o_store = end_o_st - start_o_st;
    std::cout << "bgemm time = " << elapse_o_store.count()/T << std::endl;
}
int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases;
        read_args(cases, 2, argc, argv);
        for (auto cas : cases) {
            test_bgemm(cas);
        }
    });
}
