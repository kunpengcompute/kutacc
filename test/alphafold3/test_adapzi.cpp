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

void test_adap_zi(std::vector<int64_t> &cas, int batch, int num_channels, int hidden_size, int hidden_size_2)
{

    std::mt19937 rnd(time(0));

    int samples = cas[0];
    int bucket = cas[1];

    int T = 50;

    std::cout << "test adaptive_zi of " << samples << " samples, " << bucket << " buckets" << std::endl;

    int x_size = samples*batch*bucket*hidden_size;
    int w_size = hidden_size * num_channels;
    int sc_size = batch*bucket*hidden_size_2;
    int w2_size = hidden_size_2 * num_channels;
    int b_size = num_channels;
    int temp_out_size = batch*bucket*num_channels;
    int output_size = samples*batch*bucket*num_channels;

    std::unique_ptr<float[]> x(new float[x_size]);
    std::unique_ptr<float[]> s_c(new float[sc_size]);
    std::unique_ptr<float[]> w(new float[w_size]);
    std::unique_ptr<float[]> w2(new float[w2_size]);
    std::unique_ptr<float[]> b(new float[b_size]);

    std::unique_ptr<bfloat16_t[]> x_bf16(new bfloat16_t[x_size]);
    std::unique_ptr<bfloat16_t[]> s_c_bf16(new bfloat16_t[sc_size]);
    std::unique_ptr<bfloat16_t[]> w_bf16(new bfloat16_t[w_size]);
    std::unique_ptr<bfloat16_t[]> w2_bf16(new bfloat16_t[w2_size]);
    std::unique_ptr<bfloat16_t[]> output_kutacc_bf16(new bfloat16_t[output_size]);

    std::unique_ptr<float[]> temp_x_norm(new float[output_size]);
    std::unique_ptr<float[]> temp_sc_norm(new float[temp_out_size]);
    std::unique_ptr<float[]> temp_sc_sigmoid(new float[temp_out_size]);
    std::unique_ptr<float[]> output_norm(new float[output_size]);

    std::unique_ptr<float[]> temp_out(new float[output_size]);
    std::unique_ptr<float[]> output_kutacc_fp32(new float[output_size]);


    for(int i = 0; i < x_size; ++i){
        x[i] = rnd() * 1.0 / rnd.max();
    }

    for(int i = 0; i < sc_size; ++i){
        s_c[i] = rnd() * 1.0 / rnd.max();
    }
    
    for(int i = 0; i < w_size; ++i){
        w[i] = rnd() * 1.0 / rnd.max();
    }

    for(int i = 0; i < b_size; ++i){
        b[i] = rnd() * 1.0 / rnd.max();
    }
    cvt_fp32_bf16(x.get(), x_bf16.get(), x_size);
    cvt_fp32_bf16(s_c.get(), s_c_bf16.get(), sc_size);

    matmul_3d_2d(x.get(), w.get(), temp_x_norm.get(), samples*batch, bucket, num_channels, hidden_size);
    matmul_3d_2d(s_c.get(), w2.get(), temp_sc_norm.get(), batch, bucket, num_channels, hidden_size_2);
    add_bias_broadcast(temp_sc_norm.get(), b.get(), batch*bucket, num_channels);
    sigmoid_manual(temp_sc_norm.get(), temp_sc_sigmoid.get(), temp_out_size);
    mul_broadcast(temp_x_norm.get(), temp_sc_sigmoid.get(), output_norm.get(), samples, temp_out_size);

    pack_weight_task_bf16(w.get(), w_bf16.get(), hidden_size, num_channels);
    pack_weight_task_bf16(w2.get(), w2_bf16.get(), hidden_size_2, num_channels);
    kutacc::linear_nobias(x_bf16.get(), w_bf16.get(), temp_out.get(), samples*batch*bucket, num_channels, hidden_size, false);
    kutacc::linear_bias_sigmoid(s_c_bf16.get(), w2_bf16.get(), b.get(), temp_out.get(), output_kutacc_bf16.get(), samples, batch*bucket, num_channels, hidden_size_2);
    cvt_bf16_fp32(output_kutacc_bf16.get(), output_kutacc_fp32.get(), output_size);

    for(int i = 0; i < output_size; i++){
        float out = output_kutacc_fp32[i];
        float expect = output_norm[i];
        float max_diff = std::max(expect*0.1, 0.1);
        if (std::abs(out - expect) > max_diff) {
            std::cerr << "(" << i << ") is " << out << ", expect " << expect << ", "
                    << "\n";
            exit(1);
        }
    }

    for(int t = 0; t<20; ++t)
    {
        kutacc::linear_nobias(x_bf16.get(), w_bf16.get(), temp_out.get(), samples*batch*bucket, num_channels, hidden_size, false);
        kutacc::linear_bias_sigmoid(s_c_bf16.get(), w2_bf16.get(), b.get(), temp_out.get(), output_kutacc_bf16.get(), samples, batch*bucket, num_channels, hidden_size_2);
    }
    auto start_o_st = std::chrono::steady_clock::now();
    for(int t = 0; t<T; ++t)
    {
        kutacc::linear_nobias(x_bf16.get(), w_bf16.get(), temp_out.get(), samples*batch*bucket, num_channels, hidden_size, false);
        kutacc::linear_bias_sigmoid(s_c_bf16.get(), w2_bf16.get(), b.get(), temp_out.get(), output_kutacc_bf16.get(), samples, batch*bucket, num_channels, hidden_size_2);
    }
    auto end_o_st = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapse_o_store = end_o_st - start_o_st;
    std::cout << "adap_zi time = " << elapse_o_store.count()/T << std::endl;

}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases;
        read_args_skip_n(cases, 2, argc, argv, 4);
        int batch = std::stoi(argv[1]);
        int num_channels = std::stoi(argv[2]);
        int hidden_size = std::stoi(argv[3]);
        int hidden_size_2 = std::stoi(argv[4]);
        for (auto cas : cases) {
            test_adap_zi(cas, batch, num_channels, hidden_size, hidden_size_2);
        }
    });
}
