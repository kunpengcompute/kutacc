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

void test_adap_ln(std::vector<int64_t> &cas, int D0, int D1, int D2)
{

    std::mt19937 rnd(time(0));

    int N_s = cas[0];
    int B = cas[1];

    int T = 50;
    float eps = 1e-5;

    std::cout << "test adaptive_layernorm of " << N_s << " samples, " << B << "bucket" << std::endl;

    int x_size = N_s*B*D0*D1;
    int s_size = B*D0*D2;
    int s_temp_size = B*D0*D1;
    int d_size = D1;
    int dd_size = D2 * D1;

    std::unique_ptr<float[]> x(new float[x_size]);
    std::unique_ptr<float[]> s_c(new float[s_size]);
    std::unique_ptr<float[]> ln_weight(new float[d_size]);
    std::unique_ptr<float[]> scale_weight(new float[dd_size]);
    std::unique_ptr<float[]> scale_bias(new float[d_size]);
    std::unique_ptr<float[]> c_b2(new float[dd_size]);

    std::unique_ptr<bfloat16_t[]> pack_scale_weight(new bfloat16_t[dd_size]);
    std::unique_ptr<bfloat16_t[]> pack_c_b2(new bfloat16_t[dd_size]);
    std::unique_ptr<bfloat16_t[]> pack_s_c(new bfloat16_t[s_size]);
    std::unique_ptr<bfloat16_t[]> pack_x(new bfloat16_t[x_size]);

    std::unique_ptr<float[]> x_ori(new float[x_size]);
    std::unique_ptr<float[]> s_c_ori(new float[s_size]);
    std::unique_ptr<float[]> s_c_scale(new float[s_temp_size]);
    std::unique_ptr<float[]> s_c_sigmoid(new float[s_temp_size]);
    std::unique_ptr<float[]> s_c_bias(new float[s_temp_size]);
    std::unique_ptr<float[]> output(new float[x_size]);

    for(int i = 0; i < x_size; ++i){
        x[i] = rnd() * 1.0 / rnd.max();
    }

    for(int i = 0; i < s_size; ++i){
        s_c[i] = rnd() * 1.0 / rnd.max();
    }
    
    for(int i = 0; i < dd_size; ++i){
        scale_weight[i] = rnd() * 1.0 / rnd.max();
        c_b2[i] = rnd() * 1.0 / rnd.max();
    }

    for(int i = 0; i < d_size; ++i){
        ln_weight[i] = rnd() * 1.0 / rnd.max();
        scale_bias[i] = rnd() * 1.0 / rnd.max();
    }

    layer_norm_manual(x.get(), nullptr, x_ori.get(), N_s*B, D0, D1, eps, false);
    layer_norm_manual(s_c.get(), ln_weight.get(), s_c_ori.get(), B, D0, D2, eps, true);
    matmul_3d_2d(s_c_ori.get(), scale_weight.get(), s_c_scale.get(), B, D0, D1, D2);
    add_bias_broadcast(s_c_scale.get(), scale_bias.get(), B*D0, D1);
    sigmoid_manual(s_c_scale.get(), s_c_sigmoid.get(), s_temp_size);
    matmul_3d_2d(s_c_ori.get(), c_b2.get(), s_c_bias.get(), B, D0, D1, D2);
    elementwise_fuse(s_c_sigmoid.get(), x_ori.get(), s_c_bias.get(), output.get(), N_s, B, D0, D1);

    pack_weight_task_bf16(scale_weight.get(), pack_scale_weight.get(), D2, D1);
    pack_weight_task_bf16(c_b2.get(), pack_c_b2.get(), D2, D1);
    kutacc::layernorm_noparams(x.get(), N_s*B, D0, D1, eps);
    kutacc::layernorm_pack_bf16(s_c.get(), pack_s_c.get(), ln_weight.get(), nullptr, B*D0, D2, eps);
    kutacc::adpln_merge_bf16(x.get(), pack_s_c.get(), pack_scale_weight.get(), scale_bias.get(), pack_c_b2.get(), N_s, B*D0, D1, D2, nullptr, false);

    for (int64_t i = 0; i < N_s; i++) {
        for (int64_t j = 0; j < B; j++) {
            for (int64_t m = 0; m < D0; m++) {
                for (int64_t n = 0; n < D1; n++) {
                    float out = x[i*B*D0*D1 + j*D0*D1 + m*D1 + n];
                    float expect = output[i*B*D0*D1 + j*D0*D1 + m*D1 + n];
                    if (std::abs(out - expect) > 0.1) {
                        std::cerr << "(" << i << ", " << j << ", " << m << ", " << n << ") is " << out << ", expect " << expect << ", "
                                << "\n";
                        exit(1);
                    }
                }
            }
        }
    }

    for(int i = 0; i < x_size; ++i){
        x[i] = rnd() * 1.0 / rnd.max();
    }

    for(int t = 0; t<20; ++t)
    {
        kutacc::layernorm_noparams(x.get(), N_s*B, D0, D1, eps);
        kutacc::layernorm_pack_bf16(s_c.get(), pack_s_c.get(), ln_weight.get(), nullptr, B*D0, D2, eps);
        kutacc::adpln_merge_bf16(x.get(), pack_s_c.get(), pack_scale_weight.get(), scale_bias.get(), pack_c_b2.get(), N_s, B*D0, D1, D2, pack_x.get(), true);
    }
    auto start_o_st = std::chrono::steady_clock::now();
    for(int t = 0; t<50; ++t)
    {
        kutacc::layernorm_noparams(x.get(), N_s*B, D0, D1, eps);
        kutacc::layernorm_pack_bf16(s_c.get(), pack_s_c.get(), ln_weight.get(), nullptr, B*D0, D2, eps);
        kutacc::adpln_merge_bf16(x.get(), pack_s_c.get(), pack_scale_weight.get(), scale_bias.get(), pack_c_b2.get(), N_s, B*D0, D1, D2, pack_x.get(), true);
    }
    auto end_o_st = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapse_o_store = end_o_st - start_o_st;
    std::cout << "adap_ln time = " << elapse_o_store.count()/T << std::endl;

}

int main(int argc, char **argv)
{
    kutacc::global_parallel_launch([&] {
        std::vector<std::vector<int64_t>> cases;
        read_args_skip_n(cases, 2, argc, argv, 3);
        int D0 = std::stoi(argv[1]);
        int D1 = std::stoi(argv[2]);
        int D2 = std::stoi(argv[3]);

        for (auto cas : cases) {
            test_adap_ln(cas, D0, D1, D2);
        }
    });
}
