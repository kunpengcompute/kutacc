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
#include "../linear.h"
#include "activation/sigmoid.h"
#include "math/fast_exp.h"
#include "utils/bf16.h"

namespace kutacc {
inline void matrix_right_pack_bf16_N(const bfloat16_t *matrix,bfloat16_t *matrix_pack,int K, int N)
{   // K = 768 N = 8 
    svbool_t pt16_c=svwhilelt_b16(0,N);
    svbool_t pt16_2c=svwhilelt_b16(0,2*N);
    for(int row_index = 0; row_index < K ; row_index += 2)
    {
        svbfloat16_t unz1,unz2,z1;
        unz1 = svld1_bf16(pt16_c, matrix + row_index * N );  
        unz2 = svld1_bf16(pt16_c, matrix + (row_index + 1) * N); 
        z1 = svzip1_bf16(unz1, unz2); 
        svst1_bf16(pt16_2c,matrix_pack+row_index*N, z1); 
    }
}

void msa_attention_sme_8_vsl_impl(
    float* logits,      // [h * q * k]
    bfloat16_t* v,              // [b * h * k * c]
    bfloat16_t* v_avg,                // [b * h * q * c]
    int h, int q, int k, int b, int c
) {
    // hqk @ bhkc -> bhqc 8*768*768 @ 1024*8*768*8 -> 1024*8*768*8

    int vscalew=static_cast<int>(svcntw()); //16
    svbool_t pt16=svptrue_b16();
    svbool_t pt32=svptrue_b32();

    svbool_t pt16_c = svwhilelt_b16(0,c); // 0-8

    svbool_t pt16_2c = svwhilelt_b16(0,2*c); // 0-16
    svbool_t pt16_2c_4c = svcmpgt_n_u16(pt16,svindex_u16(0, 1), 15); // 16-32

    svbool_t pt16_c_2c = svcmpgt_n_u16(pt16_2c,svindex_u16(0, 1), 7); // 8-16
    svbool_t pt16_2c_3c = svcmplt_n_u16(pt16_2c_4c,svindex_u16(0, 1), 24); // 16-24
    svbool_t pt16_3c_4c = svcmpgt_n_u16(pt16_2c_4c,svindex_u16(0, 1), 23); // 24-32

    bfloat16_t* weights=static_cast<bfloat16_t*>(aligned_alloc(64, sizeof(bfloat16_t) * static_cast<unsigned long>(8 * vscalew * k)));

    float* l_sum = static_cast<float*>(aligned_alloc(64, sizeof(float) * static_cast<unsigned long>(8 * vscalew)));
    float* l_max = static_cast<float*>(aligned_alloc(64, sizeof(float) * static_cast<unsigned long>(8 * vscalew)));
    std::fill(l_sum, l_sum + 8 * vscalew, static_cast<float>(0.0f));
    std::fill(l_max, l_max + 8 * vscalew, static_cast<float>(-FLT_MAX));
    for(int block_index=0; block_index<8; block_index++)
    {
        // online softmax: max_t = max(max_t-1, logits_t)
        // sum_t = sum[t-1] * e ^ (max_t-1 - max_t) + e ^ (logits_t - max_t)
        for(int row_index = 0; row_index < vscalew; row_index++)
        {
            #pragma unroll(8)
            for(int k_idx=0; k_idx<k; k_idx+=vscalew)
            {
                int row_n = row_index + block_index*vscalew ;
                svfloat32_t l_data = svld1_f32(pt32, logits + row_n * k + k_idx);
                float max_row = std::max(l_max[row_n], svmaxv_f32(pt32,l_data));
                l_data = svsub_n_f32_z(pt32, l_data, max_row);
                l_data = fast_exp(pt32,l_data);
                l_sum[row_n] = l_sum[row_n] * std::exp(l_max[row_n] - max_row) + svaddv_f32(pt32, l_data);
                l_max[row_n] = max_row; 
            }
        }
    }
    for(int k_idx=0; k_idx<k; k_idx+=vscalew)
    {
        for(int cycle=0; cycle<2;cycle++){
            SME_ON();
            svzero_za();
            for (int row_index=0;row_index<vscalew;row_index++) // load data to ZA tile
            {
                int logits_base = (row_index + 4 * cycle * vscalew) * k + k_idx;
                svld1_hor_za32(0, static_cast<uint32_t>(row_index), pt32, logits + logits_base);
                svld1_hor_za32(1, static_cast<uint32_t>(row_index), pt32, logits + logits_base + vscalew * k);
                svld1_hor_za32(2, static_cast<uint32_t>(row_index), pt32, logits + logits_base + 2 * vscalew * k);
                svld1_hor_za32(3, static_cast<uint32_t>(row_index), pt32, logits + logits_base + 3 * vscalew * k);
            }
            for (int col_index=0;col_index<vscalew;col_index+=2) 
            {
                int weights_base = (4 * cycle * vscalew) * k + (k_idx + col_index) * vscalew;
                svfloat32_t l_data_0, l_data_1, l_data_2, l_data_3, logits_max_0, logits_max_1, logits_sum_0, logits_sum_1;
                svbfloat16_t data_0_bf16, data_1_bf16, data_2_bf16, data_3_bf16, unz1, unz2, z1, z2;

                // ZA tile 0,1
                l_data_0 = svread_ver_za32_f32_m(l_data_0, pt32, 0, static_cast<uint32_t>(col_index));
                l_data_1 = svread_ver_za32_f32_m(l_data_1, pt32, 0, static_cast<uint32_t>(col_index+1)); 
                l_data_2 = svread_ver_za32_f32_m(l_data_2, pt32, 1, static_cast<uint32_t>(col_index));
                l_data_3 = svread_ver_za32_f32_m(l_data_3, pt32, 1, static_cast<uint32_t>(col_index+1)); 
                logits_max_0 = svld1_f32(pt32, l_max + 4 * cycle * vscalew); 
                logits_max_1 = svld1_f32(pt32, l_max + (4 * cycle + 1)* vscalew);
                logits_sum_0 = svld1_f32(pt32, l_sum + 4 * cycle * vscalew);
                logits_sum_1 = svld1_f32(pt32, l_sum + (4 * cycle + 1)* vscalew);

                l_data_0 = svsub_f32_z(pt32, l_data_0, logits_max_0);
                l_data_0 = fast_exp(pt32,l_data_0);
                l_data_0 = svdiv_f32_z(pt32,l_data_0,logits_sum_0);

                l_data_1 = svsub_f32_z(pt32, l_data_1, logits_max_0);
                l_data_1 = fast_exp(pt32,l_data_1);
                l_data_1 = svdiv_f32_z(pt32,l_data_1,logits_sum_0);

                l_data_2 = svsub_f32_z(pt32, l_data_2, logits_max_1);
                l_data_2 = fast_exp(pt32,l_data_2);
                l_data_2 = svdiv_f32_z(pt32,l_data_2,logits_sum_1);

                l_data_3 = svsub_f32_z(pt32, l_data_3, logits_max_1);
                l_data_3 = fast_exp(pt32,l_data_3);
                l_data_3 = svdiv_f32_z(pt32,l_data_3,logits_sum_1);

                data_0_bf16 = svcvt_bf16_f32_z(pt32, l_data_0);
                data_1_bf16 = svcvt_bf16_f32_z(pt32, l_data_1);
                data_2_bf16 = svcvt_bf16_f32_z(pt32, l_data_2);
                data_3_bf16 = svcvt_bf16_f32_z(pt32, l_data_3);

                unz1 = svuzp1_bf16(data_0_bf16,data_2_bf16); 
                unz2 = svuzp1_bf16(data_1_bf16,data_3_bf16);
                z1 = svzip1_bf16(unz1,unz2);
                z2 = svzip2_bf16(unz1,unz2);
                svst1_bf16(pt16, weights + weights_base, z1);
                svst1_bf16(pt16, weights + weights_base + vscalew * k, z2);

                // ZA tile 2,3
                l_data_0 = svread_ver_za32_f32_m(l_data_0, pt32, 2, static_cast<uint32_t>(col_index));
                l_data_1 = svread_ver_za32_f32_m(l_data_1, pt32, 2, static_cast<uint32_t>(col_index+1)); 
                l_data_2 = svread_ver_za32_f32_m(l_data_2, pt32, 3, static_cast<uint32_t>(col_index));
                l_data_3 = svread_ver_za32_f32_m(l_data_3, pt32, 3, static_cast<uint32_t>(col_index+1)); 
                logits_max_0 = svld1_f32(pt32, l_max + (4 * cycle + 2) * vscalew); 
                logits_max_1 = svld1_f32(pt32, l_max + (4 * cycle + 3) * vscalew); 
                logits_sum_0 = svld1_f32(pt32, l_sum + (4 * cycle + 2) * vscalew);
                logits_sum_1 = svld1_f32(pt32, l_sum + (4 * cycle + 3) * vscalew);

                l_data_0 = svsub_f32_z(pt32, l_data_0, logits_max_0);
                l_data_0 = fast_exp(pt32,l_data_0);
                l_data_0 = svdiv_f32_z(pt32,l_data_0,logits_sum_0);

                l_data_1 = svsub_f32_z(pt32, l_data_1, logits_max_0);
                l_data_1 = fast_exp(pt32,l_data_1);
                l_data_1 = svdiv_f32_z(pt32,l_data_1,logits_sum_0);

                l_data_2 = svsub_f32_z(pt32, l_data_2, logits_max_1);
                l_data_2 = fast_exp(pt32,l_data_2);
                l_data_2 = svdiv_f32_z(pt32,l_data_2,logits_sum_1);

                l_data_3 = svsub_f32_z(pt32, l_data_3, logits_max_1);
                l_data_3 = fast_exp(pt32,l_data_3);
                l_data_3 = svdiv_f32_z(pt32,l_data_3,logits_sum_1);

                data_0_bf16 = svcvt_bf16_f32_z(pt32, l_data_0);
                data_1_bf16 = svcvt_bf16_f32_z(pt32, l_data_1);
                data_2_bf16 = svcvt_bf16_f32_z(pt32, l_data_2);
                data_3_bf16 = svcvt_bf16_f32_z(pt32, l_data_3);

                unz1 = svuzp1_bf16(data_0_bf16,data_2_bf16); 
                unz2 = svuzp1_bf16(data_1_bf16,data_3_bf16);
                z1 = svzip1_bf16(unz1,unz2);
                z2 = svzip2_bf16(unz1,unz2);
                svst1_bf16(pt16, weights + weights_base + 2 * vscalew * k, z1);
                svst1_bf16(pt16, weights + weights_base + 3 * vscalew * k, z2);
            }
            SME_OFF();
        }
    }
    for(int b_idx = 0; b_idx < b; b_idx++)
    {
        int v_base = b_idx * h * k * c;
        bfloat16_t* v_transposed =static_cast<bfloat16_t*>(aligned_alloc(64, sizeof(bfloat16_t) * static_cast<unsigned long>(k * c)));
        matrix_right_pack_bf16_N(v + v_base, v_transposed, k , c);
        SME_ON();
        svzero_za();
        for(int k_idx=0; k_idx<k; k_idx+=2)
        {
            int weights_base = k_idx * vscalew;
            svbfloat16_t rowW0 = svld1_bf16(pt16,weights + weights_base);
            svbfloat16_t rowW1 = svld1_bf16(pt16,weights + weights_base + vscalew * k);
            svbfloat16_t rowW2 = svld1_bf16(pt16,weights + weights_base + 2 * vscalew * k);
            svbfloat16_t rowW3 = svld1_bf16(pt16,weights + weights_base + 3 * vscalew * k);
            svbfloat16_t rowW4 = svld1_bf16(pt16,weights + weights_base + 4 * vscalew * k);
            svbfloat16_t rowW5 = svld1_bf16(pt16,weights + weights_base + 5 * vscalew * k);
            svbfloat16_t rowW6 = svld1_bf16(pt16,weights + weights_base + 6 * vscalew * k);
            svbfloat16_t rowW7 = svld1_bf16(pt16,weights + weights_base + 7 * vscalew * k);

            svbfloat16_t colV0 = svld1_bf16(pt16_2c,v_transposed+k_idx*c);
            svbfloat16_t colV1 = svld1_bf16(pt16_2c_4c,v_transposed+(k_idx-2)*c);

            svmopa_za32_bf16_m(0,pt16,pt16_2c,rowW0,colV0);
            svmopa_za32_bf16_m(0,pt16,pt16_2c_4c,rowW1,colV1);
            svmopa_za32_bf16_m(1,pt16,pt16_2c,rowW2,colV0);
            svmopa_za32_bf16_m(1,pt16,pt16_2c_4c,rowW3,colV1);
            svmopa_za32_bf16_m(2,pt16,pt16_2c,rowW4,colV0);
            svmopa_za32_bf16_m(2,pt16,pt16_2c_4c,rowW5,colV1);
            svmopa_za32_bf16_m(3,pt16,pt16_2c,rowW6,colV0);
            svmopa_za32_bf16_m(3,pt16,pt16_2c_4c,rowW7,colV1);
        }
        for (int row_index = 0; row_index < vscalew; row_index+=2) 
        {
            int v_avg_base = b_idx * h * q * c + row_index * c;
            svfloat32_t data_0, data_1, data_2, data_3;
            svbfloat16_t data_0_bf16, data_1_bf16, data_2_bf16, data_3_bf16, unzip_1, unzip_2;
            data_0 = svread_hor_za32_f32_m(data_0,pt32,0,static_cast<uint32_t>(row_index));
            data_1 = svread_hor_za32_f32_m(data_1,pt32,0,static_cast<uint32_t>(row_index+1));
            data_2 = svread_hor_za32_f32_m(data_2,pt32,1,static_cast<uint32_t>(row_index));
            data_3 = svread_hor_za32_f32_m(data_3,pt32,1,static_cast<uint32_t>(row_index+1));

            data_0_bf16 = svcvt_bf16_f32_z(pt32, data_0); 
            data_1_bf16 = svcvt_bf16_f32_z(pt32, data_1);
            data_2_bf16 = svcvt_bf16_f32_z(pt32, data_2);
            data_3_bf16 = svcvt_bf16_f32_z(pt32, data_3);

            unzip_1 = svuzp1_bf16(data_0_bf16,data_1_bf16);
            unzip_2 = svuzp1_bf16(data_2_bf16,data_3_bf16);

            svst1_bf16(pt16_c, v_avg + v_avg_base, unzip_1); 
            svst1_bf16(pt16_2c_3c, v_avg + v_avg_base - c , unzip_1); // to match pt16_2c_3c, we need base address subtract 2c
            svst1_bf16(pt16_c_2c, v_avg + v_avg_base + (vscalew - 1) * c, unzip_1); // so as above
            svst1_bf16(pt16_3c_4c, v_avg + v_avg_base + (vscalew - 2) * c, unzip_1);
            svst1_bf16(pt16_c, v_avg + v_avg_base + (2*vscalew) * c, unzip_2); 
            svst1_bf16(pt16_2c_3c, v_avg + v_avg_base + (2*vscalew - 1)  * c, unzip_2); 
            svst1_bf16(pt16_c_2c, v_avg + v_avg_base + (3*vscalew - 1) * c, unzip_2);
            svst1_bf16(pt16_3c_4c, v_avg + v_avg_base + (3*vscalew - 2) * c, unzip_2);

            data_0 = svread_hor_za32_f32_m(data_0,pt32,2,static_cast<uint32_t>(row_index));
            data_1 = svread_hor_za32_f32_m(data_1,pt32,2,static_cast<uint32_t>(row_index+1));
            data_2 = svread_hor_za32_f32_m(data_2,pt32,3,static_cast<uint32_t>(row_index));
            data_3 = svread_hor_za32_f32_m(data_3,pt32,3,static_cast<uint32_t>(row_index+1));

            data_0_bf16 = svcvt_bf16_f32_z(pt32, data_0); 
            data_1_bf16 = svcvt_bf16_f32_z(pt32, data_1);
            data_2_bf16 = svcvt_bf16_f32_z(pt32, data_2);
            data_3_bf16 = svcvt_bf16_f32_z(pt32, data_3);

            unzip_1 = svuzp1_bf16(data_0_bf16,data_1_bf16);
            unzip_2 = svuzp1_bf16(data_2_bf16,data_3_bf16);

            svst1_bf16(pt16_c, v_avg + v_avg_base + (4*vscalew) * c, unzip_1); 
            svst1_bf16(pt16_2c_3c, v_avg + v_avg_base + (4*vscalew - 1)  * c, unzip_1); 
            svst1_bf16(pt16_c_2c, v_avg + v_avg_base + (5*vscalew - 1) * c, unzip_1);
            svst1_bf16(pt16_3c_4c, v_avg + v_avg_base + (5*vscalew - 2) * c, unzip_1);
            svst1_bf16(pt16_c, v_avg + v_avg_base + (6*vscalew) * c, unzip_2); 
            svst1_bf16(pt16_2c_3c, v_avg + v_avg_base + (6*vscalew - 1) * c, unzip_2); 
            svst1_bf16(pt16_c_2c, v_avg + v_avg_base + (7*vscalew - 1) * c, unzip_2);
            svst1_bf16(pt16_3c_4c, v_avg + v_avg_base + (7*vscalew - 2) * c, unzip_2);
        }
        SME_OFF();
        free(v_transposed);
    }
    free(l_sum);
    free(l_max);
    free(weights);
}

void msa_attention_sme_4_vsl_impl(
    float* logits,      // [h * q * k]
    bfloat16_t* v,              // [b * h * k * c]
    bfloat16_t* v_avg,                // [b * h * q * c]
    int h ,int q, int k, int b, int c
) {
    // hqk @ bhkc -> bhqc 8*768*768 @ 1024*8*768*8 -> 1024*8*768*8
    // int vscaleh=svcnth(); //32
    int vscalew=static_cast<int>(svcntw()); //16
    svbool_t pt16=svptrue_b16();
    svbool_t pt32=svptrue_b32();

    svbool_t pt16_c = svwhilelt_b16(0,c); // 0-8

    svbool_t pt16_2c = svwhilelt_b16(0,2*c); // 0-16
    svbool_t pt16_2c_4c = svcmpgt_n_u16(pt16,svindex_u16(0, 1), 15); // 16-32
    svbool_t pt16_2c_3c = svcmplt_n_u16(pt16_2c_4c,svindex_u16(0, 1), 24); // 16-24

    bfloat16_t* weights=static_cast<bfloat16_t*>(aligned_alloc(64, sizeof(bfloat16_t) * static_cast<unsigned long>(4 * vscalew * k)));
    float* l_sum = static_cast<float*>(aligned_alloc(64, sizeof(float) * static_cast<unsigned long>(4 * vscalew)));
    float* l_max = static_cast<float*>(aligned_alloc(64, sizeof(float) * static_cast<unsigned long>(4 * vscalew)));
    std::fill(l_sum, l_sum + 4* vscalew, static_cast<float>(0.0f));
    std::fill(l_max, l_max + 4* vscalew, static_cast<float>(-FLT_MAX));
    for(int block_index=0; block_index<4; block_index++)
    {
        // online softmax: max_t = max(max_t-1, logits_t)
        // sum_t = sum[t-1] * e ^ (max_t-1 - max_t) + e ^ (logits_t - max_t)
        for(int row_index = 0; row_index < vscalew; row_index++)
        {   
            #pragma unroll(4)
            for(int k_idx=0; k_idx<k; k_idx+=vscalew)
            {
                int row_n = row_index + block_index*vscalew ;
                svfloat32_t l_data = svld1_f32(pt32, logits + row_n * k + k_idx);
                float max_row = std::max(l_max[row_n], svmaxv_f32(pt32,l_data));
                l_data = svsub_n_f32_z(pt32, l_data, max_row);
                l_data = fast_exp(pt32,l_data);
                l_sum[row_n] = l_sum[row_n] * std::exp(l_max[row_n] - max_row) + svaddv_f32(pt32, l_data);
                l_max[row_n] = max_row; 
            }
        }
    }
    for(int k_idx=0; k_idx<k; k_idx+=vscalew)
    {
        SME_ON();
        svzero_za();
        for (int row_index=0;row_index<vscalew;row_index++) // load data to ZA tile
        {
            int logits_base = row_index * k + k_idx;
            svld1_hor_za32(0, static_cast<uint32_t>(row_index), pt32, logits + logits_base);
            svld1_hor_za32(1, static_cast<uint32_t>(row_index), pt32, logits + logits_base + vscalew * k);
            svld1_hor_za32(2, static_cast<uint32_t>(row_index), pt32, logits + logits_base + 2 * vscalew * k);
            svld1_hor_za32(3, static_cast<uint32_t>(row_index), pt32, logits + logits_base + 3 * vscalew * k);
        }
        for (int col_index=0;col_index<vscalew;col_index+=2) 
        {
            int weights_base = (k_idx + col_index) * vscalew;
            svfloat32_t l_data_0, l_data_1, l_data_2, l_data_3, logits_max_0, logits_max_1, logits_sum_0, logits_sum_1;
            svbfloat16_t data_0_bf16, data_1_bf16, data_2_bf16, data_3_bf16, unz1, unz2, z1, z2;

            // ZA tile 0,1
            l_data_0 = svread_ver_za32_f32_m(l_data_0, pt32, 0, static_cast<uint32_t>(col_index));
            l_data_1 = svread_ver_za32_f32_m(l_data_1, pt32, 0, static_cast<uint32_t>(col_index+1)); 
            l_data_2 = svread_ver_za32_f32_m(l_data_2, pt32, 1, static_cast<uint32_t>(col_index));
            l_data_3 = svread_ver_za32_f32_m(l_data_3, pt32, 1, static_cast<uint32_t>(col_index+1)); 
            logits_max_0 = svld1_f32(pt32, l_max ); 
            logits_max_1 = svld1_f32(pt32, l_max + vscalew);
            logits_sum_0 = svld1_f32(pt32, l_sum );
            logits_sum_1 = svld1_f32(pt32, l_sum + vscalew);

            l_data_0 = svsub_f32_z(pt32, l_data_0, logits_max_0);
            l_data_0 = fast_exp(pt32,l_data_0);
            l_data_0 = svdiv_f32_z(pt32,l_data_0,logits_sum_0);

            l_data_1 = svsub_f32_z(pt32, l_data_1, logits_max_0);
            l_data_1 = fast_exp(pt32,l_data_1);
            l_data_1 = svdiv_f32_z(pt32,l_data_1,logits_sum_0);

            l_data_2 = svsub_f32_z(pt32, l_data_2, logits_max_1);
            l_data_2 = fast_exp(pt32,l_data_2);
            l_data_2 = svdiv_f32_z(pt32,l_data_2,logits_sum_1);

            l_data_3 = svsub_f32_z(pt32, l_data_3, logits_max_1);
            l_data_3 = fast_exp(pt32,l_data_3);
            l_data_3 = svdiv_f32_z(pt32,l_data_3,logits_sum_1);

            data_0_bf16 = svcvt_bf16_f32_z(pt32, l_data_0);
            data_1_bf16 = svcvt_bf16_f32_z(pt32, l_data_1);
            data_2_bf16 = svcvt_bf16_f32_z(pt32, l_data_2);
            data_3_bf16 = svcvt_bf16_f32_z(pt32, l_data_3);

            unz1 = svuzp1_bf16(data_0_bf16,data_2_bf16); 
            unz2 = svuzp1_bf16(data_1_bf16,data_3_bf16);
            z1 = svzip1_bf16(unz1,unz2);
            z2 = svzip2_bf16(unz1,unz2);
            svst1_bf16(pt16, weights + weights_base, z1);
            svst1_bf16(pt16, weights + weights_base + vscalew * k, z2);

            // ZA tile 2,3
            l_data_0 = svread_ver_za32_f32_m(l_data_0, pt32, 2, static_cast<uint32_t>(col_index));
            l_data_1 = svread_ver_za32_f32_m(l_data_1, pt32, 2, static_cast<uint32_t>(col_index+1)); 
            l_data_2 = svread_ver_za32_f32_m(l_data_2, pt32, 3, static_cast<uint32_t>(col_index));
            l_data_3 = svread_ver_za32_f32_m(l_data_3, pt32, 3, static_cast<uint32_t>(col_index+1)); 
            logits_max_0 = svld1_f32(pt32, l_max + 2 * vscalew); 
            logits_max_1 = svld1_f32(pt32, l_max + 3 * vscalew); 
            logits_sum_0 = svld1_f32(pt32, l_sum + 2 * vscalew);
            logits_sum_1 = svld1_f32(pt32, l_sum + 3 * vscalew);

            l_data_0 = svsub_f32_z(pt32, l_data_0, logits_max_0);
            l_data_0 = fast_exp(pt32,l_data_0);
            l_data_0 = svdiv_f32_z(pt32,l_data_0,logits_sum_0);

            l_data_1 = svsub_f32_z(pt32, l_data_1, logits_max_0);
            l_data_1 = fast_exp(pt32,l_data_1);
            l_data_1 = svdiv_f32_z(pt32,l_data_1,logits_sum_0);

            l_data_2 = svsub_f32_z(pt32, l_data_2, logits_max_1);
            l_data_2 = fast_exp(pt32,l_data_2);
            l_data_2 = svdiv_f32_z(pt32,l_data_2,logits_sum_1);

            l_data_3 = svsub_f32_z(pt32, l_data_3, logits_max_1);
            l_data_3 = fast_exp(pt32,l_data_3);
            l_data_3 = svdiv_f32_z(pt32,l_data_3,logits_sum_1);

            data_0_bf16 = svcvt_bf16_f32_z(pt32, l_data_0);
            data_1_bf16 = svcvt_bf16_f32_z(pt32, l_data_1);
            data_2_bf16 = svcvt_bf16_f32_z(pt32, l_data_2);
            data_3_bf16 = svcvt_bf16_f32_z(pt32, l_data_3);

            unz1 = svuzp1_bf16(data_0_bf16,data_2_bf16); 
            unz2 = svuzp1_bf16(data_1_bf16,data_3_bf16);
            z1 = svzip1_bf16(unz1,unz2);
            z2 = svzip2_bf16(unz1,unz2);
            svst1_bf16(pt16, weights + weights_base + 2 * vscalew * k, z1);
            svst1_bf16(pt16, weights + weights_base + 3 * vscalew * k, z2);
        }
        SME_OFF();
    }
    for(int b_idx =0; b_idx < b; b_idx++)
    {
        int v_base = b_idx * h * k * c ;
        bfloat16_t* v_transposed =static_cast<bfloat16_t*>(aligned_alloc(64, sizeof(bfloat16_t) * static_cast<unsigned long>(k * c)));
        matrix_right_pack_bf16_N(v + v_base, v_transposed, k , c);
        SME_ON();
        svzero_za();
        for(int k_idx=0; k_idx<k; k_idx+=2)
        {
            int weights_base = k_idx * vscalew;
            svbfloat16_t rowW0 = svld1_bf16(pt16,weights + weights_base);
            svbfloat16_t rowW1 = svld1_bf16(pt16,weights + weights_base + vscalew * k);
            svbfloat16_t rowW2 = svld1_bf16(pt16,weights + weights_base + 2 * vscalew * k);
            svbfloat16_t rowW3 = svld1_bf16(pt16,weights + weights_base + 3 * vscalew * k);

            svbfloat16_t colV = svld1_bf16(pt16_2c,v_transposed+k_idx*c);

            svmopa_za32_bf16_m(0,pt16,pt16_2c,rowW0,colV);
            svmopa_za32_bf16_m(1,pt16,pt16_2c,rowW1,colV);
            svmopa_za32_bf16_m(2,pt16,pt16_2c,rowW2,colV); 
            svmopa_za32_bf16_m(3,pt16,pt16_2c,rowW3,colV);

        }
        for (int row_index = 0; row_index < vscalew; row_index+=2) 
        {
            int v_avg_base = b_idx * h * q * c + row_index * c;
            svfloat32_t data_0, data_1, data_2, data_3;
            svbfloat16_t data_0_bf16, data_1_bf16, data_2_bf16, data_3_bf16, unzip_1, unzip_2;
            data_0 = svread_hor_za32_f32_m(data_0,pt32,0,static_cast<uint32_t>(row_index));
            data_1 = svread_hor_za32_f32_m(data_1,pt32,0,static_cast<uint32_t>(row_index+1));
            data_2 = svread_hor_za32_f32_m(data_2,pt32,1,static_cast<uint32_t>(row_index));
            data_3 = svread_hor_za32_f32_m(data_3,pt32,1,static_cast<uint32_t>(row_index+1));

            data_0_bf16 = svcvt_bf16_f32_z(pt32, data_0); 
            data_1_bf16 = svcvt_bf16_f32_z(pt32, data_1);
            data_2_bf16 = svcvt_bf16_f32_z(pt32, data_2);
            data_3_bf16 = svcvt_bf16_f32_z(pt32, data_3);

            unzip_1 = svuzp1_bf16(data_0_bf16,data_1_bf16);
            unzip_2 = svuzp1_bf16(data_2_bf16,data_3_bf16);

            svst1_bf16(pt16_c, v_avg + v_avg_base, unzip_1); 
            svst1_bf16(pt16_2c_3c, v_avg + v_avg_base - c , unzip_1); // to match pt16_2c_3c, we need base address subtract 2c
            svst1_bf16(pt16_c, v_avg + v_avg_base + vscalew * c, unzip_2); 
            svst1_bf16(pt16_2c_3c, v_avg + v_avg_base + (vscalew - 1)  * c, unzip_2); 

            data_0 = svread_hor_za32_f32_m(data_0,pt32,2,static_cast<uint32_t>(row_index));
            data_1 = svread_hor_za32_f32_m(data_1,pt32,2,static_cast<uint32_t>(row_index+1));
            data_2 = svread_hor_za32_f32_m(data_2,pt32,3,static_cast<uint32_t>(row_index));
            data_3 = svread_hor_za32_f32_m(data_3,pt32,3,static_cast<uint32_t>(row_index+1));

            data_0_bf16 = svcvt_bf16_f32_z(pt32, data_0); 
            data_1_bf16 = svcvt_bf16_f32_z(pt32, data_1);
            data_2_bf16 = svcvt_bf16_f32_z(pt32, data_2);
            data_3_bf16 = svcvt_bf16_f32_z(pt32, data_3);

            unzip_1 = svuzp1_bf16(data_0_bf16,data_1_bf16);
            unzip_2 = svuzp1_bf16(data_2_bf16,data_3_bf16);

            svst1_bf16(pt16_c, v_avg + v_avg_base + (2*vscalew) * c, unzip_1); 
            svst1_bf16(pt16_2c_3c, v_avg + v_avg_base + (2*vscalew - 1)  * c, unzip_1); 
            svst1_bf16(pt16_c, v_avg + v_avg_base + (3*vscalew) * c, unzip_2); 
            svst1_bf16(pt16_2c_3c, v_avg + v_avg_base + (3*vscalew - 1) * c, unzip_2); 
        }
        SME_OFF();
        free(v_transposed);
    }
    free(l_sum);
    free(l_max);
    free(weights);
}


void msa_attention_sme(
    float* logits,      // [h * q * k]
    bfloat16_t* v,              // [b * h * k * c]
    bfloat16_t* v_avg,                // [b * h * q * c]
    int h, int q, int k, int b, int c
) {
    const int vscalew = static_cast<int>(svcntw()); //16
    const int data_threshold = 768;
    const int Q_BLOCK_8 = vscalew * 8;
    const int Q_BLOCK_4 = vscalew * 4;
    if(q % Q_BLOCK_8 != 0){
        int task_num = (h * q) / Q_BLOCK_4;
        int task_num_dim0 = q / Q_BLOCK_4;
        kutacc::parallel_for(0, task_num, 1, [&](int64_t start, int64_t end) {
            for(int task_idx = start; task_idx < end; task_idx += 1){
                int h_idx = (task_idx / task_num_dim0);
                int q_idx = (task_idx % task_num_dim0) * Q_BLOCK_4;
                float *part_logits_base = logits + h_idx * q * k + q_idx * k;
                bfloat16_t *part_v_base = v + h_idx * k * c;
                bfloat16_t *part_v_avg_base = v_avg + h_idx * q * c + q_idx * c;
                kutacc::msa_attention_sme_4_vsl_impl(part_logits_base, part_v_base, part_v_avg_base, h, q, k, b, c);
            }
        });
    }else if(q < data_threshold){
        int task_num = (h * q) / Q_BLOCK_8;
        int task_num_dim0 = q / Q_BLOCK_8;
        kutacc::parallel_for(0, task_num, 1, [&](int64_t start, int64_t end) {
            for(int task_idx = start; task_idx < end; task_idx += 1){
                int h_idx = (task_idx / task_num_dim0);
                int q_idx = (task_idx % task_num_dim0) * Q_BLOCK_8;
                float *part_logits_base = logits + h_idx * q * k + q_idx * k;
                bfloat16_t *part_v_base = v + h_idx * k * c;
                bfloat16_t *part_v_avg_base = v_avg + h_idx * q * c + q_idx * c;
                kutacc::msa_attention_sme_8_vsl_impl(part_logits_base, part_v_base, part_v_avg_base, h, q, k, b, c);
            }
        });
    }else{
        int max_threads = kutacc::get_thread_num();
        max_threads = std::min(max_threads, 32);
        if(max_threads < 32)
        {
            max_threads = 24;
        }
        const int Bq = max_threads / h ;
        const int q_stride = q / Bq;
        kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
        {
            int tid = kutacc::get_thread_id();
            int h_step = tid / Bq;
            int q_step = tid % Bq;
            for(int q_idx = 0; q_idx < q_stride;)
            {
                const int remain = q_stride - q_idx;
                if(remain >= Q_BLOCK_8)
                {
                    float *part_logits_base = logits + h_step * q * k + (q_step * q_stride + q_idx) * k;
                    bfloat16_t *part_v_base = v + h_step * k * c;
                    bfloat16_t *part_v_avg_base = v_avg + h_step * q * c + (q_step * q_stride + q_idx) * c;
                    kutacc::msa_attention_sme_8_vsl_impl(part_logits_base, part_v_base, part_v_avg_base, h, q, k, b, c);
                    q_idx += Q_BLOCK_8;
                }
                else if(remain >= Q_BLOCK_4)
                {
                    float *part_logits_base = logits + h_step * q * k + (q_step * q_stride + q_idx) * k;
                    bfloat16_t *part_v_base = v + h_step * k * c;
                    bfloat16_t *part_v_avg_base = v_avg + h_step * q * c + (q_step * q_stride + q_idx) * c;
                    kutacc::msa_attention_sme_4_vsl_impl(part_logits_base, part_v_base, part_v_avg_base, h, q, k, b, c);
                    q_idx += Q_BLOCK_4;
                }
                else
                {
                    break;
                }
            }
        });
    }
}
void msa_attention_linear(bfloat16_t *act, bfloat16_t *pair_act, bfloat16_t *pair_logits, bfloat16_t *v, bfloat16_t *gate, bfloat16_t *pair_logits_w, bfloat16_t *value_w,\
    bfloat16_t *gating_w, int64_t batch, int64_t seq_len, int64_t nchannels, int64_t nheads, int64_t noutput_channels, int64_t output_batches)
{
    {
        auto [tm, tn] = kutacc::compute_tm_tn(output_batches * seq_len, nheads);
        kutacc::MatrixTilingBlock tiling(tm, tn, noutput_channels);
        std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[output_batches * seq_len * noutput_channels]);
        std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[noutput_channels * nheads]);
        std::unique_ptr<__bf16[]> tmpc(new __bf16[output_batches * seq_len * nheads]);
        kutacc::bf16_gemm_pack(output_batches * seq_len, noutput_channels, std::get<0>(tiling), std::get<2>(tiling), (__bf16 *)pair_act, pack_a.get());
        kutacc::bf16_gemm_pack(nheads, noutput_channels, std::get<1>(tiling), std::get<2>(tiling), (__bf16 *)pair_logits_w, pack_b.get());
        kutacc::bf16_packed_gemm(output_batches * seq_len, nheads, noutput_channels, tiling, 
            pack_a.get(),
            pack_b.get(),
            (__bf16 *)pair_logits,
            tmpc.get());
    }
    auto [tm, tn] = kutacc::compute_tm_tn(batch * seq_len, nchannels);
    kutacc::MatrixTilingBlock tiling(tm, tn, nchannels);
    std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[batch * seq_len * nchannels]);
    kutacc::bf16_gemm_pack(batch * seq_len, nchannels, std::get<0>(tiling), std::get<2>(tiling), (__bf16 *)act, pack_a.get());
    {
        std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[nchannels * nchannels]);
        std::unique_ptr<__bf16[]> tmpc(new __bf16[batch * seq_len * nchannels]);
        kutacc::bf16_gemm_pack(nchannels, nchannels, std::get<1>(tiling), std::get<2>(tiling), (__bf16 *)value_w, pack_b.get());
        kutacc::bf16_packed_gemm(batch * seq_len, nchannels, nchannels, tiling, 
            pack_a.get(),
            pack_b.get(),
            (__bf16 *)v,
            tmpc.get());
    }
    {
        std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[nchannels * nchannels]);
        std::unique_ptr<__bf16[]> tmpc(new __bf16[batch * seq_len * nchannels]);
        kutacc::bf16_gemm_pack(nchannels, nchannels, std::get<1>(tiling), std::get<2>(tiling), (__bf16 *)gating_w, pack_b.get());
        kutacc::bf16_packed_gemm(batch * seq_len, nchannels, nchannels, tiling, 
            pack_a.get(),
            pack_b.get(),
            (__bf16 *)gate,
            tmpc.get());
    }
}
void msa_attention_out(bfloat16_t *v, bfloat16_t *v_avg, bfloat16_t *gate, bfloat16_t *out, int64_t batch, int64_t seq_len, int64_t nchannels)
{
    {
        kutacc::parallel_for(0, batch, 1, [&](int64_t start, int64_t end) {
        for (int64_t bi = start ; bi < end; bi++) {
            int64_t vl = static_cast<int64_t>(svcntw());
            svbool_t pg = svptrue_b32();
        
            for (int64_t qi = 0 ; qi < seq_len; qi++) {
                // 获取当前v_avg和gate的数据指针
                auto wavg_data = v_avg + bi * seq_len * nchannels + qi * nchannels;
                auto gate_data = gate + bi * seq_len * nchannels  + qi * nchannels;
                
                // 使用循环处理所有通道，每次处理4个向量长度
                for (int64_t ci = 0; ci < nchannels; ci += vl * 4) {
                    // 加载v_avg数据
                    svfloat32_t w0 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci);
                    svfloat32_t w1 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl);
                    svfloat32_t w2 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl * 2);
                    svfloat32_t w3 = kutacc::svld1<float, bfloat16_t>(pg, wavg_data + ci + vl * 3);
                    
                    // 加载gate数据
                    svfloat32_t g0 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci);
                    svfloat32_t g1 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl);
                    svfloat32_t g2 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl * 2);
                    svfloat32_t g3 = kutacc::svld1<float, bfloat16_t>(pg, gate_data + ci + vl * 3);
                    
                    // 计算v_avg *= sigmoid(gate)
                    w0 = svmul_x(pg, w0, kutacc::Sigmoid::call(pg, g0));
                    w1 = svmul_x(pg, w1, kutacc::Sigmoid::call(pg, g1));
                    w2 = svmul_x(pg, w2, kutacc::Sigmoid::call(pg, g2));
                    w3 = svmul_x(pg, w3, kutacc::Sigmoid::call(pg, g3));
                    
                    // 存回v_avg
                    kutacc::svst1<bfloat16_t, float>(pg, wavg_data + ci, w0);
                    kutacc::svst1<bfloat16_t, float>(pg, wavg_data + ci + vl, w1);
                    kutacc::svst1<bfloat16_t, float>(pg, wavg_data + ci + vl * 2, w2);
                    kutacc::svst1<bfloat16_t, float>(pg, wavg_data + ci + vl * 3, w3);
            }
        } 
    }
});
    }
    {
        auto [tm, tn] = kutacc::compute_tm_tn(batch * seq_len, nchannels);
        kutacc::MatrixTilingBlock tiling(tm, tn, nchannels);
        std::unique_ptr<bfloat16_t[]> pack_a(new bfloat16_t[batch * seq_len * nchannels]);
        std::unique_ptr<bfloat16_t[]> pack_b(new bfloat16_t[nchannels * nchannels]);
        std::unique_ptr<__bf16[]> tmpc(new __bf16[batch * seq_len * nchannels]);
        kutacc::bf16_gemm_pack(batch * seq_len, nchannels, std::get<0>(tiling), std::get<2>(tiling), (__bf16 *)v_avg, pack_a.get());
        kutacc::bf16_gemm_pack(nchannels, nchannels, std::get<1>(tiling), std::get<2>(tiling), (__bf16 *)v, pack_b.get());
        kutacc::bf16_packed_gemm(batch * seq_len, nchannels, nchannels, tiling, 
            pack_a.get(),
            pack_b.get(),
            (__bf16 *)out,
            tmpc.get());
    }
}
} //namespace kutacc