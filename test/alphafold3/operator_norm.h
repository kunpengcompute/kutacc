#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>
#include <cstdint>

constexpr float LN_EPS = 1e-5f;
constexpr float DIV_EPS = 1e-3f;
#define OFFSET5(a,b,c,d,e, A,B,C,D,E) ((int)a*B*C*D*E + (int)b*C*D*E + (int)c*D*E + (int)d*E + e)

void pack_weight_task_bf16(float *weight, bfloat16_t *packed_weight,int M, int N)
{
    const int vscalew = svcntw();
    const int vscaleh = svcnth();
    svbool_t pt16 = svptrue_b16();
    svbool_t pt32 = svptrue_b32();
    int max_threads = kutacc::get_thread_num();
    const int row_step = 2;
    const int col_step = vscaleh;
    const int one_task = row_step * col_step;
    int all_task = (M * N) / one_task;
    int avg_task =  all_task / max_threads;
    int busy_thread = all_task % max_threads;

	kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end){
        int tid = kutacc::get_thread_id();
        int task_num, task_start_idx, task_end_idx;
        if(tid < busy_thread)
        {
            task_num = avg_task+1;
            task_start_idx = tid*task_num;
            task_end_idx = task_start_idx+task_num;
        }
        else
        {
            task_num = avg_task;
            task_start_idx = busy_thread*(task_num+1)+(tid-busy_thread)*task_num;
            task_end_idx = task_start_idx+task_num;
        }

        int row_task_num = N / col_step;
        for(int task_idx = task_start_idx; task_idx < task_end_idx; task_idx += 1)
        {
            int row_index = (task_idx / row_task_num) * row_step;
            int col_index = (task_idx % row_task_num) * col_step;
            svfloat32_t data_0, data_1, data_2, data_3;
            data_0 = svld1(pt32, weight + row_index * N + col_index);
            data_1 = svld1(pt32, weight + row_index * N + col_index + vscalew);
            data_2 = svld1(pt32, weight + (row_index + 1) * N + col_index);
            data_3 = svld1(pt32, weight + (row_index + 1) * N + col_index + vscalew);
            svbfloat16_t unz1,unz2,z1,z2;
            unz1 = svuzp1_bf16(svcvt_bf16_f32_z(pt32,data_0), svcvt_bf16_f32_z(pt32,data_1));
            unz2 = svuzp1_bf16(svcvt_bf16_f32_z(pt32,data_2), svcvt_bf16_f32_z(pt32,data_3));
            z1 = svzip1_bf16(unz1, unz2); 
            z2 = svzip2_bf16(unz1, unz2); 
            svst1_bf16(pt16,packed_weight + col_index * M + row_index * vscalew, z1); 
            svst1_bf16(pt16,packed_weight + (col_index + vscalew)* M + row_index*vscalew, z2);
        }
    });
}

void layer_norm_manual(
    const float* x, 
    const float* weight, 
    float* out,
    int B, int Dqk, int D, float eps,
    bool use_weight
) {
    for (int b = 0; b < B; b++) {
        for (int q = 0; q < Dqk; q++) {
            int base = b * Dqk * D + q * D;
            const float* vec = x + base;
            float* out_vec = out + base;

            // 均值
            float mean = 0.0f;
            for (int d = 0; d < D; d++)
                mean += vec[d];
            mean /= D;

            // 方差
            float var = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = vec[d] - mean;
                var += diff * diff;
            }
            var /= D;

            // 归一化
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int d = 0; d < D; d++) {
                float val = (vec[d] - mean) * inv_std;
                if (use_weight) val *= weight[d];
                out_vec[d] = val;
            }
        }
    }
}

void matmul_3d_2d(
    const float* a, const float* b, float* out,
    int B, int m, int n, int k
) {
    kutacc::parallel_for(0, B*m, 1, [&](int64_t start, int64_t end) {
        for(int task_idx = start; task_idx < end; task_idx++){
            int b_idx = task_idx / m;
            int m_idx = task_idx - m * b_idx;
            int base = b_idx * m * k + m_idx * k;
            int out_base = b_idx * m * n + m_idx * n;
            for (int n_idx = 0; n_idx < n; n_idx++) {
                float sum = 0.0f;
                for (int k_idx = 0; k_idx < k; k_idx++) {
                    sum += a[base + k_idx] * b[k_idx * n + n_idx];
                }
                out[out_base + n_idx] = sum;
            }
        }
    });
}
void mul_broadcast(const float* matrix1, const float* matrix2, float* out, int B, int size){
    for(int i = 0; i < B; i++){
        int base = i * size;
        for(int j = 0; j < size; j++){
            out[base + j] = matrix1[base + j] * matrix2[j];
        }
    }
}

void sigmoid_manual(const float* x, float* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

void add_bias_broadcast(
    float* tensor, 
    const float* bias,
    int B, int D
) {
    for (int b = 0; b < B; b++) {
        int base = b * D;
        for (int d = 0; d < D; d++) {
            tensor[base + d] += bias[d];
        }
    }
}


void elementwise_fuse(
    const float* sig,
    const float* x_norm,
    const float* bias_add,
    float* out,
    int N_s, int B, int Dqk, int D
) {
    for (int s = 0; s < N_s; s++) {
        for (int b = 0; b < B; b++) {
            for (int q = 0; q < Dqk; q++) {
                int base_b = b * Dqk * D + q * D;
                int base_s = s * B * Dqk * D + base_b;

                for (int d = 0; d < D; d++) {
                    int idx_s = base_s + d;
                    int idx_b = base_b + d;

                    out[idx_s] = sig[idx_b] * x_norm[idx_s] + bias_add[idx_b];
                }
            }
        }
    }
}

void softmax_last_dim(const float* x, float* out, int B, int H, int Nq, int Nkv)
{
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int q = 0; q < Nq; q++) {
                // 取当前行指针
                int base = OFFSET5(b,h,q,0,0, B,H,Nq,Nkv,1);
                const float* in_row = x + base;
                float* out_row = out + base;

                // 找最大值（稳定softmax）
                float max_val = -1e20;
                for (int k = 0; k < Nkv; k++) {
                    if (in_row[k] > max_val) max_val = in_row[k];
                }

                // 求和
                float sum = 0.0f;
                for (int k = 0; k < Nkv; k++) {
                    sum += expf(in_row[k] - max_val);
                }

                // 计算softmax
                float inv_sum = 1.0f / sum;
                for (int k = 0; k < Nkv; k++) {
                    out_row[k] = expf(in_row[k] - max_val) * inv_sum;
                }
            }
        }
    }
}

void compute_logits(
    const float* q,
    const float* k,
    const float* pair_logits,
    const float* bias,
    float* logits,
    int N_d, int N_a, int H, int Nq, int Nkv, int D, float scale
)
{
    for (int d = 0; d < N_d; d++) {
        for (int na = 0; na < N_a; na++) {
            for (int h = 0; h < H; h++) {
                for (int q_idx = 0; q_idx < Nq; q_idx++) {
                    for (int k_idx = 0; k_idx < Nkv; k_idx++) {

                        // 计算 q * scale 与 k 的点积
                        float sum = 0.0f;
                        for (int c = 0; c < D; c++) {
                            int q_idx5 = OFFSET5(d, na, q_idx, h, c, N_d,N_a,Nq,H,D);
                            int k_idx5 = OFFSET5(d, na, k_idx, h, c, N_d,N_a,Nkv,H,D);
                            sum += q[q_idx5] * scale * k[k_idx5];
                        }

                        // 输出位置 [N_d, N_a, H, Nq, Nkv]
                        int out_idx = OFFSET5(d, na, h, q_idx, k_idx, N_d,N_a,H,Nq,Nkv);
                        int b_idx = OFFSET5(0, na, q_idx, k_idx,0, 1,N_a,Nq,Nkv,1);
                        int p_idx = OFFSET5(0, na, h, q_idx, k_idx, 1,N_a,H,Nq,Nkv);

                        logits[out_idx] = sum + bias[b_idx] + pair_logits[p_idx];
                    }
                }
            }
        }
    }
}

void weighted_sum_v(
    const float* weights,
    const float* v,
    float* out,
    int N_d, int N_a, int H, int Nq, int Nkv, int D
)
{
    // 输出先清零
    int total = N_d * N_a * Nq * H * D;
    for (int i = 0; i < total; i++) out[i] = 0.0f;

    // 计算加权和
    for (int d = 0; d < N_d; d++) {
        for (int na = 0; na < N_a; na++) {
            for (int h = 0; h < H; h++) {
                for (int q_idx = 0; q_idx < Nq; q_idx++) {
                    for (int k_idx = 0; k_idx < Nkv; k_idx++) {

                        int w_idx = OFFSET5(d,na,h,q_idx,k_idx, N_d,N_a,H,Nq,Nkv);
                        float w = weights[w_idx];

                        for (int c = 0; c < D; c++) {
                            int v_idx = OFFSET5(d,na,k_idx,h,c, N_d,N_a,Nkv,H,D);
                            int o_idx = OFFSET5(d,na,q_idx,h,c, N_d,N_a,Nq,H,D);
                            out[o_idx] += w * v[v_idx];
                        }
                    }
                }
            }
        }
    }
}
void cvt_fp32_bf16(const float* input, __bf16* output, int size) {
    for (int i = 0; i < size; i++) {
        __attribute__((aligned(2))) uint16_t bf16_val;

        // 安全读取 float 位模式
        uint32_t f_bits = *reinterpret_cast<const uint32_t*>(&input[i]);
        bf16_val = (uint16_t)(f_bits >> 16);

        // 安全写入 __bf16（唯一合法方式）
        memcpy(&output[i], &bf16_val, 2);
    }
}
void cvt_bf16_fp32(const bfloat16_t* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        uint16_t bf16_val = *reinterpret_cast<const uint16_t*>(&input[i]);
        uint32_t f32_bits = (uint32_t)bf16_val << 16;
        output[i] = *reinterpret_cast<float*>(&f32_bits);
    }
}

// input shape [D, A, B]
// output shape [B, D, A] 等价 permute(2,0,1)
void transpose_DAB_to_BDA_sve(const float* input, float* output,
                              int D, int A, int B)
{
    svbool_t pg = svptrue_b32();
    int vec_len = svcntw();

    // 输出维度顺序 B -> D -> A
    for (int b = 0; b < B; b++)
    {
        for (int d = 0; d < D; d++)
        {
            for (int a = 0; a < A; a++)
            {
                int d_off = 0;
                // SVE 向量化批量拷贝
                for (; d_off + vec_len <= D; d_off += vec_len)
                {
                    // 输入: [D, A, B] -> d, a, b
                    long long in_idx  = (long long)(d + d_off) * A * B + (long long)a * B + b;
                    // 输出: [B, D, A] -> b, d, a
                    long long out_idx = (long long)b * D * A + (long long)(d + d_off) * A + a;

                    svfloat32_t vec = svld1_f32(pg, input + in_idx);
                    svst1_f32(pg, output + out_idx, vec);
                }
                // 剩余不足向量长度的标量处理
                for (; d_off < D; d_off++)
                {
                    long long in_idx  = (long long)(d + d_off) * A * B + (long long)a * B + b;
                    long long out_idx = (long long)b * D * A + (long long)(d + d_off) * A + a;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

void transpose_BNHD_to_BHND_sve(const float* input, float* output,
                                int B, int N, int H, int D)
{
    svbool_t pg = svptrue_b32();
    int vec_len = svcntw();

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int n = 0; n < N; n++) {

                int d = 0;
                // 向量化搬运
                for (; d + vec_len <= D; d += vec_len) {
                    int in_idx  = b*N*H*D + n*H*D + h*D + d;
                    int out_idx = b*H*N*D + h*N*D + n*D + d;

                    svfloat32_t v = svld1_f32(pg, input + in_idx);
                    svst1_f32(pg, output + out_idx, v);
                }
                // 尾部标量
                for (; d < D; d++) {
                    int in_idx  = b*N*H*D + n*H*D + h*D + d;
                    int out_idx = b*H*N*D + h*N*D + n*D + d;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

void einsum_ckj_cki_to_cij(
    const float* a, const float* b, float* out,
    int batch, int C, int K, int I, int J
) {
    // 各切片尺寸
    long long slice_c_a = (long long)K * J;
    long long slice_c_b = (long long)K * I;
    long long slice_c_out = (long long)I * J;

    long long batch_c = (long long)batch * C;
    for (long long bc = 0; bc < batch_c; bc++) {
        // bc = batch_idx * C + c_idx
        const float* a_c = a + bc * slice_c_a;
        const float* b_c = b + bc * slice_c_b;
        float* out_c = out + bc * slice_c_out;

        // out_c[i,j] = sum_k a_c[k,j] * b_c[k,i]
        for (int i = 0; i < I; i++) {
            for (int j = 0; j < J; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    float aval = a_c[k * J + j];
                    float bval = b_c[k * I + i];
                    sum += aval * bval;
                }
                out_c[i * J + j] = sum;
            }
        }
    }
}

void einsum_cik_cjk_to_cij(
    const float* a, const float* b, float* out,
    int batch, int C, int I, int J, int K
) {
    long long slice_c_a = (long long)I * K;
    long long slice_c_b = (long long)J * K;
    long long slice_c_out = (long long)I * J;

    long long batch_c = (long long)batch * C;
    for (long long bc = 0; bc < batch_c; bc++) {
        const float* a_c = a + bc * slice_c_a;
        const float* b_c = b + bc * slice_c_b;
        float* out_c = out + bc * slice_c_out;

        // out_c[i,j] = sum_k a_c[i,k] * b_c[j,k]
        for (int i = 0; i < I; i++) {
            for (int j = 0; j < J; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    float aval = a_c[i * K + k];
                    float bval = b_c[j * K + k];
                    sum += aval * bval;
                }
                out_c[i * J + j] = sum;
            }
        }
    }
}



// LayerNorm [B,T,D] -> [B,T,D]
void layer_norm_chunk(
    const float* act_in, float* act_out,
    const float* ln_w, const float* ln_b,
    int B, int T, int D
) {
    long long strideBT = (long long)T * D;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* ptr = act_in + (long long)b * strideBT + (long long)t * D;
            float sum = 0.f, sum_sq = 0.f;
            for (int d = 0; d < D; d++) {
                float v = ptr[d];
                sum += v;
                sum_sq += v * v;
            }
            float mean = sum / D;
            float var = sum_sq / D - mean * mean;
            float rsqrt = 1.f / sqrt(var + LN_EPS);

            float* out_ptr = act_out + (long long)b * strideBT + (long long)t * D;
            for (int d = 0; d < D; d++) {
                float val = (ptr[d] - mean) * rsqrt;
                out_ptr[d] = val * ln_w[d] + ln_b[d];
            }
        }
    }
}

// 线性投影 weight [D,C], input [B,T,D] -> output [B,T,C]
void linear_proj(
    const float* act_ln, const float* weight, float* out,
    int B, int T, int D, int C
) {
    long long strideBT = (long long)T * D;
    long long strideTC = (long long)T * C;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* a_ptr = act_ln + (long long)b * strideBT + (long long)t * D;
            float* o_ptr = out + (long long)b * strideTC + (long long)t * C;
            for (int c = 0; c < C; c++) {
                float s = 0.f;
                for (int d = 0; d < D; d++) {
                    s += a_ptr[d] * weight[(long long)d * C + c];
                }
                o_ptr[c] = s;
            }
        }
    }
}

/**
 * compute_chunk 核心计算
 * left_act:  [B, T, C]
 * right_act: [B, T, C]
 * output_w:  [C, C, F]
 * output_b:  [F]
 * out_buf:   [T, T, F] 最终输出 permute后结果
 * workspace 三段缓存：
    perm_left  : [B, C, T]    permute(left_act, (0,2,1))
    einsum1    : [T, C, C, T] einsum('acb,ade->dceb') 输出 [x,32,32,x]
    temp_TTF   : [T, T, F]    第二次einsum输出 [x,x,128]
 */
void compute_chunk(
    const float* left_act,
    const float* right_act,
    const float* output_w,
    const float* output_b,
    float* out_buf,
    float* workspace,
    int B, int T, int C, int F
) {
    float* perm_left = workspace;
    float* einsum1 = perm_left + (long long)B * C * T;
    float* temp_TTF = einsum1 + (long long)T * C * C * T;

    // Step1: permute left_act [B,T,C] -> perm_left [B,C,T] (a,c,b)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                long long src = (long long)b * T * C + (long long)t * C + c;
                long long dst = (long long)b * C * T + (long long)c * T + t;
                perm_left[dst] = left_act[src];
            }
        }
    }

    // Step2: einsum('acb,ade->dceb')
    // X: perm_left [a,c,b] = [B,C,T]
    // Y: right_act [a,d,e] = [B,T,C]
    // 求和维度 a(B)，输出 [d,c,e,b] = [T,C,C,T]
    // einsum1[d, c1, c2, b] = sum_{a=B} perm_left[a,c1,b] * right_act[a,d,c2]
    for (int d = 0; d < T; d++) {      // d = T
        for (int c1 = 0; c1 < C; c1++) {// c = C
            for (int c2 = 0; c2 < C; c2++) {// e = C
                for (int b = 0; b < T; b++) {// b = T
                    float sum = 0.f;
                    for (int a = 0; a < B; a++) { // 收缩维度 B
                        long long l_idx = (long long)a * C * T + (long long)c1 * T + b;
                        long long r_idx = (long long)a * T * C + (long long)d * C + c2;
                        sum += perm_left[l_idx] * right_act[r_idx];
                    }
                    // einsum1 [d, c1, c2, b]
                    long long out_idx = (long long)d * C * C * T + (long long)c1 * C * T + (long long)c2 * T + b;
                    einsum1[out_idx] = sum;
                }
            }
        }
    }

    // Step3: einsum('dceb,cef->dbf') + bias
    // X: einsum1 [d,c,e,b] = [T,C,C,T]
    // W: output_w [c,e,f] = [C,C,F]
    // 输出 [d,b,f] = [T,T,F]
    for (int d = 0; d < T; d++) {
        for (int b = 0; b < T; b++) {
            for (int f = 0; f < F; f++) {
                float sum = output_b[f];
                for (int c1 = 0; c1 < C; c1++) {
                    for (int c2 = 0; c2 < C; c2++) {
                        long long e_idx = (long long)d * C * C * T + (long long)c1 * C * T + (long long)c2 * T + b;
                        long long w_idx = (long long)c1 * C * F + (long long)c2 * F + f;
                        sum += einsum1[e_idx] * output_w[w_idx];
                    }
                }
                long long tmp_idx = (long long)d * T * F + (long long)b * F + f;
                temp_TTF[tmp_idx] = sum;
            }
        }
    }

    // Step4: permute(1,0,2) 输入 [d,b,f]=[T,T,F]，交换0/1维
    // src [d,b,f] → dst [b,d,f]
    for (int d = 0; d < T; d++) {
        for (int b = 0; b < T; b++) {
            for (int f = 0; f < F; f++) {
                long long src = (long long)d * T * F + (long long)b * F + f;
                long long dst = (long long)b * T * F + (long long)d * F + f;
                out_buf[dst] = temp_TTF[src];
            }
        }
    }
}

/**
 * 顶层算子 outer_product_mean_forward
输入shape：
    act      [B, T, D]
    mask     [B, T]
    ln_w     [D]
    ln_b     [D]
    leftW    [D, C]
    rightW   [D, C]
    outW     [C, C, F]
    outB     [F]
输出shape：
    out      [T, T, F]
    norm_out [B, B]
workspace总大小 = B*C*T + T*C*C*T + T*T*F
*/
void outer_product_mean_forward(
    const float* act,
    const float* mask,
    const float* ln_w,
    const float* ln_b,
    const float* leftW,
    const float* rightW,
    const float* outW,
    const float* outB,
    float* out,
    float* norm_out,
    float* workspace,
    int B, int T, int D, int C, int F
) {
    alignas(64) float mask_bt[B * T];
    alignas(64) float act_ln[B * T * D];
    alignas(64) float left_act[B * T * C];
    alignas(64) float right_act[B * T * C];

    // 1. mask拷贝
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            mask_bt[(long long)b * T + t] = mask[(long long)b * T + t];
        }
    }

    // 2. LayerNorm
    layer_norm_chunk(act, act_ln, ln_w, ln_b, B, T, D);

    // 3. left proj * mask
    linear_proj(act_ln, leftW, left_act, B, T, D, C);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float m = mask_bt[(long long)b * T + t];
            for (int c = 0; c < C; c++) {
                long long idx = (long long)b * T * C + (long long)t * C + c;
                left_act[idx] *= m;
            }
        }
    }

    // 4. right proj * mask
    linear_proj(act_ln, rightW, right_act, B, T, D, C);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float m = mask_bt[(long long)b * T + t];
            for (int c = 0; c < C; c++) {
                long long idx = (long long)b * T * C + (long long)t * C + c;
                right_act[idx] *= m;
            }
        }
    }

    // 5. 核心einsum计算
    compute_chunk(left_act, right_act, outW, outB, out, workspace, B, T, C, F);

    // 6. norm = einsum('abc,adc->bdc') mask[B,T]
    for (int b1 = 0; b1 < B; b1++) {
        for (int b2 = 0; b2 < B; b2++) {
            float sum = 0.f;
            for (int t = 0; t < T; t++) {
                sum += mask_bt[(long long)b1 * T + t] * mask_bt[(long long)b2 * T + t];
            }
            norm_out[(long long)b1 * B + b2] = sum;
        }
    }

    // 7. 归一化除法 out[t1,t2,f] /= (eps + norm[b,b])
    for (int t1 = 0; t1 < T; t1++) {
        for (int t2 = 0; t2 < T; t2++) {
            float denom = DIV_EPS + norm_out[(long long)B * B / 2]; // norm[b,b]，B固定1024取自身对角
            for (int f = 0; f < F; f++) {
                long long idx = (long long)t1 * T * F + (long long)t2 * F + f;
                out[idx] /= denom;
            }
        }
    }
}