#include "kutacc.h"
#include "common.h"
#include "math/fast_exp.h"
namespace kutacc{
// namespace alphafold3{

void pack_left_task_2vl_2vl_bf16(const bfloat16_t *matrix,bfloat16_t *matrix_pack,int M, int N)
{
    svbool_t pt16 = svptrue_b16();
    int max_threads = kutacc::get_thread_num();
    const int row_step = SVL_bf16;
    const int col_step = 2 * SVL_bf16;
    const int one_task = row_step * col_step;
    // const int prf_stride = 64;
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

                // svprfh(pt16, matrix + (outer_row_index + row_index) * N + outer_col_index + prf_stride, svprfop::SV_PLDL1KEEP);
                // svprfh(pt16, matrix + (outer_row_index + row_index) * N + outer_col_index + SVL_bf16 + prf_stride, svprfop::SV_PLDL1KEEP);
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

template <typename T>
void block_bgemm_4vl_1vl_impl(bfloat16_t *mat1, bfloat16_t *mat2, T *results,int M, int N, int K, int c_stride)
{
    svbool_t pt16 = svptrue_b16();
    svbool_t pt16_w = svwhilelt_b16(0,16);
    svbool_t pt32 = svptrue_b32();
    const int row_step = 4 * SVL_fp32;
    const int row_single_step = SVL_fp32;
    const int col_step = SVL_fp32;
    SME_ON();
    for(int row_index = 0; row_index < M; row_index += row_step)
    {
        for(int col_index = 0; col_index < N; col_index += col_step)
        {
            bfloat16_t* mat1_base0 = mat1 + row_index * K;
            bfloat16_t* mat1_base1 = mat1 + (row_index + row_single_step ) * K;
            bfloat16_t* mat1_base2 = mat1 + (row_index + 2 * row_single_step ) * K;
            bfloat16_t* mat1_base3 = mat1 + (row_index + 3 * row_single_step ) * K;
            bfloat16_t* mat2_base0 = mat2 + col_index * K ;

            svzero_za();
            for(int k_idx = 0; k_idx < SVL_fp32 * K; k_idx += SVL_bf16)
            {
                svbfloat16_t colA0 = svld1_bf16(pt16,mat1_base0 + k_idx); 
                svbfloat16_t colA1 = svld1_bf16(pt16,mat1_base1 + k_idx); 
                svbfloat16_t colA2 = svld1_bf16(pt16,mat1_base2 + k_idx); 
                svbfloat16_t colA3 = svld1_bf16(pt16,mat1_base3 + k_idx); 
                svbfloat16_t rowB0 = svld1_bf16(pt16,mat2_base0 + k_idx);

                svmopa_za32_bf16_m(0,pt16,pt16,colA0,rowB0);
                svmopa_za32_bf16_m(1,pt16,pt16,colA1,rowB0);
                svmopa_za32_bf16_m(2,pt16,pt16,colA2,rowB0);
                svmopa_za32_bf16_m(3,pt16,pt16,colA3,rowB0);
            }
            #pragma unroll(8)
            for (int inner_row_idx=0; inner_row_idx < SVL_fp32; inner_row_idx++) 
            {
                svfloat32_t data_0 = svread_hor_za32_f32_m(svfloat32_t(),pt32,0,static_cast<uint32_t>(inner_row_idx));
                svfloat32_t data_1 = svread_hor_za32_f32_m(svfloat32_t(),pt32,1,static_cast<uint32_t>(inner_row_idx));
                svfloat32_t data_2 = svread_hor_za32_f32_m(svfloat32_t(),pt32,2,static_cast<uint32_t>(inner_row_idx));
                svfloat32_t data_3 = svread_hor_za32_f32_m(svfloat32_t(),pt32,3,static_cast<uint32_t>(inner_row_idx));
                if (sizeof(T) == 2){
                    svbfloat16_t results_bf16_0 = svuzp1_bf16(svcvt_bf16_f32_z(pt32, data_0), svdup_bf16(0.0));
                    svbfloat16_t results_bf16_1 = svuzp1_bf16(svcvt_bf16_f32_z(pt32, data_1), svdup_bf16(0.0));
                    svbfloat16_t results_bf16_2 = svuzp1_bf16(svcvt_bf16_f32_z(pt32, data_2), svdup_bf16(0.0));
                    svbfloat16_t results_bf16_3 = svuzp1_bf16(svcvt_bf16_f32_z(pt32, data_3), svdup_bf16(0.0));

                    svst1(pt16_w, ((bfloat16_t*)results) + (row_index + inner_row_idx) * c_stride + col_index, results_bf16_0);
                    svst1(pt16_w, ((bfloat16_t*)results) + (row_index + inner_row_idx + SVL_fp32) * c_stride + col_index , results_bf16_1);
                    svst1(pt16_w, ((bfloat16_t*)results) + (row_index + inner_row_idx + 2 * SVL_fp32) * c_stride + col_index , results_bf16_2);
                    svst1(pt16_w, ((bfloat16_t*)results) + (row_index + inner_row_idx + 3 * SVL_fp32) * c_stride + col_index , results_bf16_3);
                }else if (sizeof(T) == 4){
                    svst1(pt32, ((float*)results) + (row_index + inner_row_idx) * c_stride + col_index, data_0);
                    svst1(pt32, ((float*)results) + (row_index + inner_row_idx + SVL_fp32) * c_stride + col_index , data_1);
                    svst1(pt32, ((float*)results) + (row_index + inner_row_idx + 2 * SVL_fp32) * c_stride + col_index , data_2);
                    svst1(pt32, ((float*)results) + (row_index + inner_row_idx + 3 * SVL_fp32) * c_stride + col_index , data_3);
                }
            }
        }
    }
    SME_OFF();
}

void block_linear_bias_sigmoid_4vl_1vl_impl(bfloat16_t *mat1, bfloat16_t *mat2, float *bias, float* results, bfloat16_t* results_bf16, \
    int M, int N, int K, int samples, int bucket, int c_stride)
{
    svbool_t pt16 = svptrue_b16();
    svbool_t pt16_w = svwhilelt_b16(0,16);
    svbool_t pt32 = svptrue_b32();
    const int row_step = 4 * SVL_fp32;
    const int row_single_step = SVL_fp32;
    const int col_step = SVL_fp32;
    const int results_single_step_stride = SVL_fp32 * c_stride;
    SME_ON();
    for(int row_index = 0; row_index < M; row_index += row_step)
    {
        for(int col_index = 0; col_index < N; col_index += col_step)
        {
            bfloat16_t* mat1_base0 = mat1 + row_index * K;
            bfloat16_t* mat1_base1 = mat1 + (row_index + row_single_step ) * K;
            bfloat16_t* mat1_base2 = mat1 + (row_index + 2 * row_single_step ) * K;
            bfloat16_t* mat1_base3 = mat1 + (row_index + 3 * row_single_step ) * K;
            bfloat16_t* mat2_base0 = mat2 + col_index * K ;

            svzero_za();
            for(int k_idx = 0; k_idx < SVL_fp32 * K; k_idx += SVL_bf16)
            {
                svbfloat16_t colA0 = svld1_bf16(pt16,mat1_base0 + k_idx); 
                svbfloat16_t colA1 = svld1_bf16(pt16,mat1_base1 + k_idx); 
                svbfloat16_t colA2 = svld1_bf16(pt16,mat1_base2 + k_idx); 
                svbfloat16_t colA3 = svld1_bf16(pt16,mat1_base3 + k_idx); 
                svbfloat16_t rowB0 = svld1_bf16(pt16,mat2_base0 + k_idx);

                svmopa_za32_bf16_m(0,pt16,pt16,colA0,rowB0);
                svmopa_za32_bf16_m(1,pt16,pt16,colA1,rowB0);
                svmopa_za32_bf16_m(2,pt16,pt16,colA2,rowB0);
                svmopa_za32_bf16_m(3,pt16,pt16,colA3,rowB0);
            }
            #pragma unroll(2)
            for (int inner_row_idx=0; inner_row_idx < SVL_fp32; inner_row_idx++) 
            {
                svfloat32_t bias_0 = svld1_f32(pt32,bias + col_index);
                svfloat32_t zero_f32 = svdup_f32_z(pt32,0);
                svfloat32_t one_f32 = svdup_f32_z(pt32,1);

                svfloat32_t data_0 = svread_hor_za32_f32_m(svfloat32_t(),pt32,0,static_cast<uint32_t>(inner_row_idx));
                svfloat32_t data_1 = svread_hor_za32_f32_m(svfloat32_t(),pt32,1,static_cast<uint32_t>(inner_row_idx));
                svfloat32_t data_2 = svread_hor_za32_f32_m(svfloat32_t(),pt32,2,static_cast<uint32_t>(inner_row_idx));
                svfloat32_t data_3 = svread_hor_za32_f32_m(svfloat32_t(),pt32,3,static_cast<uint32_t>(inner_row_idx));
                data_0 = svadd_f32_x(pt32,data_0,bias_0);
                data_0 = svneg_f32_m(zero_f32,pt32,data_0);
                data_0 = kutacc::fast_exp(pt32,data_0);
                data_0 = svadd_f32_x(pt32,data_0,one_f32);
                data_0 = svdiv_f32_m(pt32,one_f32,data_0);

                data_1 = svadd_f32_x(pt32,data_1,bias_0);
                data_1 = svneg_f32_m(zero_f32,pt32,data_1);
                data_1 = kutacc::fast_exp(pt32,data_1);
                data_1 = svadd_f32_x(pt32,data_1,one_f32);
                data_1 = svdiv_f32_m(pt32,one_f32,data_1);

                data_2 = svadd_f32_x(pt32,data_2,bias_0);
                data_2 = svneg_f32_m(zero_f32,pt32,data_2);
                data_2 = kutacc::fast_exp(pt32,data_2);
                data_2 = svadd_f32_x(pt32,data_2,one_f32);
                data_2 = svdiv_f32_m(pt32,one_f32,data_2);
                
                data_3 = svadd_f32_x(pt32,data_3,bias_0);
                data_3 = svneg_f32_m(zero_f32,pt32,data_3);
                data_3 = kutacc::fast_exp(pt32,data_3);
                data_3 = svadd_f32_x(pt32,data_3,one_f32);
                data_3 = svdiv_f32_m(pt32,one_f32,data_3);
                for(int sample_idx = 0; sample_idx < samples; sample_idx++){
                    float* results_base = results + sample_idx * bucket * c_stride + (row_index + inner_row_idx) * c_stride + col_index;
                    bfloat16_t* results_bf16_base = results_bf16 + sample_idx * bucket * c_stride + (row_index + inner_row_idx) * c_stride + col_index;
                    svfloat32_t results_0 = svld1_f32(pt32,results_base);
                    svfloat32_t results_1 = svld1_f32(pt32,results_base + results_single_step_stride);
                    svfloat32_t results_2 = svld1_f32(pt32,results_base + 2 * results_single_step_stride);
                    svfloat32_t results_3 = svld1_f32(pt32,results_base + 3 * results_single_step_stride);

                    svfloat32_t results_fp32_0 = svmul_f32_m(pt32,data_0,results_0);
                    svbfloat16_t results_bf16_0 = svuzp1_bf16(svcvt_bf16_f32_z(pt32, results_fp32_0), svdup_bf16(0.0));
                    svst1_bf16(pt16_w, results_bf16_base, results_bf16_0);

                    svfloat32_t results_fp32_1 = svmul_f32_m(pt32,data_1,results_1);
                    svbfloat16_t results_bf16_1 = svuzp1_bf16(svcvt_bf16_f32_z(pt32, results_fp32_1), svdup_bf16(0.0));
                    svst1_bf16(pt16_w, results_bf16_base + results_single_step_stride, results_bf16_1);

                    svfloat32_t results_fp32_2 = svmul_f32_m(pt32,data_2,results_2);
                    svbfloat16_t results_bf16_2 = svuzp1_bf16(svcvt_bf16_f32_z(pt32, results_fp32_2), svdup_bf16(0.0));
                    svst1_bf16(pt16_w, results_bf16_base + 2 * results_single_step_stride, results_bf16_2);

                    svfloat32_t results_fp32_3 = svmul_f32_m(pt32,data_3,results_3);
                    svbfloat16_t results_bf16_3 = svuzp1_bf16(svcvt_bf16_f32_z(pt32, results_fp32_3), svdup_bf16(0.0));
                    svst1_bf16(pt16_w, results_bf16_base + 3 * results_single_step_stride, results_bf16_3);

                }
            }
        }
    }
    SME_OFF();
}

template <typename T>
void linear_nobias(bfloat16_t* xptr, bfloat16_t* wptr, T* resultptr, int bucket, int num_channels, int hidden_size, bool x_packed)
{
    const int row_block_size = ROW_BLOCK_SIZE;
    const int col_block_size = COL_BLOCK_SIZE;
    bfloat16_t* data_ptr = xptr;
    alignas(64) bfloat16_t pack_A[bucket * hidden_size];
    if(x_packed == false){
        std::fill(pack_A, pack_A + bucket * hidden_size, static_cast<bfloat16_t>(0.0f));
        pack_left_task_2vl_2vl_bf16(xptr, pack_A, bucket, hidden_size);
        data_ptr = pack_A;
    }

    int max_threads = kutacc::get_thread_num();
    int task_num = (bucket * num_channels) / (row_block_size * col_block_size);
    int avg_task = task_num / max_threads;
    int busy_thread = task_num - avg_task * max_threads;
    int row_task_num = num_channels / col_block_size; 
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
        for(int task_idx = start_idx; task_idx < start_idx + task_num; task_idx++ )
        {
            int row_index = (task_idx / row_task_num) * row_block_size;
            int col_index = (task_idx % row_task_num) * col_block_size;
            bfloat16_t *block_part_pack_A_base = data_ptr + row_index * hidden_size ;
            bfloat16_t *block_part_pack_B_base = wptr + col_index * hidden_size;
            T *block_part_C_base = resultptr + row_index * num_channels + col_index;
            block_bgemm_4vl_1vl_impl(block_part_pack_A_base, block_part_pack_B_base, block_part_C_base, row_block_size, col_block_size, hidden_size, num_channels);
        }
    });
    // free(pack_A);
}
void linear_bias_sigmoid(bfloat16_t* single_cond_ptr, bfloat16_t* wptr, float* bptr, float* resultptr, bfloat16_t* resultptr_bf16, int samples, int bucket, int num_channels, int hidden_size)
{
    const int row_block_size = ROW_BLOCK_SIZE;
    const int col_block_size = COL_BLOCK_SIZE;
    bfloat16_t* pack_A = static_cast<bfloat16_t*>(aligned_alloc(64, sizeof(bfloat16_t) *  static_cast<unsigned long>(bucket * hidden_size)));
    std::fill(pack_A, pack_A + bucket * hidden_size, static_cast<bfloat16_t>(0.0f));
    pack_left_task_2vl_2vl_bf16(single_cond_ptr, pack_A, bucket, hidden_size);
    int max_threads = kutacc::get_thread_num();
    int task_num = (bucket * num_channels) / (row_block_size * col_block_size);
    int avg_task = task_num / max_threads;
    int busy_thread = task_num - avg_task * max_threads;
    int row_task_num = num_channels / col_block_size; 
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
        for(int task_idx = start_idx; task_idx < start_idx + task_num; task_idx++ )
        {
            int row_index = (task_idx / row_task_num) * row_block_size;
            int col_index = (task_idx % row_task_num) * col_block_size;
            bfloat16_t *block_part_pack_A_base = pack_A + row_index * hidden_size ;
            float *part_bias_base = bptr + col_index ;
            bfloat16_t *block_part_pack_B_base = wptr + col_index * hidden_size;
            float *block_part_C_base = resultptr + row_index * num_channels + col_index;
            bfloat16_t *block_part_result_bf16_base = resultptr_bf16 + row_index * num_channels + col_index;
            block_linear_bias_sigmoid_4vl_1vl_impl(block_part_pack_A_base, block_part_pack_B_base, part_bias_base, block_part_C_base, block_part_result_bf16_base, \
            row_block_size, col_block_size, hidden_size, samples, bucket, num_channels);
        }
    });
    free(pack_A);
}

template void block_bgemm_4vl_1vl_impl<bfloat16_t>(bfloat16_t *, bfloat16_t *, bfloat16_t *,int, int, int, int);
template void block_bgemm_4vl_1vl_impl<float>(bfloat16_t *, bfloat16_t *, float *,int, int, int, int);

template void linear_nobias<bfloat16_t>(bfloat16_t* , bfloat16_t* , bfloat16_t* , int, int, int, bool);
template void linear_nobias<float>(bfloat16_t* , bfloat16_t* , float* , int, int, int, bool);

// } //namespace alphafold3
} //namespace kutaccs