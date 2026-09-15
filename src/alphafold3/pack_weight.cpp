#include "kutacc.h"
#include "common.h"
namespace kutacc{

void bgemm_pack_weight_bf16(bfloat16_t *weight,bfloat16_t *packed_weight,int M, int N)
{
    const int vscalew = svcntw();
    const int vscaleh = svcnth();
    svbool_t pt16 = svptrue_b16();
    int max_threads = kutacc::get_thread_num();
    const int row_step = 2;
    const int col_step = vscaleh;
    const int one_task = row_step * col_step;
    int all_task = (M * N) / one_task;
    int avg_task =  all_task / max_threads;
    int busy_thread = all_task % max_threads;
    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
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
            svbfloat16_t unz1,unz2,z1,z2;
            unz1 = svld1(pt16, weight + row_index * N + col_index);  
            unz2 = svld1(pt16, weight + (row_index + 1) * N + col_index); 
            z1 = svzip1_bf16(unz1, unz2); 
            z2 = svzip2_bf16(unz1, unz2); 
            svst1_bf16(pt16,packed_weight + col_index * M + row_index * vscalew, z1); 
            svst1_bf16(pt16,packed_weight + (col_index + vscalew)* M + row_index*vscalew, z2);
        }
    });
}

void glu_pack_weight_bf16(float *wptr,bfloat16_t *packed_weight,int num_channels, int num_intermediate)
{
    const int vscalew = svcntw();//16
	svbool_t pt16=svptrue_b16();
	svbool_t pt32=svptrue_b32();
    bfloat16_t *w1 = packed_weight;
    const int aligned_elements = num_channels*num_intermediate;
    bfloat16_t *w2 = packed_weight+aligned_elements;

	int task_num = (num_channels * num_intermediate) / (2 * vscalew);
	int task_num_dim0 = num_intermediate / vscalew;
	kutacc::parallel_for(0, task_num, 1, [&](int64_t start, int64_t end) {
		for(int task_idx = start; task_idx < end; task_idx += 1){
			int i = (task_idx / task_num_dim0) * 2;
			int j = (task_idx % task_num_dim0) * vscalew;

            svfloat32_t unz1 = svld1_f32(pt32,wptr+i*(num_intermediate<<1)+j);
            svfloat32_t unz2 = svld1_f32(pt32,wptr+(i+1)*(num_intermediate<<1)+j);
            svfloat32_t z1=svzip1_f32(unz1,unz2);
            svfloat32_t z2=svzip2_f32(unz1,unz2);
            svbfloat16_t z1_16 = svcvt_bf16_f32_z(pt32, z1);
			svbfloat16_t z2_16 = svcvt_bf16_f32_z(pt32, z2);
            svbfloat16_t z = svuzp1_bf16(z1_16,z2_16);
            svst1_bf16(pt16,w1+j*num_channels+i*vscalew,z);
            
            unz1 = svld1_f32(pt32,wptr+i*(num_intermediate<<1)+num_intermediate+j);
            unz2 = svld1_f32(pt32,wptr+(i+1)*(num_intermediate<<1)+num_intermediate+j);
            z1=svzip1_f32(unz1,unz2);
            z2=svzip2_f32(unz1,unz2);
            z1_16 = svcvt_bf16_f32_z(pt32, z1);
			z2_16 = svcvt_bf16_f32_z(pt32, z2);
            z = svuzp1_bf16(z1_16,z2_16);
            svst1_bf16(pt16,w2+j*num_channels+i*vscalew,z);
        }
    });
}

void glu_pack_weight_fp32(float *wptr,float *packed_weight,int num_channels, int num_intermediate)
{
    const int vscalew = svcntw();//16
	svbool_t pt=svptrue_b32();
    float *w1 = packed_weight;
    float *w2 = packed_weight+num_channels*num_intermediate;
	int task_num = (num_channels * num_intermediate) / vscalew;
	int task_num_dim0 = num_intermediate / vscalew;
	kutacc::parallel_for(0, task_num, 1, [&](int64_t start, int64_t end) {
		for(int task_idx = start; task_idx < end; task_idx += 1){
			int i = (task_idx / task_num_dim0);
			int j = (task_idx % task_num_dim0) * vscalew;
            svfloat32_t z = svld1_f32(pt,wptr+i*(num_intermediate<<1)+j);
            svst1_f32(pt,w1+j*num_channels+i*vscalew,z);
            z = svld1_f32(pt,wptr+i*(num_intermediate<<1)+num_intermediate+j);
            svst1_f32(pt,w2+j*num_channels+i*vscalew,z);
        }
    });
}

void glu_pack_weight_batch_fp32(float *wptr,float *packed_weight,int num_channels, int num_intermediate)
{
    const int vscalew = svcntw();//16
    float *w1 = packed_weight;
    float *w2 = packed_weight+num_channels*num_intermediate+(vscalew<<1);
	int task_num = (num_channels * num_intermediate) / (vscalew<<1);
	int task_num_dim0 = num_intermediate / (vscalew<<1);
	kutacc::parallel_for(0, task_num, 1, [&](int64_t start, int64_t end) {
		for(int task_idx = start; task_idx < end; task_idx += 1){
			int i = (task_idx / task_num_dim0);
			int j = (task_idx % task_num_dim0) * (vscalew<<1);

			svbool_t pt=svptrue_b32();
			const int vscalew = svcntw();
			float* wptr_ = wptr+i*(num_intermediate<<1)+j;
			int bias = j*num_channels+i*(vscalew<<1);

			svfloat32_t z1 = svld1_f32(pt,wptr_);
			svst1_f32(pt,w1+bias,z1);
			z1 = svld1_f32(pt,wptr_+vscalew);
			svst1_f32(pt,w1+bias+vscalew,z1);
			svfloat32_t z2 = svld1_f32(pt,wptr_+num_intermediate);
			svst1_f32(pt,w2+bias,z2);
			z2 = svld1_f32(pt,wptr_+num_intermediate+vscalew);
			svst1_f32(pt,w2+bias+vscalew,z2);
        }
    });
}
template<typename w_t>
void glu_pack_weight(float *wptr, w_t *packed_weight,int num_channels, int num_intermediate, bool hasbatch){
    if(sizeof(w_t) == 2){
        glu_pack_weight_bf16(wptr, (bfloat16_t *)packed_weight, num_channels, num_intermediate);
    }else if(hasbatch){
        glu_pack_weight_batch_fp32(wptr, (float *)packed_weight, num_channels, num_intermediate);
    }else{
        glu_pack_weight_fp32(wptr, (float *)packed_weight, num_channels, num_intermediate);
    }
}
template void glu_pack_weight<bfloat16_t>(float *, bfloat16_t *, int, int, bool);
template void glu_pack_weight<float>(float *, float *, int, int, bool);

} //namespace kutacc
