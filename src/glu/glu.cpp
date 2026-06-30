#include "kutacc.h"
#include "glu.h"
#include "tensor/tensor.h"
#include "math/fast_exp.h"
namespace kutacc{

void GLU_transposed(bfloat16_t *matA,bfloat16_t *matC,int N, int M)
{
	svbool_t pt16=svptrue_b16();
    for (int i=0;i<N;i+=SVL_bf16)
	{ 
        for (int j=0;j<M;j+=SVL_bf16<<1)
		{
			SME_ON();
            for (int t=0;t<SVL_bf16;t++)
			{
                svld1_hor_za16(0, t, pt16, matA + (i + t) * M + j);
                svld1_hor_za16(1, t, pt16, matA + (i + t) * M + j+SVL_bf16);
            }
            for (int t=0;t<SVL_bf16;t+=2)
			{
                svbfloat16_t unz1,unz2,z1,z2;
				unz1 = svread_ver_za16_bf16_m(unz1,pt16,0,t);
	            unz2 = svread_ver_za16_bf16_m(unz2,pt16,0,t+1);
	            z1 = svzip1_bf16(unz1,unz2);
	            z2 = svzip2_bf16(unz1,unz2);
	            svst1_bf16(pt16,matC + i*M + (j+t)*SVL_fp32, z1);
	            svst1_bf16(pt16,matC + (i+SVL_fp32)*M + (j+t)*SVL_fp32, z2);
	            
	            unz1 = svread_ver_za16_bf16_m(unz1,pt16,1,t);
	            unz2 = svread_ver_za16_bf16_m(unz2,pt16,1,t+1);
	            z1 = svzip1_bf16(unz1,unz2);
	            z2 = svzip2_bf16(unz1,unz2);
	            svst1_bf16(pt16,matC + i*M + (j+t+SVL_bf16)*SVL_fp32, z1);
	            svst1_bf16(pt16,matC + (i+SVL_fp32)*M + (j+t+SVL_bf16)*SVL_fp32, z2);
            }
            SME_OFF();
        }
    }
}

void GLU_transposed(float *matA,float *matC,int N, int M)
{
    svbool_t pt=svptrue_b32();
    for (int i=0;i<N;i+=SVL_fp32)
	{
        for (int j=0;j<M;j+=(SVL_fp32<<2))
		{
			SME_ON();
            for (int t=0;t<SVL_fp32;t++)
			{
                svld1_hor_za32(0, t, pt, matA + (i + t) * M + j);
                svld1_hor_za32(1, t, pt, matA + (i + t) * M + j + SVL_fp32);
                svld1_hor_za32(2, t, pt, matA + (i + t) * M + j + (SVL_fp32<<1));
                svld1_hor_za32(3, t, pt, matA + (i + t) * M + j + ((SVL_fp32<<1)+SVL_fp32));
            }
            for (int t=0;t<SVL_fp32;t++)
			{
				svst1_ver_za32(0, t, pt, matC + i*M + (j+t)*SVL_fp32);
                svst1_ver_za32(1, t, pt, matC + i*M + (j+t+SVL_fp32)*SVL_fp32);
                svst1_ver_za32(2, t, pt, matC + i*M + (j+t+(SVL_fp32<<1))*SVL_fp32);
                svst1_ver_za32(3, t, pt, matC + i*M + (j+t+((SVL_fp32<<1)+SVL_fp32))*SVL_fp32);
            }
            SME_OFF();
        }
    }
}

void GLU_transposed_parallel(bfloat16_t *matA,bfloat16_t *matC,int N, int M)
{
	svbool_t pt16=svptrue_b16();
	int task_num = (N * M) / (SVL_bf16 * (SVL_bf16<<1));
	int task_num_dim0 = (M / (SVL_bf16<<1));
	kutacc::parallel_for(0, task_num, 1, [&](int64_t start, int64_t end) {
		for(int task_idx = start; task_idx < end; task_idx += 1){
			int i = (task_idx / task_num_dim0) * SVL_bf16;
			int j = (task_idx % task_num_dim0) * (SVL_bf16<<1);
			SME_ON();
            for (int t=0;t<SVL_bf16;t++)
			{
                svld1_hor_za16(0, t, pt16, matA + (i + t) * M + j);
                svld1_hor_za16(1, t, pt16, matA + (i + t) * M + j+SVL_bf16);
            }
            for (int t=0;t<SVL_bf16;t+=2)
			{
                svbfloat16_t unz1,unz2,z1,z2;
				unz1 = svread_ver_za16_bf16_m(unz1,pt16,0,t);
	            unz2 = svread_ver_za16_bf16_m(unz2,pt16,0,t+1);
	            z1 = svzip1_bf16(unz1,unz2);
	            z2 = svzip2_bf16(unz1,unz2);
	            svst1_bf16(pt16,matC + i*M + (j+t)*SVL_fp32, z1);
	            svst1_bf16(pt16,matC + (i+SVL_fp32)*M + (j+t)*SVL_fp32, z2);
	            
	            unz1 = svread_ver_za16_bf16_m(unz1,pt16,1,t);
	            unz2 = svread_ver_za16_bf16_m(unz2,pt16,1,t+1);
	            z1 = svzip1_bf16(unz1,unz2);
	            z2 = svzip2_bf16(unz1,unz2);
	            svst1_bf16(pt16,matC + i*M + (j+t+SVL_bf16)*SVL_fp32, z1);
	            svst1_bf16(pt16,matC + (i+SVL_fp32)*M + (j+t+SVL_bf16)*SVL_fp32, z2);
            }
            SME_OFF();
        }
    });
}

void GLU_transposed_parallel(float *matA,float *matC,int N, int M)
{
    svbool_t pt=svptrue_b32();

	int task_num = (N * M) / (SVL_fp32 * (SVL_fp32<<2));
	int task_num_dim0 = (M / (SVL_fp32<<2));
	kutacc::parallel_for(0, task_num, 1, [&](int64_t start, int64_t end) {
		for(int task_idx = start; task_idx < end; task_idx += 1){
			int i = (task_idx / task_num_dim0) * SVL_fp32;
			int j = (task_idx % task_num_dim0) * (SVL_fp32<<2);
			SME_ON();
            for (int t=0;t<SVL_fp32;t++)
			{
                svld1_hor_za32(0, t, pt, matA + (i + t) * M + j);
                svld1_hor_za32(1, t, pt, matA + (i + t) * M + j + SVL_fp32);
                svld1_hor_za32(2, t, pt, matA + (i + t) * M + j + (SVL_fp32<<1));
                svld1_hor_za32(3, t, pt, matA + (i + t) * M + j + ((SVL_fp32<<1)+SVL_fp32));
            }
            for (int t=0;t<SVL_fp32;t++)
			{
				svst1_ver_za32(0, t, pt, matC + i*M + (j+t)*SVL_fp32);
                svst1_ver_za32(1, t, pt, matC + i*M + (j+t+SVL_fp32)*SVL_fp32);
                svst1_ver_za32(2, t, pt, matC + i*M + (j+t+(SVL_fp32<<1))*SVL_fp32);
                svst1_ver_za32(3, t, pt, matC + i*M + (j+t+((SVL_fp32<<1)+SVL_fp32))*SVL_fp32);
            }
            SME_OFF();
        }
    });
}

void glu_sigmoid(bfloat16_t* xptr,float* wptr,bfloat16_t* maskptr,bfloat16_t* resultptr, int Bn, int Nb, int num_channels)
{
	int num_intermediate=num_channels<<1;
	bfloat16_t* w1=static_cast<bfloat16_t*>(aligned_alloc(64, sizeof(bfloat16_t) * num_channels*num_intermediate));
	bfloat16_t* w2=static_cast<bfloat16_t*>(aligned_alloc(64, sizeof(bfloat16_t) * num_channels*num_intermediate));
	svbool_t pt16=svptrue_b16();
	svbool_t pt32=svptrue_b32();
    int max_threads = kutacc::get_thread_num();
	int task_num = (num_channels * num_intermediate) / (2 * SVL_fp32);
	int task_num_dim0 = num_intermediate / SVL_fp32;
	kutacc::parallel_for(0, task_num, 1, [&](int64_t start, int64_t end) {
		for(int task_idx = start; task_idx < end; task_idx += 1){
			int i = (task_idx / task_num_dim0) * 2;
			int j = (task_idx % task_num_dim0) * SVL_fp32;

            svfloat32_t unz1 = svld1_f32(pt32,wptr+i*(num_intermediate<<1)+j);
            svfloat32_t unz2 = svld1_f32(pt32,wptr+(i+1)*(num_intermediate<<1)+j);
            svfloat32_t z1=svzip1_f32(unz1,unz2);
            svfloat32_t z2=svzip2_f32(unz1,unz2);
            svbfloat16_t z1_16 = svcvt_bf16_f32_z(pt32, z1);
			svbfloat16_t z2_16 = svcvt_bf16_f32_z(pt32, z2);
            svbfloat16_t z = svuzp1_bf16(z1_16,z2_16);
            svst1_bf16(pt16,w1+j*num_channels+i*SVL_fp32,z);
            
            unz1 = svld1_f32(pt32,wptr+i*(num_intermediate<<1)+num_intermediate+j);
            unz2 = svld1_f32(pt32,wptr+(i+1)*(num_intermediate<<1)+num_intermediate+j);
            z1=svzip1_f32(unz1,unz2);
            z2=svzip2_f32(unz1,unz2);
            z1_16 = svcvt_bf16_f32_z(pt32, z1);
			z2_16 = svcvt_bf16_f32_z(pt32, z2);
            z = svuzp1_bf16(z1_16,z2_16);
            svst1_bf16(pt16,w2+j*num_channels+i*SVL_fp32,z);
        }
    });
    kutacc::parallel_for(0, Bn, 1, [&](int64_t start, int64_t end)
	{
		for (int i=start;i<end;i++)
		{
			bfloat16_t* x_trans=static_cast<bfloat16_t*>(aligned_alloc(64, sizeof(bfloat16_t) * num_channels*Nb));
			GLU_transposed(xptr + i*Nb*num_channels,x_trans,Nb,num_channels);
			SME_ON();
			for (int j=0;j<Nb;j+=SVL_fp32*2)
			{
				svbfloat16_t maskrow=svld1_bf16(pt16,maskptr+i*Nb+j);
				svbfloat16_t zerorow=svdup_bf16(0.0f);
				svfloat32_t maskrow_32_1=svreinterpret_f32(svzip1_bf16(zerorow,maskrow));
				svfloat32_t maskrow_32_2=svreinterpret_f32(svzip2_bf16(zerorow,maskrow));
				for (int k=0;k<num_intermediate;k+=SVL_fp32)
				{
					svzero_za();
					for (int l=0;l<num_channels;l+=2)
					{
						svbfloat16_t colX1 = svld1_bf16(pt16,x_trans+j*num_channels+l*SVL_fp32);
						svbfloat16_t colX2 = svld1_bf16(pt16,x_trans+(j+SVL_fp32)*num_channels+l*SVL_fp32);
						svbfloat16_t rowW1 = svld1_bf16(pt16,w1+k*num_channels+l*SVL_fp32);
						svbfloat16_t rowW2 = svld1_bf16(pt16,w2+k*num_channels+l*SVL_fp32);
						
						svmopa_za32_bf16_m(0,pt16,pt16,colX1,rowW1);
						svmopa_za32_bf16_m(1,pt16,pt16,colX1,rowW2);
						svmopa_za32_bf16_m(2,pt16,pt16,colX2,rowW1);
						svmopa_za32_bf16_m(3,pt16,pt16,colX2,rowW2);
					}
					for (int t=0;t<SVL_fp32;t++)
					{
						svfloat32_t a_data_1 = svread_ver_za32_f32_m(a_data_1,pt32,0,t);
						svfloat32_t b_data_1 = svread_ver_za32_f32_m(b_data_1,pt32,1,t);
						svfloat32_t a_data_2 = svread_ver_za32_f32_m(a_data_2,pt32,2,t);
						svfloat32_t b_data_2 = svread_ver_za32_f32_m(b_data_2,pt32,3,t);
						b_data_1 = svmul_f32_z(pt32,b_data_1,maskrow_32_1);           
						a_data_1 = svneg_f32_z(pt32,a_data_1);
						a_data_1 = kutacc::fast_exp(pt32, a_data_1);

						a_data_1 = svadd_n_f32_z(pt32,a_data_1,1.0f);
						a_data_1 = svdiv_f32_z(pt32,b_data_1,a_data_1);
						
						b_data_2 = svmul_f32_z(pt32,b_data_2,maskrow_32_2);  
						a_data_2 = svneg_f32_z(pt32,a_data_2);
						a_data_2 = kutacc::fast_exp(pt32, a_data_2);

						a_data_2 = svadd_n_f32_z(pt32,a_data_2,1.0f);
						a_data_2 = svdiv_f32_z(pt32,b_data_2,a_data_2);
						
						svbfloat16_t a_data_bf16_1 = svcvt_bf16_f32_z(pt32, a_data_1);
						svbfloat16_t a_data_bf16_2 = svcvt_bf16_f32_z(pt32, a_data_2);
						svbfloat16_t a_data_bf16 = svuzp1_bf16(a_data_bf16_1,a_data_bf16_2);
						svst1_bf16(pt16,resultptr+(k+t)*Bn*Nb+i*Nb+j,a_data_bf16);
					}
				}
			}
			SME_OFF();
			free(x_trans);
		}
	});
	free(w1);
	free(w2);
}

void glu_pack_weight_bf16(bfloat16_t* weight, bfloat16_t* packed_weight, int M, int N)
{
    int max_threads = kutacc::get_thread_num();
    int avg_task = N / SVL_fp32 / max_threads;
    int busy_thread = N / SVL_fp32-avg_task*max_threads;
    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
        int tid = kutacc::get_thread_id();
        int task_num, start_idx, end_idx;
        if(tid < busy_thread)
        {
            task_num = avg_task+1;
            start_idx = tid*task_num*SVL_fp32;
            end_idx = start_idx+task_num*SVL_fp32;
        }
        else
        {
            task_num = avg_task;
            start_idx = busy_thread*(task_num+1)*SVL_fp32+(tid-busy_thread)*task_num*SVL_fp32;
            end_idx = start_idx+task_num*SVL_fp32;
        }
		svbool_t p16all = svptrue_b16();
        for(int nidx = start_idx; nidx < end_idx; nidx += SVL_fp32)
        {

            bfloat16_t* w_nidx = weight+nidx;
            bfloat16_t* p_w_nidx = packed_weight+M*nidx;
            #pragma unroll(8)
            for(int m = 0; m < M; m += 2)
            {
                svbfloat16_t w_d1 = svld1_bf16(p16all, w_nidx + m*N);
				svbfloat16_t w_d2 = svld1_bf16(p16all, w_nidx + (m+1)*N);
				w_d1 = svzip1_bf16(w_d1, w_d2);
				svst1_bf16(p16all, p_w_nidx+m*SVL_fp32, w_d1);
            }
        }
    
    });

}

void glu_pack_left_bf16(bfloat16_t* matA, bfloat16_t* packed_matA, int M, int N)
{
    int max_threads = kutacc::get_thread_num();
    int avg_task = M / SVL_fp32 / max_threads;
    int busy_thread = M / SVL_fp32-avg_task*max_threads;
    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
		SME_ON(); 
        int tid = kutacc::get_thread_id();
        int task_num, start_idx, end_idx;
        if(tid < busy_thread)
        {
            task_num = avg_task+1;
            start_idx = tid*task_num*SVL_fp32;
            end_idx = start_idx+task_num*SVL_fp32;
        }
        else
        {
            task_num = avg_task;
            start_idx = busy_thread*(task_num+1)*SVL_fp32+(tid-busy_thread)*task_num*SVL_fp32;
            end_idx = start_idx+task_num*SVL_fp32;
        }
        svbool_t p32all = svptrue_b32();
        for(int midx = start_idx; midx < end_idx; midx += SVL_fp32)
        {
            bfloat16_t* m_midx = matA+midx*N;
            bfloat16_t* p_m_midx = packed_matA+midx*N;
			if(N>=128)
			{
				for(int nidx = 0; nidx < N; nidx += SVL_bf16*4)
				{
					bfloat16_t* m0 = m_midx;
					bfloat16_t* m1 = m0+SVL_bf16;
					bfloat16_t* m2 = m1+SVL_bf16;
					bfloat16_t* m3 = m2+SVL_bf16;
					for(int trow = 0; trow < SVL_fp32; ++trow)
					{
						svld1_hor_za32(0, trow, p32all, (float*)(m0+trow*N));
						svld1_hor_za32(1, trow, p32all, (float*)(m1+trow*N));
						svld1_hor_za32(2, trow, p32all, (float*)(m2+trow*N));
						svld1_hor_za32(3, trow, p32all, (float*)(m3+trow*N));
					}
					bfloat16_t* pm0 = p_m_midx;
					bfloat16_t* pm1 = pm0+2*SVL_fp32*SVL_fp32;
					bfloat16_t* pm2 = pm1+2*SVL_fp32*SVL_fp32;
					bfloat16_t* pm3 = pm2+2*SVL_fp32*SVL_fp32;
					for(int tcol = 0; tcol < SVL_fp32; ++tcol)
					{
						svst1_ver_za32(0, tcol, p32all, (float*)(pm0+tcol*SVL_bf16)); 
						svst1_ver_za32(1, tcol, p32all, (float*)(pm1+tcol*SVL_bf16)); 
						svst1_ver_za32(2, tcol, p32all, (float*)(pm2+tcol*SVL_bf16)); 
						svst1_ver_za32(3, tcol, p32all, (float*)(pm3+tcol*SVL_bf16)); 					
					}
					m_midx += 4*SVL_bf16;
					p_m_midx += 4*2*SVL_fp32*SVL_fp32;			
				}
			}
			else
			{
				for(int nidx = 0; nidx < N; nidx += SVL_bf16*2)
				{
					bfloat16_t* m0 = m_midx;
					bfloat16_t* m1 = m0+SVL_bf16;
					for(int trow = 0; trow < SVL_fp32; ++trow)
					{
						svld1_hor_za32(0, trow, p32all, (float*)(m0+trow*N));
						svld1_hor_za32(1, trow, p32all, (float*)(m1+trow*N));
					}
					bfloat16_t* pm0 = p_m_midx;
					bfloat16_t* pm1 = pm0+2*SVL_fp32*SVL_fp32;
					for(int tcol = 0; tcol < SVL_fp32; ++tcol)
					{
						svst1_ver_za32(0, tcol, p32all, (float*)(pm0+tcol*SVL_bf16)); 
						svst1_ver_za32(1, tcol, p32all, (float*)(pm1+tcol*SVL_bf16)); 				
					}
					m_midx += 2*SVL_bf16;
					p_m_midx += 2*2*SVL_fp32*SVL_fp32;			
				}				
			}
        }
        SME_OFF();    
    });

}

//the N of the right_matrix is 256, so we only split the left_matrix
void glu_mask_permute(bfloat16_t* packed_x, bfloat16_t* packed_weight, bfloat16_t* o, float* maskptr, int Bn, int Nb, int C, int C1)
{
	int M = Bn*Nb;
    int max_threads = kutacc::get_thread_num();
    int avg_task = M / SVL_bf16 / max_threads;
    int busy_thread = M / SVL_bf16-avg_task*max_threads;
	bfloat16_t* pw1 = packed_weight;
	bfloat16_t* pw2 = packed_weight+C*C1;
    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
    {
        int tid = kutacc::get_thread_id();
        int task_num, start_idx, end_idx;
        if(tid < busy_thread)
        {
            task_num = avg_task+1;
            start_idx = tid*task_num*SVL_bf16;
            end_idx = start_idx+task_num*SVL_bf16;
        }
        else
        {
            task_num = avg_task;
            start_idx = busy_thread*(task_num+1)*SVL_bf16+(tid-busy_thread)*task_num*SVL_bf16;
            end_idx = start_idx+task_num*SVL_bf16;
        }
    	SME_ON();
		svbool_t p32all = svptrue_b32();
		svbool_t p16all = svptrue_b16();
		svfloat32_t zero_vec_f32 = svdup_f32(0.0f);
		for(int nidx = start_idx; nidx < end_idx; nidx += SVL_bf16)
		{
			bfloat16_t* x_nidx = packed_x+nidx*C;
			bfloat16_t* x0 = x_nidx;
			bfloat16_t* x1 = x0+SVL_fp32*C;
			float* m0 = maskptr+nidx;
			float* m1 = m0+SVL_fp32;
			svfloat32_t m_d0 = svld1_f32(p32all, m0);
			svfloat32_t m_d1 = svld1_f32(p32all, m1);
			for(int c1idx = 0; c1idx < C1; c1idx += SVL_fp32)
			{
				bfloat16_t* w1 = pw1+c1idx*C;
				bfloat16_t* w2 = pw2+c1idx*C;
				svzero_za();
				for(int cidx = 0; cidx < C; cidx += 2)
				{
					svbfloat16_t x_d0 = svld1_bf16(p16all, x0+cidx*SVL_fp32);
					svbfloat16_t x_d1 = svld1_bf16(p16all, x1+cidx*SVL_fp32);
					svbfloat16_t w1_d = svld1_bf16(p16all, w1+cidx*SVL_fp32);
					svbfloat16_t w2_d = svld1_bf16(p16all, w2+cidx*SVL_fp32);

					svmopa_za32_bf16_m(0, p16all, p16all, x_d0, w1_d);
					svmopa_za32_bf16_m(1, p16all, p16all, x_d1, w1_d);
					svmopa_za32_bf16_m(2, p16all, p16all, x_d0, w2_d);
					svmopa_za32_bf16_m(3, p16all, p16all, x_d1, w2_d);
				}
			
				for (int tcol = 0; tcol<SVL_fp32; tcol++)
				{
					svfloat32_t lf0 = svread_ver_za32_f32_m(zero_vec_f32, p32all, 0, tcol);
					svfloat32_t lf1 = svread_ver_za32_f32_m(zero_vec_f32, p32all, 1, tcol);
					svfloat32_t rf0 = svread_ver_za32_f32_m(zero_vec_f32, p32all, 2, tcol);
					svfloat32_t rf1 = svread_ver_za32_f32_m(zero_vec_f32, p32all, 3, tcol);

	                lf0 = svneg_f32_z(p32all,lf0);
					lf1 = svneg_f32_z(p32all,lf1);
					rf0 = svmul_f32_z(p32all, rf0, m_d0);
					rf1 = svmul_f32_z(p32all, rf1, m_d1);
					lf0 = kutacc::fast_exp(p32all, lf0);
					lf1 = kutacc::fast_exp(p32all, lf1);
					lf0 = svadd_n_f32_z(p32all, lf0, 1.0f);
					lf1 = svadd_n_f32_z(p32all, lf1, 1.0f);

					lf0 = svdiv_f32_z(p32all, rf0, lf0);
					lf1 = svdiv_f32_z(p32all, rf1, lf1);

					svbfloat16_t lb0 = svcvt_bf16_f32_z(p32all, lf0);
					svbfloat16_t lb1 = svcvt_bf16_f32_z(p32all, lf1);
					lb0 = svuzp1_bf16(lb0,lb1);

					svst1_bf16(p16all, o+(c1idx+tcol)*Bn*Nb+nidx,lb0);
				}
			}
		}
    	SME_OFF();
	});
}
void glu_swish_c2(bfloat16_t* xptr, bfloat16_t* resultptr,int bucket,int num_channels,int num_intermediate, bfloat16_t* w1, bfloat16_t* w2)
{
	svbool_t pt16=svptrue_b16();
	svbool_t pt32=svptrue_b32();
    
	bfloat16_t* x_trans=static_cast<bfloat16_t*>(aligned_alloc(64, sizeof(bfloat16_t) * num_channels*bucket));
	GLU_transposed_parallel(xptr,x_trans,bucket,num_channels);
	int task_num = (bucket * num_intermediate) / (SVL_fp32 * (SVL_fp32*2));
	int task_num_dim0 = (num_intermediate / (SVL_fp32*2));
	kutacc::parallel_for(0, task_num, 1, [&](int64_t start, int64_t end) {
		for(int task_idx = start; task_idx < end; task_idx += 1){
			int j = (task_idx / task_num_dim0) * SVL_fp32;
			int k = (task_idx % task_num_dim0) * (SVL_fp32*2);

			SME_ON();
			svzero_za();
			for (int l=0;l<num_channels;l+=2)
			{
				svbfloat16_t colX = svld1_bf16(pt16,x_trans+j*num_channels+l*SVL_fp32);
            	svbfloat16_t rowW1_1 = svld1_bf16(pt16,w1+k*num_channels+l*SVL_fp32);
            	svbfloat16_t rowW2_1 = svld1_bf16(pt16,w2+k*num_channels+l*SVL_fp32);
            	
            	svbfloat16_t rowW1_2 = svld1_bf16(pt16,w1+(k+SVL_fp32)*num_channels+l*SVL_fp32);
            	svbfloat16_t rowW2_2 = svld1_bf16(pt16,w2+(k+SVL_fp32)*num_channels+l*SVL_fp32);
            	svmopa_za32_bf16_m(0,pt16,pt16,colX,rowW1_1);
            	svmopa_za32_bf16_m(1,pt16,pt16,colX,rowW2_1);
            	svmopa_za32_bf16_m(2,pt16,pt16,colX,rowW1_2);
            	svmopa_za32_bf16_m(3,pt16,pt16,colX,rowW2_2);
			}
			for (int t=0;t<SVL_fp32;t++)
			{
                svfloat32_t a_data_1 = svread_hor_za32_f32_m(a_data_1,pt32,0,t);
                svfloat32_t b_data_1 = svread_hor_za32_f32_m(b_data_1,pt32,1,t);
                svfloat32_t a_data_2 = svread_hor_za32_f32_m(a_data_2,pt32,2,t);
                svfloat32_t b_data_2 = svread_hor_za32_f32_m(b_data_2,pt32,3,t);
                
                b_data_1 = svmul_f32_z(pt32,b_data_1,a_data_1);
                a_data_1 = svneg_f32_z(pt32,a_data_1);
                a_data_1 = kutacc::fast_exp(pt32, a_data_1); 

                a_data_1 = svadd_n_f32_z(pt32,a_data_1,1.0f);
                a_data_1 = svdiv_f32_z(pt32,b_data_1,a_data_1);
                
                b_data_2 = svmul_f32_z(pt32,b_data_2,a_data_2);
                a_data_2 = svneg_f32_z(pt32,a_data_2);
                a_data_2 = kutacc::fast_exp(pt32, a_data_2);

                a_data_2 = svadd_n_f32_z(pt32,a_data_2,1.0f);
                a_data_2 = svdiv_f32_z(pt32,b_data_2,a_data_2);
                
				svbfloat16_t a_data_bf16_1 = svcvt_bf16_f32_z(pt32, a_data_1);
				svbfloat16_t a_data_bf16_2 = svcvt_bf16_f32_z(pt32, a_data_2);
				svbfloat16_t a_data_bf16 = svuzp1_bf16(a_data_bf16_1,a_data_bf16_2);
				svst1_bf16(pt16,resultptr + (j+t)*num_intermediate + k,a_data_bf16);
            }
            SME_OFF();
		}
	});
	
	free(x_trans);
}


void glu_swish_cbatch_bf16(float* xptr,bfloat16_t* resultptr, int batch ,int bucket,int num_channels,int num_intermediate, float* w1, float* w2)
{
	svbool_t pt=svptrue_b32();
	svbool_t pt16=svptrue_b16();
    float* x_trans=static_cast<float*>(aligned_alloc(64, sizeof(float) * (batch*num_channels*bucket+SVL_fp32)));
	GLU_transposed_parallel(xptr,x_trans,batch*bucket,num_channels);

    int max_threads = kutacc::get_thread_num();
	int coll_tasks = num_intermediate / SVL_fp32 / 4;
	int colls_perthread = coll_tasks / max_threads;
	int rem_colls = coll_tasks % max_threads;
    
    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
	{
		int thread_id = kutacc::get_thread_id();
		int jlow = thread_id * colls_perthread * SVL_fp32 * 4 + (thread_id < rem_colls ? thread_id * SVL_fp32* 4 : rem_colls * SVL_fp32* 4);
		int jup = jlow + colls_perthread * SVL_fp32 * 4 + (thread_id < rem_colls ? SVL_fp32* 4 : 0);
		SME_ON();
		for (int i=0;i<batch;i++) {
			for (int j=0;j<bucket;j+=SVL_fp32)
			{
				for (int k=jlow;k<jup;k+=SVL_fp32*2)
				{
					svzero_za();
					for (int l=0;l<num_channels;l++)
					{
						svfloat32_t colX = svld1_f32(pt,x_trans+i*bucket*num_channels+j*num_channels+l*SVL_fp32);
						svfloat32_t rowW1_1 = svld1_f32(pt,w1+k*num_channels+l*SVL_fp32*2);
						svmopa_za32_f32_m(0,pt,pt,colX,rowW1_1);
						svfloat32_t rowW1_2 = svld1_f32(pt,w1+k*num_channels+l*SVL_fp32*2+SVL_fp32);
						svmopa_za32_f32_m(2,pt,pt,colX,rowW1_2);
						svfloat32_t rowW2_1 = svld1_f32(pt,w2+k*num_channels+l*SVL_fp32*2);
						svmopa_za32_f32_m(1,pt,pt,colX,rowW2_1);
						svfloat32_t rowW2_2 = svld1_f32(pt,w2+k*num_channels+l*SVL_fp32*2+SVL_fp32);
						svmopa_za32_f32_m(3,pt,pt,colX,rowW2_2);
					}
					for (int t=0;t<SVL_fp32;t++)
					{
						svfloat32_t a_data_1 = svread_hor_za32_f32_m(a_data_1,pt,0,t);
						svfloat32_t b_data_1 = svread_hor_za32_f32_m(b_data_1,pt,1,t);
						svfloat32_t a_data_2 = svread_hor_za32_f32_m(a_data_2,pt,2,t);
						svfloat32_t b_data_2 = svread_hor_za32_f32_m(b_data_2,pt,3,t);
						
						b_data_1 = svmul_f32_z(pt,b_data_1,a_data_1);
						a_data_1 = svneg_f32_z(pt,a_data_1);
						a_data_1 = kutacc::fast_exp(pt, a_data_1);
						a_data_1 = svadd_n_f32_z(pt,a_data_1,1.0f);
						a_data_1 = svdiv_f32_z(pt,b_data_1,a_data_1);
						
						b_data_2 = svmul_f32_z(pt,b_data_2,a_data_2);
						a_data_2 = svneg_f32_z(pt,a_data_2);
						a_data_2 = kutacc::fast_exp(pt, a_data_2);
						a_data_2 = svadd_n_f32_z(pt,a_data_2,1.0f);
						a_data_2 = svdiv_f32_z(pt,b_data_2,a_data_2);
						
						svbfloat16_t bf_a_data1 = svcvt_bf16_f32_x(pt, a_data_1);
						svbfloat16_t bf_a_data2 = svcvt_bf16_f32_x(pt, a_data_2);
						svbfloat16_t a_data_bf16 = svuzp1_bf16(bf_a_data1,bf_a_data2);
						svst1_bf16(pt16,resultptr + i*bucket*num_intermediate + (j+t)*num_intermediate + k,a_data_bf16);
					}
				}
			}

		}
		SME_OFF();
	});
	free(x_trans);
}

void glu_swish_c34_fp32_bf16(float* xptr,bfloat16_t* resultptr,int shape0,int bucket,int num_channels,int num_intermediate, float* w1, float* w2)
{
	svbool_t pt=svptrue_b32();
    kutacc::parallel_for(0, shape0, 1, [&](int64_t start, int64_t end)
	{
		for (int i=start;i<end;i++)
		{
			svbool_t pt16=svptrue_b16();
			float* x_trans=static_cast<float*>(aligned_alloc(64, sizeof(float) * num_channels*bucket));
			GLU_transposed(xptr + i*bucket*num_channels,x_trans,bucket,num_channels);
			SME_ON();
			for (int j=0;j<bucket;j+=SVL_fp32)
			{
				for (int k=0;k<num_intermediate;k+=SVL_fp32*2)
				{
					svzero_za();
					for (int l=0;l<num_channels;l++)
					{
						svfloat32_t colX = svld1_f32(pt,x_trans+j*num_channels+l*SVL_fp32);
						svfloat32_t rowW1_1 = svld1_f32(pt,w1+k*num_channels+l*SVL_fp32);
						svfloat32_t rowW2_1 = svld1_f32(pt,w2+k*num_channels+l*SVL_fp32);
						
						svfloat32_t rowW1_2 = svld1_f32(pt,w1+(k+SVL_fp32)*num_channels+l*SVL_fp32);
						svfloat32_t rowW2_2 = svld1_f32(pt,w2+(k+SVL_fp32)*num_channels+l*SVL_fp32);
						svmopa_za32_f32_m(0,pt,pt,colX,rowW1_1);
						svmopa_za32_f32_m(1,pt,pt,colX,rowW2_1);
						svmopa_za32_f32_m(2,pt,pt,colX,rowW1_2);
						svmopa_za32_f32_m(3,pt,pt,colX,rowW2_2);
					}
					for (int t=0;t<SVL_fp32;t++)
					{
						svfloat32_t a_data_1 = svread_hor_za32_f32_m(a_data_1,pt,0,t);
						svfloat32_t b_data_1 = svread_hor_za32_f32_m(b_data_1,pt,1,t);
						svfloat32_t a_data_2 = svread_hor_za32_f32_m(a_data_2,pt,2,t);
						svfloat32_t b_data_2 = svread_hor_za32_f32_m(b_data_2,pt,3,t);
						
						b_data_1 = svmul_f32_z(pt,b_data_1,a_data_1);
						a_data_1 = svneg_f32_z(pt,a_data_1);
						a_data_1 = kutacc::fast_exp(pt, a_data_1);
						a_data_1 = svadd_n_f32_z(pt,a_data_1,1.0f);
						a_data_1 = svdiv_f32_z(pt,b_data_1,a_data_1);
						
						b_data_2 = svmul_f32_z(pt,b_data_2,a_data_2);
						a_data_2 = svneg_f32_z(pt,a_data_2);
						a_data_2 = kutacc::fast_exp(pt, a_data_2);
						a_data_2 = svadd_n_f32_z(pt,a_data_2,1.0f);
						a_data_2 = svdiv_f32_z(pt,b_data_2,a_data_2);
						
						svbfloat16_t bf_a_data1 = svcvt_bf16_f32_x(pt, a_data_1);
						svbfloat16_t bf_a_data2 = svcvt_bf16_f32_x(pt, a_data_2);
						svbfloat16_t a_data_bf16 = svuzp1_bf16(bf_a_data1,bf_a_data2);
						svst1_bf16(pt16,resultptr + i*bucket*num_intermediate + (j+t)*num_intermediate + k,a_data_bf16);
					}
				}
			}
			SME_OFF();
			free(x_trans);
		}
	});
}

void glu_swish_cbatch_fp32(float* xptr,float* resultptr, int batch ,int bucket,int num_channels,int num_intermediate, float* w1, float* w2)
{
	svbool_t pt=svptrue_b32();
    float* x_trans=static_cast<float*>(aligned_alloc(64, sizeof(float) * (batch*num_channels*bucket+SVL_fp32)));
	GLU_transposed_parallel(xptr,x_trans,batch*bucket,num_channels);

    int max_threads = kutacc::get_thread_num();
	int coll_tasks = num_intermediate / SVL_fp32 / 4;
	int colls_perthread = coll_tasks / max_threads;
	int rem_colls = coll_tasks % max_threads;
    
    kutacc::parallel_for(0, max_threads, 1, [&](int64_t start, int64_t end)
	{
		int thread_id = kutacc::get_thread_id();
		int jlow = thread_id * colls_perthread + (thread_id < rem_colls ? thread_id * SVL_fp32* 4 : rem_colls * SVL_fp32* 4);
		int jup = jlow + colls_perthread + (thread_id < rem_colls ? SVL_fp32* 4 : 0);
		svbool_t pt=svptrue_b32();
		SME_ON();
		for (int i=0;i<batch;i++) {
			for (int j=0;j<bucket;j+=SVL_fp32)
				{
					for (int k=jlow;k<jup;k+=SVL_fp32*2)
					{
						svzero_za();
						for (int l=0;l<num_channels;l++)
						{
							svfloat32_t colX = svld1_f32(pt,x_trans+i*bucket*num_channels+j*num_channels+l*SVL_fp32);
							svfloat32_t rowW1_1 = svld1_f32(pt,w1+k*num_channels+l*SVL_fp32*2);
							svmopa_za32_f32_m(0,pt,pt,colX,rowW1_1);
							svfloat32_t rowW1_2 = svld1_f32(pt,w1+k*num_channels+l*SVL_fp32*2+SVL_fp32);
							svmopa_za32_f32_m(2,pt,pt,colX,rowW1_2);
							svfloat32_t rowW2_1 = svld1_f32(pt,w2+k*num_channels+l*SVL_fp32*2);
							svmopa_za32_f32_m(1,pt,pt,colX,rowW2_1);
							svfloat32_t rowW2_2 = svld1_f32(pt,w2+k*num_channels+l*SVL_fp32*2+SVL_fp32);
							svmopa_za32_f32_m(3,pt,pt,colX,rowW2_2);
						}
						for (int t=0;t<SVL_fp32;t++)
						{
							svfloat32_t a_data_1 = svread_hor_za32_f32_m(a_data_1,pt,0,t);
							svfloat32_t b_data_1 = svread_hor_za32_f32_m(b_data_1,pt,1,t);
							svfloat32_t a_data_2 = svread_hor_za32_f32_m(a_data_2,pt,2,t);
							svfloat32_t b_data_2 = svread_hor_za32_f32_m(b_data_2,pt,3,t);
							
							b_data_1 = svmul_f32_z(pt,b_data_1,a_data_1);
							a_data_1 = svneg_f32_z(pt,a_data_1);
							a_data_1 = kutacc::fast_exp(pt, a_data_1);
							a_data_1 = svadd_n_f32_z(pt,a_data_1,1.0f);
							a_data_1 = svdiv_f32_z(pt,b_data_1,a_data_1);
							
							b_data_2 = svmul_f32_z(pt,b_data_2,a_data_2);
							a_data_2 = svneg_f32_z(pt,a_data_2);
							a_data_2 = kutacc::fast_exp(pt, a_data_2);
							a_data_2 = svadd_n_f32_z(pt,a_data_2,1.0f);
							a_data_2 = svdiv_f32_z(pt,b_data_2,a_data_2);
							
							svst1_f32(pt,resultptr + i*bucket*num_intermediate + (j+t)*num_intermediate + k,a_data_1);
							svst1_f32(pt,resultptr + i*bucket*num_intermediate + (j+t)*num_intermediate + k+SVL_fp32,a_data_2);
						}
					}
				}
		}
		SME_OFF();
	});
	free(x_trans);
}

void glu_swish_c34_bf16_bf16(bfloat16_t* xptr,bfloat16_t* resultptr,int shape0,int bucket,int num_channels,int num_intermediate, bfloat16_t* w1, bfloat16_t* w2)
{
	svbool_t pt16=svptrue_b16();
	svbool_t pt32=svptrue_b32();

    kutacc::parallel_for(0, shape0, 1, [&](int64_t start, int64_t end)
	{
		for (int i=start;i<end;i++)
		{
			svbool_t pt16=svptrue_b16();
			svbool_t pt32=svptrue_b32();
			bfloat16_t* x_trans=static_cast<bfloat16_t*>(aligned_alloc(64, sizeof(bfloat16_t) * num_channels*bucket));

			GLU_transposed(xptr + i*bucket*num_channels,x_trans,bucket,num_channels);
			SME_ON();
			for (int j=0;j<bucket;j+=SVL_fp32)
			{
				for (int k=0;k<num_intermediate;k+=SVL_fp32*2)
				{
					svzero_za();
					for (int l=0;l<num_channels;l+=2)
					{
						svbfloat16_t colX = svld1_bf16(pt16,x_trans+j*num_channels+l*SVL_fp32);
						svbfloat16_t rowW1_1 = svld1_bf16(pt16,w1+k*num_channels+l*SVL_fp32);
						svmopa_za32_bf16_m(0,pt16,pt16,colX,rowW1_1);
						svbfloat16_t rowW1_2 = svld1_bf16(pt16,w1+(k+SVL_fp32)*num_channels+l*SVL_fp32);
						svmopa_za32_bf16_m(2,pt16,pt16,colX,rowW1_2);
						svbfloat16_t rowW2_1 = svld1_bf16(pt16,w2+k*num_channels+l*SVL_fp32);
						svmopa_za32_bf16_m(1,pt16,pt16,colX,rowW2_1);
						svbfloat16_t rowW2_2 = svld1_bf16(pt16,w2+(k+SVL_fp32)*num_channels+l*SVL_fp32);
						svmopa_za32_bf16_m(3,pt16,pt16,colX,rowW2_2);
					}
					for (int t=0;t<SVL_fp32;t++)
					{
						svfloat32_t a_data_1 = svread_hor_za32_f32_m(a_data_1,pt32,0,t);
						svfloat32_t b_data_1 = svread_hor_za32_f32_m(b_data_1,pt32,1,t);
						svfloat32_t a_data_2 = svread_hor_za32_f32_m(a_data_2,pt32,2,t);
						svfloat32_t b_data_2 = svread_hor_za32_f32_m(b_data_2,pt32,3,t);
						
						b_data_1 = svmul_f32_z(pt32,b_data_1,a_data_1);
						a_data_1 = svneg_f32_z(pt32,a_data_1);
						a_data_1 = kutacc::fast_exp(pt32, a_data_1);
						a_data_1 = svadd_n_f32_z(pt32,a_data_1,1.0f);
						a_data_1 = svdiv_f32_z(pt32,b_data_1,a_data_1);
						
						b_data_2 = svmul_f32_z(pt32,b_data_2,a_data_2);
						a_data_2 = svneg_f32_z(pt32,a_data_2);
						a_data_2 = kutacc::fast_exp(pt32, a_data_2);
						a_data_2 = svadd_n_f32_z(pt32,a_data_2,1.0f);
						a_data_2 = svdiv_f32_z(pt32,b_data_2,a_data_2);
						
						svbfloat16_t a_data_bf16_1 = svcvt_bf16_f32_z(pt32, a_data_1);
						svbfloat16_t a_data_bf16_2 = svcvt_bf16_f32_z(pt32, a_data_2);
						svbfloat16_t a_data_bf16 = svuzp1_bf16(a_data_bf16_1,a_data_bf16_2);
						svst1_bf16(pt16,resultptr + i*bucket*num_intermediate + (j+t)*num_intermediate + k,a_data_bf16);
					}
				}
			}
			SME_OFF();
			free(x_trans);
		}
	});
}

void glu_swish_c34_fp32_fp32(float* xptr,float* resultptr,int shape0,int bucket,int num_channels,int num_intermediate, float* w1, float* w2)
{
	svbool_t pt=svptrue_b32();
    kutacc::parallel_for(0, shape0, 1, [&](int64_t start, int64_t end)
	{
		for (int i=start;i<end;i++)
		{
			float* x_trans=static_cast<float*>(aligned_alloc(64, sizeof(float) * num_channels*bucket));
			GLU_transposed(xptr + i*bucket*num_channels,x_trans,bucket,num_channels);
			SME_ON();
			for (int j=0;j<bucket;j+=SVL_fp32)
			{
				for (int k=0;k<num_intermediate;k+=SVL_fp32*2)
				{
					svzero_za();
					for (int l=0;l<num_channels;l++)
					{
						svfloat32_t colX = svld1_f32(pt,x_trans+j*num_channels+l*SVL_fp32);
						svfloat32_t rowW1_1 = svld1_f32(pt,w1+k*num_channels+l*SVL_fp32);
						svfloat32_t rowW2_1 = svld1_f32(pt,w2+k*num_channels+l*SVL_fp32);
						
						svfloat32_t rowW1_2 = svld1_f32(pt,w1+(k+SVL_fp32)*num_channels+l*SVL_fp32);
						svfloat32_t rowW2_2 = svld1_f32(pt,w2+(k+SVL_fp32)*num_channels+l*SVL_fp32);
						svmopa_za32_f32_m(0,pt,pt,colX,rowW1_1);
						svmopa_za32_f32_m(1,pt,pt,colX,rowW2_1);
						svmopa_za32_f32_m(2,pt,pt,colX,rowW1_2);
						svmopa_za32_f32_m(3,pt,pt,colX,rowW2_2);
					}
					for (int t=0;t<SVL_fp32;t++)
					{
						svfloat32_t a_data_1 = svread_hor_za32_f32_m(a_data_1,pt,0,t);
						svfloat32_t b_data_1 = svread_hor_za32_f32_m(b_data_1,pt,1,t);
						svfloat32_t a_data_2 = svread_hor_za32_f32_m(a_data_2,pt,2,t);
						svfloat32_t b_data_2 = svread_hor_za32_f32_m(b_data_2,pt,3,t);
						
						b_data_1 = svmul_f32_z(pt,b_data_1,a_data_1);
						a_data_1 = svneg_f32_z(pt,a_data_1);
						a_data_1 = kutacc::fast_exp(pt, a_data_1);
						a_data_1 = svadd_n_f32_z(pt,a_data_1,1.0f);
						a_data_1 = svdiv_f32_z(pt,b_data_1,a_data_1);
						
						b_data_2 = svmul_f32_z(pt,b_data_2,a_data_2);
						a_data_2 = svneg_f32_z(pt,a_data_2);
						a_data_2 = kutacc::fast_exp(pt, a_data_2);
						a_data_2 = svadd_n_f32_z(pt,a_data_2,1.0f);
						a_data_2 = svdiv_f32_z(pt,b_data_2,a_data_2);
						
						svst1_f32(pt,resultptr + i*bucket*num_intermediate + (j+t)*num_intermediate + k,a_data_1);
						svst1_f32(pt,resultptr + i*bucket*num_intermediate + (j+t)*num_intermediate + k+SVL_fp32,a_data_2);
					}
				}
			}
			SME_OFF();
			free(x_trans);
		}
	});
}
template<typename x_t, typename r_t, typename w1_t, typename w2_t>
void glu_swish(x_t* xptr, r_t* resultptr, int shape0, int bucket, int num_channels, int num_intermediate, w1_t* w1, w2_t* w2, bool hasbatch){
	if(shape0 == 0){
		glu_swish_c2((bfloat16_t*)xptr, (bfloat16_t*)resultptr, bucket, num_channels, num_intermediate, (bfloat16_t*)w1, (bfloat16_t*)w2);
	}else if(sizeof(x_t) == 2){
		glu_swish_c34_bf16_bf16((bfloat16_t*)xptr, (bfloat16_t*)resultptr, shape0, bucket, num_channels, num_intermediate, (bfloat16_t*)w1, (bfloat16_t*)w2);
	}else if(hasbatch){
		if(sizeof(r_t) == 2){
			glu_swish_cbatch_bf16((float*)xptr, (bfloat16_t*)resultptr, shape0, bucket, num_channels, num_intermediate, (float*)w1, (float*)w2);
		}else{
			glu_swish_cbatch_fp32((float*)xptr, (float*)resultptr, shape0, bucket, num_channels, num_intermediate, (float*)w1, (float*)w2);
		}
	}else{
		if(sizeof(r_t) == 2){
			glu_swish_c34_fp32_bf16((float*)xptr, (bfloat16_t*)resultptr, shape0, bucket, num_channels, num_intermediate, (float*)w1, (float*)w2);
		}else{
			glu_swish_c34_fp32_fp32((float*)xptr, (float*)resultptr, shape0, bucket, num_channels, num_intermediate, (float*)w1, (float*)w2);
		}
	}
}
template void glu_swish<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t>(bfloat16_t*, bfloat16_t*, int, int, int, int, bfloat16_t*, bfloat16_t*, bool);
template void glu_swish<float, bfloat16_t, float, float>(float* , bfloat16_t*, int, int, int, int, float*, float*, bool);
template void glu_swish<float, float, float, float>(float* , float*, int, int, int, int, float*, float*, bool);

} //namespace kutacc
