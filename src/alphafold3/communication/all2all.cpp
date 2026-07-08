#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#include <string.h>
#include <arm_sve.h>
#include <iostream>
#include "kutacc.h"
#include "utils/collapse.h"
namespace kutacc{
template<typename data_t>
void shm_all2all(
    const data_t * src,
    data_t * out_ptr,
    int64_t m_full,
    int64_t n_full,
    int64_t inner_size,
    int64_t size0,
    int64_t size1,
    int64_t input_stride0,
    int64_t input_stride1,
    int64_t output_stride0,
    int64_t output_stride1,
    int64_t m_rank,
    int64_t m_buffer_size,
    void * m_kupl_recvbuf,
    kupl_shm_win_h m_recvbuf_win
) {
    const int64_t world_size = 8;
    const int64_t rank = m_rank;
    // const int64_t inner_size = get_inner_size(data);
    const int64_t block_m = (m_full + world_size - 1) / world_size;
    const int64_t block_n = (n_full + world_size - 1) / world_size;

    bool is_split_dim0 = (size0 < m_full);
    bool is_split_dim1 = (size1 < n_full);


    void* shared_buffer = m_kupl_recvbuf;
    if(shared_buffer == nullptr){
        std::cout << "Shared buffer not initialized" << std::endl;
    } 
    uint8_t* buffers[8];
    for (int i = 0; i < 8; ++i) {
        kupl_shm_win_query(m_recvbuf_win, i, (void**)&buffers[i]);
    }

    const int64_t elem_size = sizeof(data_t);
    const int64_t total_shared_size = 768*768*1024;
    const int64_t actual_per_rank = total_shared_size / world_size;
    const int64_t required_per_rank = world_size * block_m * block_n * inner_size * elem_size;

    const int64_t denominator = world_size * block_n * inner_size * elem_size;
    const int64_t subblock_m = std::max<int64_t>(1, m_buffer_size / denominator);

    for (int64_t block_mi_start = 0; block_mi_start < block_m; block_mi_start += subblock_m) {
        int64_t size_m = std::min(subblock_m, block_m - block_mi_start);
        int64_t par_size = world_size * size_m * block_n;
        // ============ STAGE 1: Scatter ============
        kutacc::parallel_for(0, par_size, 1, [&](int64_t start, int64_t end) {
            kutacc::collapse_for(start, end, world_size, size_m, block_n,
                [&](int64_t target_rank, int64_t sbmi, int64_t bni) {
                    int64_t local_m_idx = sbmi + block_mi_start;
                    int64_t local_n_idx = bni;

                    bool valid_in_input = false;
                    int64_t src_linear_idx = 0;

                    if (is_split_dim1) {
                        int64_t global_m = target_rank * block_m + local_m_idx;
                        valid_in_input = (global_m < m_full) && 
                                        (local_n_idx < (n_full - rank * block_n));
                        if (valid_in_input) {
                            src_linear_idx = global_m * input_stride0 + local_n_idx * input_stride1;
                        }
                    } else {
                        int64_t global_n = target_rank * block_n + local_n_idx;
                        valid_in_input = (local_m_idx < (m_full - rank * block_m)) && 
                                        (global_n < n_full);
                        if (valid_in_input) {
                            src_linear_idx = local_m_idx * input_stride0 + global_n * input_stride1;
                        }
                    }

                    if (!valid_in_input) return;

                    if (target_rank == rank) {
                        // Self-bypass
                        int64_t out_linear_idx = 0;
                        if (is_split_dim1) {
                            int64_t global_n = rank * block_n + local_n_idx;
                            out_linear_idx = local_m_idx * output_stride0 + global_n * output_stride1;
                        } else {
                            int64_t global_m = rank * block_m + local_m_idx;
                            out_linear_idx = global_m * output_stride0 + local_n_idx * output_stride1;
                        }

                        // SVE copy with CORRECT predicate
                        int64_t i = 0;
                        if(elem_size == 2){
                            while (i < inner_size) {
                                svbool_t pg = svwhilelt_b16(i, inner_size);
                                svuint16_t v = svld1_u16(pg, (uint16_t*)src + src_linear_idx + i);
                                svst1_u16(pg, (uint16_t*)out_ptr + out_linear_idx + i, v);
                                i += svcnth();
                            }
                        }else{
                            while (i < inner_size) {
                                svbool_t pg = svwhilelt_b32(i, inner_size);
                                svfloat32_t v = svld1_f32(pg, (float*)src + src_linear_idx + i);
                                svst1_f32(pg, (float*)out_ptr + out_linear_idx + i, v);
                                i += svcntw();
                            }
                        }
                    } else {
                        // Send to others
                        data_t* dst = reinterpret_cast<data_t*>(buffers[target_rank]) +
                                      (rank * block_m * block_n + local_m_idx * block_n + local_n_idx) * inner_size;

                        int64_t i = 0;
                        if(elem_size == 2){
                            while (i < inner_size) {
                                svbool_t pg = svwhilelt_b16(i, inner_size);
                                svuint16_t v = svld1_u16(pg, (uint16_t*)src + src_linear_idx + i);
                                svst1_u16(pg, (uint16_t*)dst + i, v);
                                i += svcnth();
                            }
                        }else{
                            while (i < inner_size) {
                                svbool_t pg = svwhilelt_b32(i, inner_size);
                                svfloat32_t v = svld1_f32(pg, (float*)src + src_linear_idx + i);
                                svst1_f32(pg, (float*)dst + i, v);
                                i += svcntw();
                            }
                        }
                    }
                });
        });
        kupl_shm_fence(m_recvbuf_win);

        // ============ STAGE 2: Gather ============
        kutacc::parallel_for(0, par_size, 1, [&](int64_t start, int64_t end) {
            kutacc::collapse_for(start, end, world_size, size_m, block_n,
                [&](int64_t src_rank, int64_t sbmi, int64_t bni) {
                    if (src_rank == rank) return;

                    int64_t local_m_idx = sbmi + block_mi_start;
                    int64_t local_n_idx = bni;

                    bool valid_in_output = false;
                    int64_t out_linear_idx = 0;

                    if (is_split_dim1) {
                        int64_t global_m = rank * block_m + local_m_idx;
                        int64_t global_n = src_rank * block_n + local_n_idx;
                        valid_in_output = (global_m < m_full) && (global_n < n_full);
                        if (valid_in_output) {
                            out_linear_idx = local_m_idx * output_stride0 + global_n * output_stride1;
                        }
                    } else {
                        int64_t global_m = src_rank * block_m + local_m_idx;
                        int64_t global_n = rank * block_n + local_n_idx;
                        valid_in_output = (global_m < m_full) && (global_n < n_full);
                        if (valid_in_output) {
                            out_linear_idx = global_m * output_stride0 + local_n_idx * output_stride1;
                        }
                    }

                    if (!valid_in_output) return;

                    data_t* src_ptr = reinterpret_cast<data_t*>(buffers[rank]) +
                                      (src_rank * block_m * block_n + local_m_idx * block_n + local_n_idx) * inner_size;
                    data_t* dst = out_ptr + out_linear_idx;

                    int64_t i = 0;
                    if(elem_size == 2){
                        while (i < inner_size) {
                            svbool_t pg = svwhilelt_b16(i, inner_size);
                            svuint16_t v = svld1_u16(pg, (uint16_t*)src_ptr + i);
                            svst1_u16(pg, (uint16_t*)dst + i, v);
                            i += svcnth();
                        }
                    }else{
                        while (i < inner_size) {
                            svbool_t pg = svwhilelt_b32(i, inner_size);
                            svfloat32_t v = svld1_f32(pg, (float*)src_ptr + i);
                            svst1_f32(pg, (float*)dst + i, v);
                            i += svcntw();
                        }
                    }
                });
        });
        kupl_shm_fence(m_recvbuf_win);
    }
    kupl_shm_fence(m_recvbuf_win);// Ensure all done before return
}

template void shm_all2all<float>(const float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, void *, kupl_shm_win_h);
template void shm_all2all<uint16_t>(const uint16_t*, uint16_t*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, void *, kupl_shm_win_h);

}// namespace kutacc