#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#include <string.h>
#include <arm_sve.h>
#include <iostream>
#include "kutacc.h"
namespace kutacc{
// 优化的BF16 allgather函数
void shm_allgather_comm8v2_optimized_bf16(
    void *sendbuf, 
    int64_t sendcount, 
    void *recvbuf, 
    int64_t recvcount, 
    int64_t rank,
    uint8_t **buffers, 
    kupl_shm_win_h m_recvbuf_win
) {
    // 优化点2: 改进负载均衡
    int64_t base_part = sendcount / 32;
    int64_t remainder = sendcount % 32;
    int64_t part_lengths[32];
    int64_t start_indices[32];
    int64_t current_start = 0;
    for (int i = 0; i < 32; ++i) {
        part_lengths[i] = base_part + (i < remainder ? 1 : 0);
        start_indices[i] = current_start;
        current_start += part_lengths[i];
    }

    int64_t my_dieleader = rank < 4 ? 0 : 4;
    int64_t read_from_rank[8] = {4, 5, 6, 7, 0, 1, 2, 3};

    // 优化点3: SVE向量化优化
    const int64_t vl = svcnth();
    svbool_t all_pred = svptrue_b16();
    int64_t src_rank = read_from_rank[rank];

    kutacc::parallel_for(0, 32, 1, [&](int64_t start, int64_t end) {
        // 优化点1: 预计算指针
        uint16_t* peer_buffers[4];
        for (int p = 0; p < 4; ++p) {
            int peer = my_dieleader + p;
            peer_buffers[p] = reinterpret_cast<uint16_t*>(buffers[peer]);
        }
        
        const uint16_t* src_buffer = reinterpret_cast<const uint16_t*>(buffers[src_rank]);
        const uint16_t* send_buffer = reinterpret_cast<const uint16_t*>(sendbuf);

        for (int tid = start; tid < end; ++tid) {
            int64_t part_length = part_lengths[tid];
            int64_t start_i = start_indices[tid];
            
            if (part_length == 0) continue;

            // 阶段1: 写入同die所有rank
            const uint16_t* sendbuf_part = send_buffer + start_i;
            int64_t full_vectors = part_length / vl;
            int64_t remainder_elements = part_length % vl;

            // 预计算每个peer的目标指针
            uint16_t* peer_dst_ptrs[4];
            for (int p = 0; p < 4; ++p) {
                peer_dst_ptrs[p] = peer_buffers[p] + (rank * sendcount + start_i);
            }

            // 优化点1: 循环重排序 - 先处理所有peer
            for (int p = 0; p < 4; ++p) {
                if (my_dieleader + p == rank) continue; // 跳过自己
                
                uint16_t* dst = peer_dst_ptrs[p];
                
                // 主循环使用全predicate
                for (int64_t i = 0; i < full_vectors * vl; i += vl) {
                    auto sve = svld1_u16(all_pred, sendbuf_part + i);
                    svst1_u16(all_pred, dst + i, sve);
                }
                
                // 尾部处理
                if (remainder_elements > 0) {
                    svbool_t pred = svwhilelt_b16(full_vectors * vl, part_length);
                    auto sve = svld1_u16(pred, sendbuf_part + full_vectors * vl);
                    svst1_u16(pred, dst + full_vectors * vl, sve);
                }
            }

            // 阶段2: 从对侧die读取并广播
            const uint16_t* read_from = src_buffer + (src_rank * sendcount + start_i);
            
            // 预计算每个peer的目标指针
            uint16_t* peer_bcast_ptrs[4];
            for (int p = 0; p < 4; ++p) {
                peer_bcast_ptrs[p] = peer_buffers[p] + (src_rank * sendcount + start_i);
            }

            // 预取源数据
            __builtin_prefetch(read_from, 0, 3);

            for (int p = 0; p < 4; ++p) {
                uint16_t* dst = peer_bcast_ptrs[p];
                
                for (int64_t i = 0; i < full_vectors * vl; i += vl) {
                    // 预取下一块
                    if (i + vl < part_length) {
                        __builtin_prefetch(read_from + i + vl, 0, 2);
                    }
                    auto sve = svld1_u16(all_pred, read_from + i);
                    svst1_u16(all_pred, dst + i, sve);
                }
                
                // 尾部处理
                if (remainder_elements > 0) {
                    svbool_t pred = svwhilelt_b16(full_vectors * vl, part_length);
                    auto sve = svld1_u16(pred, read_from + full_vectors * vl);
                    svst1_u16(pred, dst + full_vectors * vl, sve);
                }
            }
        }
    });

    kupl_shm_fence(m_recvbuf_win);
}

// 优化的FP32 allgather函数
void shm_allgather_comm8v2_optimized_fp32(
    void *sendbuf, 
    int64_t sendcount, 
    void *recvbuf, 
    int64_t recvcount, 
    int64_t rank,
    uint8_t **buffers, 
    kupl_shm_win_h m_recvbuf_win
) {
    // 优化点2: 改进负载均衡
    int64_t base_part = sendcount / 32;
    int64_t remainder = sendcount % 32;
    int64_t part_lengths[32];
    int64_t start_indices[32];
    int64_t current_start = 0;
    for (int i = 0; i < 32; ++i) {
        part_lengths[i] = base_part + (i < remainder ? 1 : 0);
        start_indices[i] = current_start;
        current_start += part_lengths[i];
    }

    int64_t my_dieleader = rank < 4 ? 0 : 4;
    int64_t read_from_rank[8] = {4, 5, 6, 7, 0, 1, 2, 3};

    // 优化点3: SVE向量化优化
    const int64_t vl = svcntw();
    svbool_t all_pred = svptrue_b32();
    int64_t src_rank = read_from_rank[rank];

    kutacc::parallel_for(0, 32, 1, [&](int64_t start, int64_t end) {
        // 优化点1: 预计算指针
        float* peer_buffers[4];
        for (int p = 0; p < 4; ++p) {
            int peer = my_dieleader + p;
            peer_buffers[p] = reinterpret_cast<float*>(buffers[peer]);
        }
        
        const float* src_buffer = reinterpret_cast<const float*>(buffers[src_rank]);
        const float* send_buffer = reinterpret_cast<const float*>(sendbuf);

        for (int tid = start; tid < end; ++tid) {
            int64_t part_length = part_lengths[tid];
            int64_t start_i = start_indices[tid];
            
            if (part_length == 0) continue;

            // 阶段1: 写入同die所有rank
            const float* sendbuf_part = send_buffer + start_i;
            int64_t full_vectors = part_length / vl;
            int64_t remainder_elements = part_length % vl;

            // 预计算每个peer的目标指针
            float* peer_dst_ptrs[4];
            for (int p = 0; p < 4; ++p) {
                peer_dst_ptrs[p] = peer_buffers[p] + (rank * sendcount + start_i);
            }

            // 优化点1: 循环重排序
            for (int p = 0; p < 4; ++p) {
                if (my_dieleader + p == rank) continue;
                
                float* dst = peer_dst_ptrs[p];
                
                for (int64_t i = 0; i < full_vectors * vl; i += vl) {
                    auto sve = svld1_f32(all_pred, sendbuf_part + i);
                    svst1_f32(all_pred, dst + i, sve);
                }
                
                if (remainder_elements > 0) {
                    svbool_t pred = svwhilelt_b32(full_vectors * vl, part_length);
                    auto sve = svld1_f32(pred, sendbuf_part + full_vectors * vl);
                    svst1_f32(pred, dst + full_vectors * vl, sve);
                }
            }

            // 阶段2: 从对侧die读取并广播
            const float* read_from = src_buffer + (src_rank * sendcount + start_i);
            
            float* peer_bcast_ptrs[4];
            for (int p = 0; p < 4; ++p) {
                peer_bcast_ptrs[p] = peer_buffers[p] + (src_rank * sendcount + start_i);
            }

            __builtin_prefetch(read_from, 0, 3);

            for (int p = 0; p < 4; ++p) {
                float* dst = peer_bcast_ptrs[p];
                
                for (int64_t i = 0; i < full_vectors * vl; i += vl) {
                    if (i + vl < part_length) {
                        __builtin_prefetch(read_from + i + vl, 0, 2);
                    }
                    auto sve = svld1_f32(all_pred, read_from + i);
                    svst1_f32(all_pred, dst + i, sve);
                }
                
                if (remainder_elements > 0) {
                    svbool_t pred = svwhilelt_b32(full_vectors * vl, part_length);
                    auto sve = svld1_f32(pred, read_from + full_vectors * vl);
                    svst1_f32(pred, dst + full_vectors * vl, sve);
                }
            }
        }
    });

    kupl_shm_fence(m_recvbuf_win);
}

void parallel_vectorized_copy_optimized_bf16(const uint16_t* src, uint16_t* dst, int64_t total_elements) {
    // 优化点2: 改进负载均衡
    int64_t base_part = total_elements / 32;
    int64_t remainder = total_elements % 32;
    int64_t part_lengths[32];
    int64_t start_indices[32];
    int64_t current_start = 0;
    for (int i = 0; i < 32; ++i) {
        part_lengths[i] = base_part + (i < remainder ? 1 : 0);
        start_indices[i] = current_start;
        current_start += part_lengths[i];
    }

    // 优化点3: SVE向量化优化
    const int64_t vl = svcnth();
    svbool_t all_pred = svptrue_b16();

    kutacc::parallel_for(0, 32, 1, [&](int64_t start, int64_t end) {
        for (int64_t tid = start; tid < end; ++tid) {
            int64_t part_length = part_lengths[tid];
            int64_t start_i = start_indices[tid];
            
            if (part_length == 0) continue;

            const uint16_t* thread_src = src + start_i;
            uint16_t* thread_dst = dst + start_i;
            
            int64_t full_vectors = part_length / vl;
            int64_t remainder_elements = part_length % vl;

            // 主循环使用全predicate
            for (int64_t i = 0; i < full_vectors * vl; i += vl) {
                auto data = svld1_u16(all_pred, reinterpret_cast<const uint16_t*>(thread_src + i));
                svst1_u16(all_pred, reinterpret_cast<uint16_t*>(thread_dst + i), data);
            }

            // 尾部处理
            if (remainder_elements > 0) {
                int64_t tail_start = full_vectors * vl;
                svbool_t pred = svwhilelt_b16(tail_start, part_length);
                auto data = svld1_u16(pred, reinterpret_cast<const uint16_t*>(thread_src + tail_start));
                svst1_u16(pred, reinterpret_cast<uint16_t*>(thread_dst + tail_start), data);
            }
        }
    });
}

void parallel_vectorized_copy_optimized_fp32(const float* src, float* dst, int64_t total_elements) {
    // 优化点2: 改进负载均衡
    int64_t base_part = total_elements / 32;
    int64_t remainder = total_elements % 32;
    int64_t part_lengths[32];
    int64_t start_indices[32];
    int64_t current_start = 0;
    for (int i = 0; i < 32; ++i) {
        part_lengths[i] = base_part + (i < remainder ? 1 : 0);
        start_indices[i] = current_start;
        current_start += part_lengths[i];
    }

    // 优化点3: SVE向量化优化
    const int64_t vl = svcntw();
    svbool_t all_pred = svptrue_b32();

    kutacc::parallel_for(0, 32, 1, [&](int64_t start, int64_t end) {
        for (int64_t tid = start; tid < end; ++tid) {
            int64_t part_length = part_lengths[tid];
            int64_t start_i = start_indices[tid];
            
            if (part_length == 0) continue;

            const float* thread_src = src + start_i;
            float* thread_dst = dst + start_i;
            
            int64_t full_vectors = part_length / vl;
            int64_t remainder_elements = part_length % vl;

            // 主循环使用全predicate
            for (int64_t i = 0; i < full_vectors * vl; i += vl) {
                auto data = svld1_f32(all_pred, thread_src + i);
                svst1_f32(all_pred, thread_dst + i, data);
            }

            // 尾部处理
            if (remainder_elements > 0) {
                int64_t tail_start = full_vectors * vl;
                svbool_t pred = svwhilelt_b32(tail_start, part_length);
                auto data = svld1_f32(pred, thread_src + tail_start);
                svst1_f32(pred, thread_dst + tail_start, data);
            }
        }
    });
}
template <typename data_t>
uint8_t* shm_all_gather(
    const data_t* src, 
    int64_t local_elements,
    size_t elem_size,
    int64_t m_rank,
    int64_t m_world_size,
    void * m_kupl_recvbuf,
    kupl_shm_win_h m_recvbuf_win
) {
    const int64_t world_size = m_world_size;
    const int64_t total_elements = world_size * local_elements;
    void* shared_buffer = m_kupl_recvbuf;
    void* my_send_region = static_cast<char*>(shared_buffer) + m_rank * local_elements * elem_size;
    if(sizeof(data_t) == 2){
        uint16_t* dst = reinterpret_cast<uint16_t*>(my_send_region);
        kutacc::parallel_vectorized_copy_optimized_bf16((uint16_t*)src, dst, local_elements);
    }else{
        float* dst = reinterpret_cast<float*>(my_send_region);
        kutacc::parallel_vectorized_copy_optimized_fp32((float*)src, dst, local_elements);
    }

    uint8_t* buffers[8];
    for (int i = 0; i < 8; ++i) {
        kupl_shm_win_query(m_recvbuf_win, i, (void**)&buffers[i]);
    }
    kupl_shm_fence(m_recvbuf_win);
    if(sizeof(data_t) == 2){
        kutacc::shm_allgather_comm8v2_optimized_bf16(
            my_send_region, local_elements, shared_buffer, total_elements,
            m_rank, buffers, m_recvbuf_win
        );
    }else{
        kutacc::shm_allgather_comm8v2_optimized_fp32(
            my_send_region, local_elements, shared_buffer, total_elements,
            m_rank, buffers, m_recvbuf_win
        );
    }
    return buffers[m_rank];
}

template uint8_t* shm_all_gather<float>(const float*, int64_t , size_t, int64_t, int64_t, void *, kupl_shm_win_h);
template uint8_t* shm_all_gather<uint16_t>(const uint16_t*, int64_t , size_t, int64_t, int64_t, void *, kupl_shm_win_h);

}// namespace kutacc