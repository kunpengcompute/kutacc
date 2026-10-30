#include <stdio.h>
#include <omp.h>
#include <unistd.h>
#include <string.h>
#include <arm_sve.h>
#include <iostream>
#include <vector>
#include "kutacc.h"
namespace kutacc {
void shm_allgather_comm8v2_32_dim1_fp32(
    const void* sendbuf,
    int64_t B,
    int64_t H_local,
    int64_t inner_dims_numel,
    void* recvbuf,
    int64_t rank,
    uint8_t** buffers,
    kupl_shm_win_h m_recvbuf_win
) {
    const int64_t world_size = 8;
    int64_t H_global = H_local * world_size;
    
    int64_t base_part = H_local / 32;
    int64_t remainder = H_local % 32;
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

    const int64_t vl = svcntw();
    svbool_t all_pred = svptrue_b32();
    int64_t full_vectors = inner_dims_numel / vl;
    int64_t remainder_elements = inner_dims_numel % vl;

    kutacc::parallel_for(0, 32, 1, [&](int64_t start, int64_t end) {
        float* peer_buffers[4];
        for (int p = 0; p < 4; ++p) {
            int peer = my_dieleader + p;
            peer_buffers[p] = reinterpret_cast<float*>(buffers[peer]);
        }
        
        for (int tid = start; tid < end; ++tid) {
            int64_t part_length = part_lengths[tid];
            int64_t start_i = start_indices[tid];
            
            for (int64_t b = 0; b < B; ++b) {
                const float* sendbuf_batch = static_cast<const float*>(sendbuf) + b * (H_local * inner_dims_numel);
                int64_t batch_offset = b * (H_global * inner_dims_numel);
                int64_t my_global_offset = rank * H_local * inner_dims_numel;

                float* peer_batch_ptrs[4];
                for (int p = 0; p < 4; ++p) {
                    peer_batch_ptrs[p] = peer_buffers[p] + batch_offset + my_global_offset;
                }

                for (int64_t i = 0; i < part_length; ++i) {
                    int64_t local_i = start_i + i;
                    int64_t row_offset = local_i * inner_dims_numel;
                    const float* src_row = sendbuf_batch + row_offset;

                    for (int p = 0; p < 4; ++p) {
                        float* dst_row = peer_batch_ptrs[p] + row_offset;

                        for (int64_t inner = 0; inner < full_vectors * vl; inner += vl) {
                            auto sve = svld1_f32(all_pred, src_row + inner);
                            svst1_f32(all_pred, dst_row + inner, sve);
                        }

                        if (remainder_elements > 0) {
                            svbool_t pred = svwhilelt_b32(full_vectors * vl, inner_dims_numel);
                            auto sve = svld1_f32(pred, src_row + full_vectors * vl);
                            svst1_f32(pred, dst_row + full_vectors * vl, sve);
                        }
                    }
                }
            }
        }
    });

    int64_t src_rank = read_from_rank[rank];
    kutacc::parallel_for(0, 32, 1, [&](int64_t start, int64_t end) {
        float* peer_buffers[4];
        for (int p = 0; p < 4; ++p) {
            int peer = my_dieleader + p;
            peer_buffers[p] = reinterpret_cast<float*>(buffers[peer]);
        }
        
        const float* src_buffer = reinterpret_cast<const float*>(buffers[src_rank]);
        
        for (int tid = start; tid < end; ++tid) {
            int64_t part_length = part_lengths[tid];
            int64_t start_i = start_indices[tid];
            
            for (int64_t b = 0; b < B; ++b) {
                int64_t batch_offset = b * (H_global * inner_dims_numel);
                int64_t src_global_offset = src_rank * H_local * inner_dims_numel;
                const float* src_base = src_buffer + batch_offset + src_global_offset;

                float* peer_batch_ptrs[4];
                for (int p = 0; p < 4; ++p) {
                    peer_batch_ptrs[p] = peer_buffers[p] + batch_offset + src_global_offset;
                }

                for (int64_t i = 0; i < part_length; ++i) {
                    int64_t local_i = start_i + i;
                    int64_t row_offset = local_i * inner_dims_numel;
                    const float* src_row = src_base + row_offset;

                    __builtin_prefetch(src_row, 0, 3);

                    for (int p = 0; p < 4; ++p) {
                        float* dst_row = peer_batch_ptrs[p] + row_offset;
                        for (int64_t inner = 0; inner < full_vectors * vl; inner += vl) {
                            if (inner + vl < inner_dims_numel) {
                                __builtin_prefetch(src_row + inner + vl, 0, 2);
                            }
                            auto sve = svld1_f32(all_pred, src_row + inner);
                            svst1_f32(all_pred, dst_row + inner, sve);
                        }

                        if (remainder_elements > 0) {
                            svbool_t pred = svwhilelt_b32(full_vectors * vl, inner_dims_numel);
                            auto sve = svld1_f32(pred, src_row + full_vectors * vl);
                            svst1_f32(pred, dst_row + full_vectors * vl, sve);
                        }
                    }
                }
            }
        }
    });

    kupl_shm_fence(m_recvbuf_win);
}

void shm_allgather_comm8v2_32_dim1_bf16(
    const void* sendbuf,
    int64_t B,
    int64_t H_local,
    int64_t inner_dims_numel,
    void* recvbuf,
    int64_t rank,
    uint8_t** buffers,
    kupl_shm_win_h m_recvbuf_win
) {
    const int64_t world_size = 8;
    int64_t H_global = H_local * world_size;
    
    int64_t base_part = H_local / 32;
    int64_t remainder = H_local % 32;
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

    const int64_t vl = svcnth();
    svbool_t all_pred = svptrue_b16();
    int64_t full_vectors = inner_dims_numel / vl;
    int64_t remainder_elements = inner_dims_numel % vl;

    kutacc::parallel_for(0, 32, 1, [&](int64_t start, int64_t end) {
        uint16_t* peer_buffers[4];
        for (int p = 0; p < 4; ++p) {
            int peer = my_dieleader + p;
            peer_buffers[p] = reinterpret_cast<uint16_t*>(buffers[peer]);
        }
        
        for (int tid = start; tid < end; ++tid) {
            int64_t part_length = part_lengths[tid];
            int64_t start_i = start_indices[tid];
            
            for (int64_t b = 0; b < B; ++b) {
                const uint16_t* sendbuf_batch = static_cast<const uint16_t*>(sendbuf) + b * (H_local * inner_dims_numel);
                int64_t batch_offset = b * (H_global * inner_dims_numel);
                int64_t my_global_offset = rank * H_local * inner_dims_numel;

                uint16_t* peer_batch_ptrs[4];
                for (int p = 0; p < 4; ++p) {
                    peer_batch_ptrs[p] = peer_buffers[p] + batch_offset + my_global_offset;
                }

                for (int64_t i = 0; i < part_length; ++i) {
                    int64_t local_i = start_i + i;
                    int64_t row_offset = local_i * inner_dims_numel;
                    const uint16_t* src_row = sendbuf_batch + row_offset;

                    for (int p = 0; p < 4; ++p) {
                        uint16_t* dst_row = peer_batch_ptrs[p] + row_offset;

                        for (int64_t inner = 0; inner < full_vectors * vl; inner += vl) {
                            auto sve = svld1_u16(all_pred, src_row + inner);
                            svst1_u16(all_pred, dst_row + inner, sve);
                        }

                        if (remainder_elements > 0) {
                            svbool_t pred = svwhilelt_b16(full_vectors * vl, inner_dims_numel);
                            auto sve = svld1_u16(pred, src_row + full_vectors * vl);
                            svst1_u16(pred, dst_row + full_vectors * vl, sve);
                        }
                    }
                }
            }
        }
    });

    int64_t src_rank = read_from_rank[rank];
    kutacc::parallel_for(0, 32, 1, [&](int64_t start, int64_t end) {
        uint16_t* peer_buffers[4];
        for (int p = 0; p < 4; ++p) {
            int peer = my_dieleader + p;
            peer_buffers[p] = reinterpret_cast<uint16_t*>(buffers[peer]);
        }
        
        const uint16_t* src_buffer = reinterpret_cast<const uint16_t*>(buffers[src_rank]);
        
        for (int tid = start; tid < end; ++tid) {
            int64_t part_length = part_lengths[tid];
            int64_t start_i = start_indices[tid];
            
            for (int64_t b = 0; b < B; ++b) {
                int64_t batch_offset = b * (H_global * inner_dims_numel);
                int64_t src_global_offset = src_rank * H_local * inner_dims_numel;
                const uint16_t* src_base = src_buffer + batch_offset + src_global_offset;

                uint16_t* peer_batch_ptrs[4];
                for (int p = 0; p < 4; ++p) {
                    peer_batch_ptrs[p] = peer_buffers[p] + batch_offset + src_global_offset;
                }

                for (int64_t i = 0; i < part_length; ++i) {
                    int64_t local_i = start_i + i;
                    int64_t row_offset = local_i * inner_dims_numel;
                    const uint16_t* src_row = src_base + row_offset;

                    __builtin_prefetch(src_row, 0, 3);

                    for (int p = 0; p < 4; ++p) {
                        uint16_t* dst_row = peer_batch_ptrs[p] + row_offset;

                        for (int64_t inner = 0; inner < full_vectors * vl; inner += vl) {
                            if (inner + vl < inner_dims_numel) {
                                __builtin_prefetch(src_row + inner + vl, 0, 2);
                            }
                            auto sve = svld1_u16(all_pred, src_row + inner);
                            svst1_u16(all_pred, dst_row + inner, sve);
                        }

                        if (remainder_elements > 0) {
                            svbool_t pred = svwhilelt_b16(full_vectors * vl, inner_dims_numel);
                            auto sve = svld1_u16(pred, src_row + full_vectors * vl);
                            svst1_u16(pred, dst_row + full_vectors * vl, sve);
                        }
                    }
                }
            }
        }
    });

    kupl_shm_fence(m_recvbuf_win);
}


template <typename data_t>
void shm_allgather_dim1_write_local(
    const data_t* src,
    data_t* dst,
    int64_t B,
    int64_t H_local,
    int64_t H_global,
    int64_t inner_dims_numel,
    int64_t m_rank
) {
    const int64_t row = H_local * inner_dims_numel;
    kutacc::parallel_for(0, B, 1, [&](int64_t start_b, int64_t end_b) {
        for (int64_t b = start_b; b < end_b; ++b) {
            data_t* dst_batch = dst + b * (H_global * inner_dims_numel) + m_rank * row;
            const data_t* src_batch = src + b * row;
            if (sizeof(data_t) == 4) {
                const int64_t vl = svcntw();
                svbool_t all_pred = svptrue_b32();
                int64_t full_vectors = inner_dims_numel / vl;
                int64_t rem = inner_dims_numel % vl;
                for (int64_t h = 0; h < H_local; ++h) {
                    float* dst_row = reinterpret_cast<float*>(dst_batch + h * inner_dims_numel);
                    const float* src_row = reinterpret_cast<const float*>(src_batch + h * inner_dims_numel);
                    for (int64_t inner = 0; inner < full_vectors * vl; inner += vl) {
                        auto sve = svld1_f32(all_pred, src_row + inner);
                        svst1_f32(all_pred, dst_row + inner, sve);
                    }
                    if (rem > 0) {
                        svbool_t pred = svwhilelt_b32(full_vectors * vl, inner_dims_numel);
                        auto sve = svld1_f32(pred, src_row + full_vectors * vl);
                        svst1_f32(pred, dst_row + full_vectors * vl, sve);
                    }
                }
            } else {
                const int64_t vl = svcnth();
                svbool_t all_pred = svptrue_b16();
                int64_t full_vectors = inner_dims_numel / vl;
                int64_t rem = inner_dims_numel % vl;
                for (int64_t h = 0; h < H_local; ++h) {
                    uint16_t* dst_row = reinterpret_cast<uint16_t*>(dst_batch + h * inner_dims_numel);
                    const uint16_t* src_row = reinterpret_cast<const uint16_t*>(src_batch + h * inner_dims_numel);
                    for (int64_t inner = 0; inner < full_vectors * vl; inner += vl) {
                        auto sve = svld1_u16(all_pred, src_row + inner);
                        svst1_u16(all_pred, dst_row + inner, sve);
                    }
                    if (rem > 0) {
                        svbool_t pred = svwhilelt_b16(full_vectors * vl, inner_dims_numel);
                        auto sve = svld1_u16(pred, src_row + full_vectors * vl);
                        svst1_u16(pred, dst_row + full_vectors * vl, sve);
                    }
                }
            }
        }
    });
}

template <typename data_t>
void shm_allgather_dim1_general(
    const data_t* /*src*/,
    int64_t B,
    int64_t H_local,
    int64_t inner_dims_numel,
    int64_t rank,
    int64_t world_size,
    uint8_t** buffers,
    kupl_shm_win_h m_recvbuf_win
) {
    // PULL peers' local H-slices into our buffer (after local write + fence).
    const int64_t H_global = H_local * world_size;
    const int64_t row = H_local * inner_dims_numel;
    const size_t bytes = static_cast<size_t>(row) * sizeof(data_t);
    data_t* local = reinterpret_cast<data_t*>(buffers[rank]);
    for (int64_t p = 0; p < world_size; ++p) {
        if (p == rank) continue;
        const data_t* peer = reinterpret_cast<const data_t*>(buffers[p]);
        for (int64_t b = 0; b < B; ++b) {
            const data_t* src_slice =
                peer + b * H_global * inner_dims_numel + p * row;
            data_t* dst_slice =
                local + b * H_global * inner_dims_numel + p * row;
            memcpy(dst_slice, src_slice, bytes);
        }
    }
    kupl_shm_fence(m_recvbuf_win);
}

template <typename data_t>
uint8_t* shm_all_gather_dim1(const data_t* src, int64_t B, int64_t H_local, int64_t inner_dims_numel,
    int64_t m_rank, int64_t m_world_size, void * m_kupl_recvbuf, kupl_shm_win_h m_recvbuf_win) {

    const int64_t world_size = m_world_size;
    const int64_t H_global = H_local * world_size;
    void* shared_buffer = m_kupl_recvbuf;

    shm_allgather_dim1_write_local<data_t>(
        src, static_cast<data_t*>(shared_buffer),
        B, H_local, H_global, inner_dims_numel, m_rank);
    kupl_shm_fence(m_recvbuf_win);

    std::vector<uint8_t*> buffers(static_cast<size_t>(world_size));
    for (int64_t i = 0; i < world_size; ++i) {
        kupl_shm_win_query(m_recvbuf_win, static_cast<int>(i), (void**)&buffers[i]);
    }

    if (world_size == 8) {
        // Die-pair optimized path (keeps previous W=8 behaviour).
        if (sizeof(data_t) == 4) {
            kutacc::shm_allgather_comm8v2_32_dim1_fp32(
                (float*)src, B, H_local, inner_dims_numel,
                shared_buffer, m_rank, buffers.data(), m_recvbuf_win);
        } else {
            kutacc::shm_allgather_comm8v2_32_dim1_bf16(
                (uint16_t*)src, B, H_local, inner_dims_numel,
                shared_buffer, m_rank, buffers.data(), m_recvbuf_win);
        }
    } else if (world_size > 1) {
        shm_allgather_dim1_general<data_t>(
            src, B, H_local, inner_dims_numel, m_rank, world_size,
            buffers.data(), m_recvbuf_win);
    }
    return buffers[m_rank];
}

template uint8_t* shm_all_gather_dim1<float>(const float*, int64_t , int64_t, int64_t, int64_t, int64_t, void *, kupl_shm_win_h);
template uint8_t* shm_all_gather_dim1<uint16_t>(const uint16_t*, int64_t , int64_t, int64_t, int64_t, int64_t, void *, kupl_shm_win_h);

}// namespace kutacc