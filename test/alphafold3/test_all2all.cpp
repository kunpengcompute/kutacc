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
#include <iostream>
#include <sys/time.h>
#include <random>

#include "shm_initialize.h"

template <typename scalar_t>
void test_all2all(int batch, int bucket, int size, int splited_dim)
{
    FLASH_ASSERT(splited_dim == 0 || splited_dim == 1);
    std::cout << "test all2all in af3 of " << batch << " batch, " << bucket << " buckets, " << size << " size, " << " splited_dim " << splited_dim << std::endl;
    // Rank-dependent data makes source-rank routing errors observable.
    std::mt19937 rnd(42 + global_rank);

    int elem_num = batch * bucket * size; 
    size_t elem_size = sizeof(scalar_t);
    scalar_t *sendbuf = (scalar_t *)malloc( elem_num * sizeof(scalar_t));
    scalar_t* recvbuf = (scalar_t *)malloc( elem_num * sizeof(scalar_t));;
    scalar_t* allgather_data = (scalar_t*)malloc(
        static_cast<size_t>(global_size) * elem_num * sizeof(scalar_t));
    for(int i = 0; i < elem_num; i++){
        sendbuf[i] = rnd() * 1.0 / rnd.max();
    }
    int recv_dim0, recv_dim1, gather_dim1;
    if(splited_dim == 0){
        recv_dim0 = batch * global_size;
        recv_dim1 = bucket / global_size;
        gather_dim1 = bucket;
    }else{
        recv_dim0 = batch / global_size;
        gather_dim1 = bucket * global_size;
        recv_dim1 = bucket * global_size;
    }
    if(elem_size == 2){
        const uint16_t * sendbuf_u16 = reinterpret_cast<const uint16_t*>(sendbuf);
        uint16_t * recvbuf_u16 = reinterpret_cast<uint16_t*>(recvbuf);
        if(splited_dim == 0){ // [batch,bucket,size] -> [batch*8,bucket/8,size]
            MPI_Allgather(sendbuf, elem_num * elem_size, MPI_BYTE,
                          allgather_data, elem_num * elem_size, MPI_BYTE, MPI_COMM_WORLD);
            kutacc::shm_all2all(sendbuf_u16, recvbuf_u16, recv_dim0, bucket, size, batch, bucket, 
            bucket*size, size, recv_dim1*size, size, global_rank, global_size, MAX_BUFFER_SIZE, (void*)local_buffer, kupl_socket_win);
        }else{ // [batch,bucket,size] -> [batch/8,bucket*8,size]
            MPI_Allgather(sendbuf, elem_num * elem_size, MPI_BYTE,
                          allgather_data, elem_num * elem_size, MPI_BYTE, MPI_COMM_WORLD);
            kutacc::shm_all2all(sendbuf_u16, recvbuf_u16, batch, recv_dim1, size, batch, bucket, 
            bucket*size, size, recv_dim1*size, size, global_rank, global_size, MAX_BUFFER_SIZE, (void*)local_buffer, kupl_socket_win);
       }
    }else{
        if(splited_dim == 0){ // [batch,bucket,size] -> [batch*8,bucket/8,size]
            MPI_Allgather(sendbuf, elem_num * elem_size, MPI_BYTE,
                          allgather_data, elem_num * elem_size, MPI_BYTE, MPI_COMM_WORLD);
            kutacc::shm_all2all((float*)sendbuf, (float*)recvbuf, recv_dim0, bucket, size, batch, bucket, 
            bucket*size, size, recv_dim1*size, size, global_rank, global_size, MAX_BUFFER_SIZE, (void*)local_buffer, kupl_socket_win);
        }else{ // [batch,bucket,size] -> [batch/8,bucket*8,size]
            MPI_Allgather(sendbuf, elem_num * elem_size, MPI_BYTE,
                          allgather_data, elem_num * elem_size, MPI_BYTE, MPI_COMM_WORLD);
            kutacc::shm_all2all((float*)sendbuf, (float*)recvbuf, batch, recv_dim1, size, batch, bucket, 
            bucket*size, size, recv_dim1*size, size, global_rank, global_size, MAX_BUFFER_SIZE, (void*)local_buffer, kupl_socket_win);
       }
    }
    float max_diff = 0.0f;
    for(int i = 0; i < recv_dim0; i++){
        for(int j = 0; j < recv_dim1; j++){
            for(int k = 0; k < size; k++){
                int allgather_idx;
                if (splited_dim == 0) {
                    allgather_idx = i * gather_dim1 * size
                                  + (global_rank * recv_dim1 + j) * size + k;
                } else {
                    const int src_rank = j / bucket;
                    const int src_col = j % bucket;
                    const int src_row = global_rank * recv_dim0 + i;
                    allgather_idx = src_rank * elem_num
                                  + src_row * bucket * size
                                  + src_col * size + k;
                }
                int recv_idx = i*recv_dim1*size + j*size + k;
                if(elem_size == 2){
                    uint16_t recv_u16 = *reinterpret_cast<const uint16_t*>(&recvbuf[recv_idx]);
                    uint32_t recv_u32 = (uint32_t)recv_u16 << 16;
                    float recv_fp32 =  *reinterpret_cast<float*>(&recv_u32);
                    uint16_t allgather_u16 = *reinterpret_cast<const uint16_t*>(&allgather_data[allgather_idx]);
                    uint32_t allgather_u32 = (uint32_t)allgather_u16 << 16;
                    float allgather_fp32 =  *reinterpret_cast<float*>(&allgather_u32);
                    max_diff = std::max(max_diff, abs(recv_fp32 - allgather_fp32));
                }else{
                    max_diff = std::max(max_diff, abs((float)recvbuf[recv_idx] - (float)allgather_data[allgather_idx]));
                }
            }
        }
    }

    if (max_diff != 0.0f) {
        std::cerr << "rank " << global_rank
                  << ": all2all correctness check failed, max_diff = "
                  << max_diff << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    uint64_t start = kutacc_now_ns();
    for(int i = 0; i < test_times; i++){
        if(elem_size == 2){
            const uint16_t * sendbuf_u16 = reinterpret_cast<const uint16_t*>(sendbuf);
            uint16_t * recvbuf_u16 = reinterpret_cast<uint16_t*>(recvbuf);
            if(splited_dim == 0){ // [batch,bucket,size] -> [batch*8,bucket/8,size]
                kutacc::shm_all2all(sendbuf_u16, recvbuf_u16, recv_dim0, bucket, size, batch, bucket, 
                bucket*size, size, recv_dim1*size, size, global_rank, global_size, MAX_BUFFER_SIZE, (void*)local_buffer, kupl_socket_win);
            }else{ // [batch,bucket,size] -> [batch/8,bucket*8,size]
                kutacc::shm_all2all(sendbuf_u16, recvbuf_u16, batch, recv_dim1, size, batch, bucket, 
                bucket*size, size, recv_dim1*size, size, global_rank, global_size, MAX_BUFFER_SIZE, (void*)local_buffer, kupl_socket_win);
            }
        }else{
            if(splited_dim == 0){ // [batch,bucket,size] -> [batch*8,bucket/8,size]
                kutacc::shm_all2all((float*)sendbuf, (float*)recvbuf, recv_dim0, bucket, size, batch, bucket, 
                bucket*size, size, recv_dim1*size, size, global_rank, global_size, MAX_BUFFER_SIZE, (void*)local_buffer, kupl_socket_win);
            }else{ // [batch,bucket,size] -> [batch/8,bucket*8,size]
                kutacc::shm_all2all((float*)sendbuf, (float*)recvbuf, batch, recv_dim1, size, batch, bucket, 
                bucket*size, size, recv_dim1*size, size, global_rank, global_size, MAX_BUFFER_SIZE, (void*)local_buffer, kupl_socket_win);
            }
        }
    }
    uint64_t end = kutacc_now_ns();
    double sum_time = end - start;

    std::cout << "rank " << global_rank << ", elem_size = " << elem_size << ", elem_num = " << elem_num << ", max_diff = " << max_diff << ", avg_time = " << sum_time / test_times / 1000 << " us. "<< std::endl;
    free(sendbuf);
    free(recvbuf);
    free(allgather_data);
}

int main(int argc, char *argv[])
{
    kutacc::global_parallel_launch([&] {
        MPI_Init(&argc, &argv);
        // This test only exercises intra-node KUPL shared memory. Initializing
        // the RDMA transport makes it depend on an otherwise unused NIC.
        shm_init(false);
        kutacc_time_init();
       // cases in AlphaFold3 inference
        test_all2all<float>(64,8,64,1);
        test_all2all<float>(128,16,64,1);
        test_all2all<float>(320,40,64,1);
        test_all2all<float>(448,56,64,1);
        test_all2all<float>(640,80,64,1);
        test_all2all<float>(768,96,64,1);
        test_all2all<float>(8,64,64,0);
        test_all2all<float>(16,128,64,0);
        test_all2all<float>(40,320,64,0);
        test_all2all<float>(56,448,64,0);
        test_all2all<float>(80,640,64,0);
        test_all2all<float>(96,768,64,0);
        test_all2all<bfloat16_t>(64,8,64,1);
        test_all2all<bfloat16_t>(128,16,64,1);
        test_all2all<bfloat16_t>(320,40,64,1);
        test_all2all<bfloat16_t>(448,56,64,1);
        test_all2all<bfloat16_t>(640,80,64,1);
        test_all2all<bfloat16_t>(768,96,64,1);
        test_all2all<bfloat16_t>(8,64,64,0);
        test_all2all<bfloat16_t>(16,128,64,0);
        test_all2all<bfloat16_t>(40,320,64,0);
        test_all2all<bfloat16_t>(56,448,64,0);
        test_all2all<bfloat16_t>(80,640,64,0);
        test_all2all<bfloat16_t>(96,768,64,0);
        shm_finalize();
        MPI_Finalize();
    });
    return 0;
}
