/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
void test_all_gather_dim1(int batch, int bucket, int size)
{
    std::cout << "test all_gather_dim1 in af3 of " << batch << " batch, " << bucket << " buckets, " << size << " size " << std::endl;
    std::mt19937 rnd(time(0));
    int elem_num = batch * bucket * size; 
    size_t elem_size = sizeof(scalar_t);
    scalar_t *sendbuf = (scalar_t *)malloc( elem_num * sizeof(scalar_t));
    uint8_t* recvbuf;

    for(int i = 0; i < elem_num; i++){
        sendbuf[i] = rnd() * 1.0 / rnd.max();
    }
    if(elem_size == 2){
        const uint16_t * sendbuf_u16 = reinterpret_cast<const uint16_t*>(sendbuf);
        recvbuf = kutacc::shm_all_gather_dim1(sendbuf_u16, batch, bucket, size, global_rank, (void*)local_buffer, kupl_socket_win);
    }else{
        recvbuf = kutacc::shm_all_gather_dim1(sendbuf, batch, bucket, size, global_rank, (void*)local_buffer, kupl_socket_win);
    }
    scalar_t * recv_data = (scalar_t *)recvbuf;
    float max_diff = 0.0;
    for(int i = 0; i < batch; i++){
        for(int j = 0; j < bucket; j++){
            for(int k = 0; k < size; k++){
                int send_idx = i*bucket*size + j*size + k;
                int recv_idx = i*bucket*global_size*size + (global_rank*bucket + j)*size + k;
                if(elem_size == 2){
                    uint16_t recv_u16 = *reinterpret_cast<const uint16_t*>(&recv_data[recv_idx]);
                    uint32_t recv_u32 = (uint32_t)recv_u16 << 16;
                    float recv_fp32 =  *reinterpret_cast<float*>(&recv_u32);
                    uint16_t send_u16 = *reinterpret_cast<const uint16_t*>(&sendbuf[send_idx]);
                    uint32_t send_u32 = (uint32_t)send_u16 << 16;
                    float send_fp32 =  *reinterpret_cast<float*>(&send_u32);
                    max_diff = std::max(max_diff, abs(recv_fp32 - send_fp32));
                }else{
                    max_diff = std::max(max_diff, abs((float)recv_data[recv_idx] - (float)sendbuf[send_idx]));
                }
            }
        }
    }
    if(max_diff > 0.1){
        std::cerr << "maxdiff is " << max_diff << ", too much diff between send and recv " << "\n";
        exit(1);
    }

    uint64_t start = kutacc_now_ns();
    for(int i = 0; i < test_times; i++){
        if(elem_size == 2){
            const uint16_t * sendbuf_u16 = reinterpret_cast<const uint16_t*>(sendbuf);
            recvbuf = kutacc::shm_all_gather_dim1(sendbuf_u16, batch, bucket, size, global_rank, (void*)local_buffer, kupl_socket_win);
        }else{
            recvbuf = kutacc::shm_all_gather_dim1(sendbuf, batch, bucket, size, global_rank, (void*)local_buffer, kupl_socket_win);
        }
    }
    uint64_t end = kutacc_now_ns();
    double sum_time = end - start;

    std::cout << "rank " << global_rank << ", elem_size = " << elem_size << ", elem_num = " << elem_num << ", max_diff = " << max_diff << ", avg_time = " << sum_time / test_times / 1000 << " us. "<< std::endl;
    free(sendbuf);
}

int main(int argc, char *argv[])
{
    kutacc::global_parallel_launch([&] {
        MPI_Init(&argc, &argv);
        shm_init();
        kutacc_time_init();
       // cases in AlphaFold3 inference
        test_all_gather_dim1<float>(64,8,64);
        test_all_gather_dim1<float>(128,16,64);
        test_all_gather_dim1<float>(320,40,64);
        test_all_gather_dim1<float>(448,56,64);
        test_all_gather_dim1<float>(640,80,64);
        test_all_gather_dim1<float>(768,96,64);
        test_all_gather_dim1<bfloat16_t>(64,8,64);
        test_all_gather_dim1<bfloat16_t>(128,16,64);
        test_all_gather_dim1<bfloat16_t>(320,40,64);
        test_all_gather_dim1<bfloat16_t>(448,56,64);
        test_all_gather_dim1<bfloat16_t>(640,80,64);
        test_all_gather_dim1<bfloat16_t>(768,96,64);
        shm_finalize();
        MPI_Finalize();
    });
    return 0;
}