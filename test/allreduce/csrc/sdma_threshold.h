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
#include <unistd.h>
#include <cmath>
#include <cerrno>
#include <fcntl.h>
#include <cctype>
#include <termios.h>
#include <sys/types.h>
#include <sys/mman.h>

#define MAP_SIZE 4096UL  			// 映射的内存区大小（一般为一个页框大小）
#define MAP_MASK (MAP_SIZE - 1)   	// MAP_MASK = 0XFFF

#define ADDR_OFFSET 1
#define READ_MAX_LEN 0xF00000U      // 15M
#define WRITE_MAX_LEN 1U            // 1B

//*****************************************************************************
/**
* @brief 直接写入到内存实际的物理地址。
* @details 通过 mmap 映射关系，找到对应的内存实际物理地址对应的虚拟地址，然后写入数据。
* 写入长度，每次最低4字节
* @param[in] writeAddr, unsigned long, 需要操作的物理地址。
* @param[in] buf，unsigned char *, 需要写入的数据。
* @param[in] len，unsigned long, 需要写入的长度，1字节为单位。
* @return ret, int, 如果发送成功，返回0，发送失败，返回-1。
*/
inline int devMemFd = -1;
inline void *map_base = nullptr;
inline void *map_base2 = nullptr;
inline void *map_base3 = nullptr;
inline void *map_base4 = nullptr;

#define SDMA_THRH_4 0x00001404
#define SDMA_THRH_3 0x00001403
#define SDMA_CTL_ADDR1 0x40cfe077c
#define SDMA_CTL_ADDR2 0x60cfe077c
#define SDMA_CTL_ADDR3 0x8040cfe077c
#define SDMA_CTL_ADDR4 0x8060cfe077c

inline int SdmaCtlThredInit()
{
    int fd;
    unsigned long addr = SDMA_CTL_ADDR1;

    if ((fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) {
        fprintf(stderr, "Error (%d) [%s]\n", errno, strerror(errno));
        return -1;
    }
	/* Map one page */ // 将内核空间映射到用户空间
    map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, addr & ~MAP_MASK);
    if (map_base == (void *) -1) {
        fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno));
        close(fd);
        return -1;
    }
    addr = SDMA_CTL_ADDR2;
    map_base2 = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, addr & ~MAP_MASK);
    if (map_base2 == (void *) -1) {
        fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno));
        close(fd);
        return -1;
    }
    addr = SDMA_CTL_ADDR3;
    map_base3 = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, addr & ~MAP_MASK);
    if (map_base == (void *) -1) {
        fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno));
        close(fd);
        return -1;
    }
    addr = SDMA_CTL_ADDR4;
    map_base4 = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, addr & ~MAP_MASK);
    if (map_base4 == (void *) -1) {
        fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, __FILE__, errno, strerror(errno));
        close(fd);
        return -1;
    }
    devMemFd = fd;
    return 0;
}

inline int DevmemWrite(unsigned long baseAddr, unsigned long writeAddr, unsigned char* buf, unsigned long len)
{
    unsigned char *virt_addr = (unsigned char*)(baseAddr + (writeAddr & MAP_MASK));
    memcpy(virt_addr, buf, len);
    return 0;
}

inline void SetSdmaThreshold(unsigned int Id, int val)
{
    unsigned int initVal = 0x00001400;
    unsigned int setVal = initVal + val;
    if (Id == 0) {
        DevmemWrite((unsigned long)map_base, SDMA_CTL_ADDR1, (unsigned char*)&setVal, 32);
    } else if (Id == 1) {
        DevmemWrite((unsigned long)map_base3, SDMA_CTL_ADDR3, (unsigned char*)&setVal, 32);
    } else if (Id == 2) {
        DevmemWrite((unsigned long)map_base2, SDMA_CTL_ADDR2, (unsigned char*)&setVal, 32);
    } else {
        DevmemWrite((unsigned long)map_base4, SDMA_CTL_ADDR4, (unsigned char*)&setVal, 32);
    }
}