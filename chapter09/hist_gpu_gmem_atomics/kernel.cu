// hist_gpu_gmem_atomics

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../common/book.h"

#include <stdio.h>

#define SIZE (100*1024*1024)

__global__ void histo_kernel(unsigned char* buffer,
                             long size, 
                             unsigned int* histo) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (tid < size) {
        // 表示在CUDA C中使用原子操作的方式。函数调用atomicAdd(address, val)
        // 将生成一个原子的序列操作，这个操作序列包括读取地址address处的值，将val增加到这个值上，
        // 以及将结果保存回地址address。底层硬件将确保当执行这些操作时，
        // 其他任何线程都不会读取或写入地址address上的值，这样就能确保得到预计的结果。
        atomicAdd(&(histo[buffer[tid]]), 1);
        tid += stride;
    }
}

int main() {
    // 随机生成100MB的随机数据
    unsigned char* buffer = (unsigned char*)big_random_block(SIZE);

    // 初始化计时事件
    cudaEvent_t start, end;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // 在GPU上为文件的数据分配内存
    unsigned char* dev_buffer;
    unsigned int* dev_histo;
    HANDLE_ERROR(cudaMalloc((void**)&dev_buffer, SIZE * sizeof(char)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_histo, 256 * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE * sizeof(char), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemset(dev_histo, 0, 256 * sizeof(int)));

    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount * 2; // 将Block的数量设置为GPU中处理器数量的2倍
    histo_kernel<<<blocks, 256>>>(dev_buffer, SIZE, dev_histo);

    unsigned int histo[256];
    HANDLE_ERROR(cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    // 得到停止时间并显示计时结果
    HANDLE_ERROR(cudaEventRecord(end, 0));
    HANDLE_ERROR(cudaEventSynchronize(end));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, end));
    printf("Time to generate:  %3.1f ms\n", elapsedTime);

    // 验证直方图的所有元素加起来是否等于正确的值(应该等于SIZE)
    long histoCount = 0;
    for (int i = 0; i < 256; i++) {
        histoCount += histo[i];
    }
    printf("Histogram Sum:  %ld\n", histoCount);

    // 验证与CPU得到的是相同的计数值
    for (int i = 0; i < SIZE; i++) {
        histo[buffer[i]]--;
    }
    for (int i = 0; i < 256; i++) {
        if (histo[i] != 0) {
            printf("Failure at %d!  Off by %d\n", i, histo[i]);
        }
    }

    // 释放事件、内存
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));
    HANDLE_ERROR(cudaFree(dev_buffer));
    HANDLE_ERROR(cudaFree(dev_histo));
    free(buffer);

    return 0;
}