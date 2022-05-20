// basic_single_stream

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../common/book.h"
#include <stdio.h>

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

__global__ void kernel(int* a, int* b, int* c) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        int id1 = (id + 1) % 256;
        int id2 = (id + 2) % 256;
        float as = (a[id] + a[id1] + a[id2]) / 3.0f;
        float bs = (b[id] + b[id1] + b[id2]) / 3.0f;
        c[id] = (as + bs) / 2;
    }
}

int main() {
    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_ERROR(cudaGetDevice(&whichDevice));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
    // 选择一个支持设备重叠(Device Overlap)功能的设备:
    // 能够在执行一个CUDA C核函数的同时，还能在设备与主机之间执行复制操作
    if (!prop.deviceOverlap) {
        printf("Device will not handle overlaps, so no "
            "speed up from stream\n"); 
    }

    cudaEvent_t start, end;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // 初始化Stream
    cudaStream_t stream;
    HANDLE_ERROR(cudaStreamCreate(&stream));

    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;

    // 在GPU上分配内存
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    // 分配由Stream使用的Page-Locked内存
    HANDLE_ERROR(cudaHostAlloc((void**)&host_a,
                                FULL_DATA_SIZE * sizeof(int),
                                cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_b,
                                FULL_DATA_SIZE * sizeof(int),
                                cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_c,
                                FULL_DATA_SIZE * sizeof(int),
                                cudaHostAllocDefault));

    // 使用随机整数填充主机内存
    for (int i = 0; i < FULL_DATA_SIZE; i++) {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    // 在整体数据上循环，每个数据块的大小为N
    for (int offset = 0; offset < FULL_DATA_SIZE; offset +=N) {
        // 将Page-Locked Memory以异步方式复制到Device上
        // 第一次复制
        HANDLE_ERROR(cudaMemcpyAsync(dev_a, 
                                host_a + offset,			// 加上一个偏移offset
                                N * sizeof(int), 
                                cudaMemcpyHostToDevice,
                                stream));					// 在这个stream中进行复制
        // 第二次复制
        HANDLE_ERROR(cudaMemcpyAsync(dev_b,
                                host_b + offset, 
                                N * sizeof(int),
                                cudaMemcpyHostToDevice,
                                stream));

        kernel<<<N / 256, 256>>>(dev_a, dev_b, dev_c);
        
        // 第三次复制
        // 将数据从Device复制到Page-Locked Memory
        HANDLE_ERROR(cudaMemcpyAsync(host_c + offset,
                                dev_c,
                                N * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream));
    }

    // 将计算结果从页锁定内存复制到主机内存
    HANDLE_ERROR(cudaStreamSynchronize(stream));

    HANDLE_ERROR(cudaEventRecord(end, 0));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, end));

    HANDLE_ERROR(cudaFreeHost(host_a));
    HANDLE_ERROR(cudaFreeHost(host_b));
    HANDLE_ERROR(cudaFreeHost(host_c));
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));
    HANDLE_ERROR(cudaStreamDestroy(stream));  // 释放流

    return 0;
}