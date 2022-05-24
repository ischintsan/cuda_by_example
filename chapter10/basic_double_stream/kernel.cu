// basic_double_stream

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

    // 初始化流
    cudaStream_t stream0, stream1;
    HANDLE_ERROR(cudaStreamCreate(&stream0));
    HANDLE_ERROR(cudaStreamCreate(&stream1));

    int* host_a, * host_b, * host_c;
    int* dev_a0, * dev_b0, * dev_c0;  // 为stream0分配的GPU内存
    int* dev_a1, * dev_b1, * dev_c1;  // 为stream1分配的GPU内存

    // 在GPU上分配内存
    HANDLE_ERROR(cudaMalloc((void**)&dev_a0, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b0, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c0, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_a1, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b1, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c1, N * sizeof(int)));

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
    for (int offset = 0; offset < FULL_DATA_SIZE; offset+=N*2) {  // 有2个流，因此每次循环后偏移2N
        // 将Page-Locked Memory以异步方式复制到Device上
        // 第一次复制(stream0)
        HANDLE_ERROR(cudaMemcpyAsync(dev_a0,
                                     host_a + offset,
                                     N * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream0));
        // 第二次复制(stream0)
        HANDLE_ERROR(cudaMemcpyAsync(dev_b0,
                                     host_b + offset,
                                     N * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream0));
        // 执行核函数
        kernel<<<N / 256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);

        // 将数据从Device复制回Page-Locked Memory
        HANDLE_ERROR(cudaMemcpyAsync(host_c + offset,
                                     dev_c0,
                                     N * sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     stream0));

        // 第三次复制(stream1)
        HANDLE_ERROR(cudaMemcpyAsync(dev_a1,
                                     host_a + offset + N,
                                     N * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream1));
       
        // 第四次复制(stream1)
        HANDLE_ERROR(cudaMemcpyAsync(dev_b1,
                                     host_b + offset + N,
                                     N * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream1));

        // 执行核函数
        kernel<<<N / 256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);

        // 将数据从Device复制回Page-Locked Memory
        HANDLE_ERROR(cudaMemcpyAsync(host_c + offset + N,
                                     dev_c1,
                                     N * sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     stream1));
    }
    // 对流进行同步
    HANDLE_ERROR(cudaStreamSynchronize(stream0));
    HANDLE_ERROR(cudaStreamSynchronize(stream1));

    HANDLE_ERROR(cudaEventRecord(end, 0));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, end));
    printf("Time taken: %3.1f ms\n", elapsedTime);

    HANDLE_ERROR(cudaFreeHost(host_a));
    HANDLE_ERROR(cudaFreeHost(host_b));
    HANDLE_ERROR(cudaFreeHost(host_c));
    HANDLE_ERROR(cudaFree(dev_a0));
    HANDLE_ERROR(cudaFree(dev_b0));
    HANDLE_ERROR(cudaFree(dev_c0));
    HANDLE_ERROR(cudaFree(dev_a1));
    HANDLE_ERROR(cudaFree(dev_b1));
    HANDLE_ERROR(cudaFree(dev_c1));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));
    HANDLE_ERROR(cudaStreamDestroy(stream0));  // 释放流
    HANDLE_ERROR(cudaStreamDestroy(stream1));

	return 0;
}