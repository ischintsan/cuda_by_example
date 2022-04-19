// add_loop_very_long_blocks

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"

#define N (65536*1024)

__global__ void add(int* a, int* b, int* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid +=  gridDim.x * blockDim.x;  // 步长：启动的Thread的数量 = 启动的Block数量(gridDim.x) * 每个Block包含的Thread的数量(blockDim.x)
                                         // 让每个Thread处理 N / (gridDim.x * blockDim.x)个数据
    }
}

int main() {
    int* a, * b, * c;
    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));

    int* dev_a, * dev_b, * dev_c;
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    // 启动128个Block，每个Block含有128个线程
    // 让每个Thread处理 N / 16384个数据
    add<<<128, 128>>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    free(a);
    free(b);
    free(c);

    return 0;
}