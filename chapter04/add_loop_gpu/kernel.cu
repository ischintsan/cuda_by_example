// add_loop_gpu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"

#define N 10

__global__ void add(int* a, int* b, int* c) {
    int tid = blockIdx.x;  // 当前执行着设备代码的线程块(Block)的索引,第一个线程块的索引为0
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int a[N], b[N], c[N];
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

    // <<<b, t>>> b:设备在执行核函数时使用的并行线程块(Block)数量。t:CUDA Runtime在每个线程块中创建的线程数量
    // N个线程块 * 1个线程/线程块 = N个并行线程
    // 当启动核函数时，我们将并行线程块(Block)的数量指定为N。这个并行线程块集合也称为一个线程格(Grid),
    // 这是告诉CUDA Runtime，我们想要一个一维的线程格，其中包含N个线程块。
    add<<<N, 1 >>>(dev_a, dev_b, dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    return 0;
}