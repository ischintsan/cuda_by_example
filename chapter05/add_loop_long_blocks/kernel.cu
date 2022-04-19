// add_loop_long_blocks

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"

#define N (33*1024)  // 这么大的N，如果只用一个Block是做不了的

__global__ void add(int* a, int* b, int* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

// 当N很大时要修改项目属性->配置属性->链接器->系统->堆栈保留大小(单位: byte)
// 或者分配在全局区或堆区
// int a[N], b[N], c[N];  // 分配在全局区
int main() {
    //int a[N], b[N], c[N];  // 分配在栈区
    int* a, * b, * c;
    a = (int*)malloc(N * sizeof(int));  // 分配在堆区，最后记得要释放
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

    // (127 + N) / 128 : N/128向上取整
    // 每个Block有128个Thread
    // 注意( N + 127 ) / 128不能超过maxGridSize的限制
    add<<<(127 + N) / 128, 128>>>(dev_a, dev_b, dev_c);  

    HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    free(a);  // 释放之前分配的堆区内存
    free(b);
    free(c);

    return 0;
}