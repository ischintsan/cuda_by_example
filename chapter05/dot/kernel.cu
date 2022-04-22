// dot

#ifdef __INTELLISENSE__

// in here put whatever is your favorite flavor of intellisense workarounds
void __syncthreads(void);

#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"

#define imin(a,b) (a<b?a:b)
#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)

// 默认启动32个Block，每个Block包含256个Thread
// 当N小于8192时，这样启动就会造成Thread的浪费，需要减少启动的Block的数量
// 当N小于或等于8192时，每个Thread处理一对矢量元素；当N大于8192时，存在Thread处理多对矢量元素
const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float* a, float* b, float* c) {
    __shared__ float cache[threadsPerBlock];  // 使用关键字__shared__声明一个变量驻留在共享内存中
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N) {
        // 一个Thread需要计算多个矢量元素的乘积
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    // cache将对应threadIdx的Thread计算后的多个矢量元素的乘积累加起来
    cache[cacheIndex] = temp;

    // 对Block中的Thread进行同步
    __syncthreads();

    // 归约(Reduction)算法求和
    // 对于归约运算来说，以下代码要求threadsPerBlock必须是2的倍数
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main() {
    float *a, *b, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float c;

    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float)));

    // 将两个矢量初始化为特殊的数列，方便之后使用平反和数列求和公式验证
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    // 在CPU上完成最终的求和运算
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];  // partial_c[i]就是第i个Block返回的元素乘积之和，一共有blocksPerGrid个Block，把它们加起来得到最终结果
    }

    // 验证结果。应该与公式计算的结果一致
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_partial_c));

    free(a);
    free(b);
    free(partial_c);

    return 0;
}