// shared_bitmap

#ifdef __INTELLISENSE__

// in here put whatever is your favorite flavor of intellisense workarounds
void __syncthreads(void);

#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "math.h"
#include "../../common/book.h"
#include "../../common/cpu_bitmap.h"

#define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel(unsigned char* ptr) {
    // 将threadIdx/blockIdx映射到像素位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * gridDim.x * blockDim.x + x;

    // 声明一个共享缓冲区，因为要让CUDA Runtime启动的block有(16,16)个线程，所有缓冲区大小也设置为16*16，
    // 让每个线程在该缓冲区中都有一个对应的位置
    __shared__ float shared[16][16];

    // 现在计算这个位置上的的值
    const float period = 128.0f;
    
    shared[threadIdx.x][threadIdx.y] =
        255 * (sinf(x * 2.0f * PI / period) + 1.0f) *
        (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;

    __syncthreads();
    // 最后，把这些值保存回像素，保留x和y的次序
    ptr[offset * 4 + 0] = 0;
    // 注意这里，当索引为(threadIdx.x, threadIdx.y)的Thread完成对缓冲区shared的写入后，要在这里对
    // shared[15 - threadIdx.x][15 - threadIdx.y]进行读取时，
    // 索引为(15 - threadIdx.x, 15 - threadIdx.y)的Thread可能还没完成对缓冲区shared的写入，
    // 因此需要在之前加上__syncthreads();
    ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main() {
    CPUBitmap bitmap(DIM, DIM);
    unsigned char* dev_ptr;

    HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, bitmap.image_size()));

    dim3 threads(16, 16);
    dim3 blocks(DIM / 16, DIM / 16);
    kernel<<<blocks, threads>>>(dev_ptr);
    
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_ptr, bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    HANDLE_ERROR(cudaFree(dev_ptr));

    return 0;
}