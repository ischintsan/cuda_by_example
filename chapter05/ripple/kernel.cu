
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"
#include "../../common/cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel(unsigned char* ptr, int ticks) {
    // 将threadIdx/blockIdx映射到像素位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = y * blockDim.x * gridDim.x + x;

    // 下面的代码与动画有关，不用管了
    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrtf(fx * fx + fy * fy);
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                         cos(d / 10.0f - ticks / 7.0f) /
                                         (d / 10.0f + 1.0f));
    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;

}

struct DataBlock
{
    unsigned char* dev_bitmap;
    CPUAnimBitmap* bitmap;
};

void generate_frame(DataBlock* d, int ticks) {
    // (DIM/16, DIM/16)个Block组成一个Grid
    // 每个Block中有(16, 16)个Thread
    // 所以一共有(DIM, DIM)个Thread，对应DIM*DIM尺寸的图像，每一个像素由一个Thread处理
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);

    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(),
                            d->dev_bitmap,
                            d->bitmap->image_size(),
                            cudaMemcpyDeviceToHost));
}

// 释放在GPU上分配的显存
void cleanup(DataBlock* d) {
    HANDLE_ERROR(cudaFree(d->dev_bitmap));
}

int main() {
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));

    // 每次生成一帧图像，调用一次generate_frame，之后将分配的显存释放掉
    bitmap.anim_and_exit((void(*)(void*, int))generate_frame,
                         (void(*)(void*))cleanup);

    return 0;
}