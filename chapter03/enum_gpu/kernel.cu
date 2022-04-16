// enum_gpu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"

int main() {
    cudaDeviceProp prop;

    int count;
    // 获取CUDA设备的数量
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        // 设备序号
        printf("   --- General Information for device %d ---\n", i);
        // 标识设备的ASCII字符串
        printf("Name:  %s\n", prop.name);
        // 设备的算力
        printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
        // 时钟频率(单位: kHz)
        printf("Clock rate:  %d\n", prop.clockRate);
        // 设备是否可以同时复制内存并执行内核
        printf("Device copy overlap:  ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        // 指定内核是否有运行时间限制
        printf("Kernel execution timeout :  ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");

        printf("   --- Memory Information for device %d ---\n", i);
        // 设备上可用的全局内存(单位: byte)
        printf("Total global mem:  %ld\n", prop.totalGlobalMem);
        // 设备上可用的恒定内存(单位: byte)
        printf("Total constant Mem:  %ld\n", prop.totalConstMem);
        // 在内存复制的允许的最大间距(单位: byte)
        printf("Max mem pitch:  %ld\n", prop.memPitch);
        // 纹理的对齐要求
        printf("Texture Alignment:  %ld\n", prop.textureAlignment);

        printf("   --- MP Information for device %d ---\n", i);
        // 设备上的多处理器数量
        printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
        // 每个线程块(Block)可用的共享内存(单位: byte)
        printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
        // 每个线程块(Block)可用32位寄存器
        printf("Registers per mp:  %d\n", prop.regsPerBlock);
        // 在一个线程束(Warp)中包含的线程数量
        printf("Threads in warp:  %d\n", prop.warpSize);
        // 每一个线程块(Block)可包含的最大线程数量
        printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
        // 在多维线程块(Block)数组中，每一维可以包含的线程块数量
        printf("Max thread dimensions:  (%d, %d, %d)\n",
            prop.maxThreadsDim[0],
            prop.maxThreadsDim[1],
            prop.maxThreadsDim[2]);
        // 在每一个线程格(Grid)中，每一维可以包含的线程块(Block)数量
        printf("Max grid dimensions:  (%d, %d, %d)\n",
            prop.maxGridSize[0],
            prop.maxGridSize[1],
            prop.maxGridSize[2]);

        printf("\n");
    }

    return 0;
}
