// copy_timed

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../common/book.h"

#include <stdio.h>

#define SIZE    (64*1024*1024)

float cuda_malloc_test(int size, bool up) {
    cudaEvent_t start, end;
    int *a, *dev_a;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));
    
    // 分配主机缓冲区和GPU缓冲区
    a = (int*)malloc(size * sizeof(int));  // 使用标准C函数malloc()来分配可分页主机内存
    HANDLE_NULL(a);
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(int)));

    HANDLE_ERROR(cudaEventRecord(start, 0));
    // 执行100次复制操作，并由参数up指定复制方向
    for (int i = 0; i < 100; i++) {
        if (up) {
            // cudaMemcpyHostToDevice
            HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
        }else{
            // cudaMemcpyDeviceToHost
            HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost));
        }
    }
    HANDLE_ERROR(cudaEventRecord(end, 0));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, end));

    free(a);
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));

    return elapsedTime;
}

float cuda_host_alloc_test(int size, bool up) {
    cudaEvent_t start, end;
    int* a, * dev_a;
    float elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));

    // 分配主机缓冲区和GPU缓冲区
    HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(int), cudaHostAllocDefault));  // 使用cudaHostAlloc()来分配固定内存
    HANDLE_NULL(a);
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(int)));

    HANDLE_ERROR(cudaEventRecord(start, 0));
    // 执行100次复制操作，并由参数up指定复制方向
    for (int i = 0; i < 100; i++) {
        if (up) {
            // cudaMemcpyHostToDevice
            HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
        }
        else {
            // cudaMemcpyDeviceToHost
            HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost));
        }
    }
    HANDLE_ERROR(cudaEventRecord(end, 0));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, end));

    HANDLE_ERROR(cudaFreeHost(a));  // 使用cudaFreeHost()释放由cudaHostAlloc()分配的内存
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));

    return elapsedTime;
}

int main() {
    float elapsedTime;
    float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;

    // 测试从Host到Device的复制性能(使用malloc分配的内存)
    elapsedTime = cuda_malloc_test(SIZE, true);
    printf("Time using malloc: %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy up: %3.1f\n", MB / (elapsedTime / 1000));

    // 测试从Device到Host的复制性能(使用malloc分配的内存)
    elapsedTime = cuda_malloc_test(SIZE, false);
    printf("Time using malloc: %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy down: %3.1f\n", MB / (elapsedTime / 1000));

    // 测试从Host到Device的复制性能(使用cudaHostAlloc分配的内存)
    elapsedTime = cuda_host_alloc_test(SIZE, true);
    printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy up: %3.1f\n", MB / (elapsedTime / 1000));

    // 测试从Device到Host的复制性能(使用cudaHostAlloc分配的内存)
    elapsedTime = cuda_host_alloc_test(SIZE, false);
    printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
    printf("\tMB/s during copy down: %3.1f\n", MB / (elapsedTime / 1000));


    return 0;
}