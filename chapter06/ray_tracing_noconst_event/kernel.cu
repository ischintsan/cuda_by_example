// ray_tracing_noconst_event

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include "../../common/book.h"
#include "../../common/cpu_bitmap.h"

#define INF 2e10f
#define rnd( x ) (x * rand() / RAND_MAX)
#define DIM 1024
#define SPHERES 20

struct Sphere
{
    float r, g, b;
    float radius;
    float x, y, z;

    __device__ float hit(float ox, float oy, float* n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

__global__ void kernel(Sphere* s, unsigned char* ptr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * gridDim.x * blockDim.x + x;
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}


int main() {
    CPUBitmap bitmap(DIM, DIM);
    unsigned char* dev_ptr;
    Sphere* dev_s;

    // 记录起始时间
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));  // 创建一个事件
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&dev_s, SPHERES * sizeof(Sphere)));

    Sphere* spheres = (Sphere*)malloc(SPHERES * sizeof(Sphere));
    for (int i = 0; i < SPHERES; i++) {
        spheres[i].r = rnd(1.0f);
        spheres[i].g = rnd(1.0f);
        spheres[i].b = rnd(1.0f);
        spheres[i].x = rnd(1000.0f) - 500;
        spheres[i].y = rnd(1000.0f) - 500;
        spheres[i].z = rnd(1000.0f) - 500;
        spheres[i].radius = rnd(100.0f) + 20;
    }
    HANDLE_ERROR(cudaMemcpy(dev_s, spheres, SPHERES * sizeof(Sphere), cudaMemcpyHostToDevice));
    free(spheres);

    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel << <blocks, threads >> > (dev_s, dev_ptr);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_ptr,
        bitmap.image_size(),
        cudaMemcpyDeviceToHost));

    // 获取结束时间，并显示计时结果
    HANDLE_ERROR(cudaEventRecord(stop, 0));  // 记录事件
    HANDLE_ERROR(cudaEventSynchronize(stop));  // 阻塞后面的语句，直到GPU执行到达stop事件

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate:  %3.1f ms\n", elapsedTime);

    // 销毁事件，类似对malloc()分配的内存进行free()
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    bitmap.display_and_exit();

    HANDLE_ERROR(cudaFree(dev_s));
    HANDLE_ERROR(cudaFree(dev_ptr));

    return 0;
}