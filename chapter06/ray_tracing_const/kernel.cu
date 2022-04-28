// ray_tracing_const

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
    float r, g, b;  //	球面的颜色
    float radius;   // 球面半径
    float x, y, z;  // 球面的中心坐标(x,y,z)

    // 对于来自(ox,oy)处的光线，计算是否会与这个球面相交，
    // 如果相交，计算从相机到光线命中球面处的距离
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

// 常量内存，需要静态分配
__constant__ Sphere dev_s[SPHERES];

__global__ void kernel(unsigned char* ptr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * gridDim.x * blockDim.x + x;

    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = dev_s[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = dev_s[i].r * fscale;
            g = dev_s[i].g * fscale;
            b = dev_s[i].b * fscale;
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


    HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, bitmap.image_size()));

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
    // 使用特殊的内存拷贝函数cudaMemcpyToSymbol()将主机内存拷贝到GPU，使用方法与cudaMemcpy类似
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_s, spheres, SPHERES * sizeof(Sphere)));
    free(spheres);

    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<blocks, threads>>>(dev_ptr);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_ptr,
        bitmap.image_size(),
        cudaMemcpyDeviceToHost));
    bitmap.display_and_exit();

    //HANDLE_ERROR(cudaFree(dev_s)); 不需要使用cudaFree()对常量内存dev_s进行释放
    HANDLE_ERROR(cudaFree(dev_ptr));

    return 0;
}