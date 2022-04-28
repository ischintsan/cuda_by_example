// ray_tracing_noconst

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
        // 判断直线与球面相交的情况
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

__global__ void kernel(Sphere* s, unsigned char* ptr) {
    // 将threadIdx/blockIdx映射到像素位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * gridDim.x * blockDim.x + x;
    // 将图像坐标(x, y)偏移DIM/2，这样z轴将穿过图像的中心
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    // 每条光线都需要判断与球面相交的情况，使用迭代的方式调用hit方法判断
    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = s[i].hit(ox, oy, &n);
        // 如果光线命中了当前当前的球面，那么接着判断命中位置与相机之间的距离是否比上一次命中的距离更加解决。
        // 如果更加接近，那么将这个距离保存为新的最接近球面，保存这个球面的rgb颜色值，
        // 当这个循环结束时，当前线程就会知道与相机最接近的球面的颜色。
        // 如果没有命中，则rgb为初始值(0,0,0)
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

    // 在GPU上分配内存以计算输出位图
    HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, bitmap.image_size()));
    // 为Sphere数据集分配内存
    HANDLE_ERROR(cudaMalloc((void**)&dev_s, SPHERES * sizeof(Sphere)));

    // 分配临时内存，对其初始化，并复制到GPU上的内存，然后再释放临时内存
    Sphere* spheres = (Sphere*)malloc(SPHERES * sizeof(Sphere));
    // 生成SPHERES个随机球面
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
    free(spheres);  // 复制到GPU后就可以释放临时缓冲区了

    // 从球面数据中生成一张bitmap
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<blocks, threads>>>(dev_s, dev_ptr);

    // 将bitmap从GPU复制回CPU以显示
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_ptr,
        bitmap.image_size(),
        cudaMemcpyDeviceToHost));
    bitmap.display_and_exit();

    // 释放内存
    HANDLE_ERROR(cudaFree(dev_s));
    HANDLE_ERROR(cudaFree(dev_ptr));

    return 0;
}