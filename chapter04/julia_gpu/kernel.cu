// julia_gpu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"
#include "../../common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex
{
    float   r;
    float   i;
    // cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ cuComplex(float a, float b) : r(a), i(b) { }  // __device__表示代码将在设备而不是主机上运行。
                                                             // 由于声明为__device__函数，因此只能从其他__device__
                                                             // 或__global__函数中调用他们
    __device__ float magnitude2(void)						 
    {														 
        return r * r + i * i;
    }
    __device__ cuComplex operator * (const cuComplex& a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator + (const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(unsigned char* ptr) {
    // 将threadIdx/blockIdx映射到像素位置
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = x + y * gridDim.x;  // 对所有的线程块而言，gridDim是常数，用来保存线程格每一维的大小，
                                     // gridDim.x是线程格的宽度

    int juliaValue = julia(x, y);
    ptr[offset * 4 + 0] = 255 * juliaValue;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main() {
    CPUBitmap bitmap(DIM, DIM);
    unsigned char* dev_ptr;

    HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, bitmap.image_size()));

    dim3 grid(DIM, DIM);  // 二维线程格(Grid)。dim3表示一个三维数组，在这里第三维其实就是1，同dim3 grid(DIM, DIM, 1)
    kernel<<<grid, 1>>>(dev_ptr);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_ptr, bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();

    HANDLE_ERROR(cudaFree(dev_ptr));

    return 0;
}