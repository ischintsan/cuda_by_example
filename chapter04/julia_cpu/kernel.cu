// julia_cpu

#include <stdio.h>
#include "../../common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex
{
    float r;  // 复数的实数部分
    float i;  // 复数的虚数部分

    cuComplex(float a, float b) :r(a), i(b) { }

    float magnitude2(void)
    {
        return r * r + i * i;  // 复数的模的平方
    }

    cuComplex operator * (const cuComplex& a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    cuComplex operator + (const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

int julia(int x, int y)
{
    const float scale = 1.5;
    // DIM / 2 - x、DIM / 2 - y将原点定位到图像中心
    // 除以(DIM / 2)是为了确保图像的范围为[-1.0, 1.0]
    // scale是用来缩放图像的，可以自行修改
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);  // 复数C = -0.8 + 0.156i
    cuComplex z(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++)
    {
        z = z * z + c; //Zn+1 = Zn^2 + C
        if (z.magnitude2() > 1000)
        {
            // 迭代200次，每次迭代完都判断结果是否超过阈值(这里是1000)，如果超过就不属于julia集
            return 0;
        }
    }
    return 1;  // 属于Julia集
}

void kernel(unsigned char* ptr)
{
    for (int y = 0; y < DIM; y++)
    {
        for (int x = 0; x < DIM; x++)
        {
            int offset = x + y * DIM;  // 像素在内存中的线性偏移，因为图像在内存中实际是一维存储的

            int juliaValue = julia(x, y);  // 判断点(x, y)是否属于Julia集合，属于返回1，不属于返回0
            // juliaValue为0时为黑色(0,0,0)，为1时为红色(255,0,0)
            ptr[offset * 4 + 0] = 255 * juliaValue; // red通道
            ptr[offset * 4 + 1] = 0;                // green通道
            ptr[offset * 4 + 2] = 0;                // blue通道
            ptr[offset * 4 + 3] = 255;              // alpha通道
        }
    }
}

int main()
{
    CPUBitmap bitmap(DIM, DIM);
    unsigned char* ptr = bitmap.get_ptr();
    kernel(ptr);  // 将指向图像的指针传递给核函数
    bitmap.display_and_exit();

    getchar();
    return 0;
}