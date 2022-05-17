// heat_2d

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"
#include "../../common/cpu_anim.h"

#define DIM 1024
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

// 声明纹理引用，这些变量将位于GPU上
texture<float, 2>  texConstSrc;
texture<float, 2>  texIn;
texture<float, 2>  texOut;

__global__ void copy_const_kernel(float* iptr) {
    // 将threadIdx/BlockIdx映射到像素位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * gridDim.x * blockDim.x + x;

    // 当温度不为0时，才会执行复制。这是为了维持非热源单位在上一次计算得到的温度值
    float center = tex2D(texConstSrc, x, y);
    if (center != 0)
        iptr[offset] = center;
}

__global__ void blend_kernel(float* dst, bool dstOut) {
    // 将threadIdx/BlockIdx映射到像素位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * gridDim.x * blockDim.x + x;

    float t, l, c, r, b;
    if (dstOut) {
        t = tex2D(texIn, x, y - 1); // top
        l = tex2D(texIn, x - 1, y); // left
        c = tex2D(texIn, x, y);     // center
        r = tex2D(texIn, x + 1, y); // right
        b = tex2D(texIn, x, y + 1); // bottom
    }
    else {
        t = tex2D(texOut, x, y - 1);
        l = tex2D(texOut, x - 1, y);
        c = tex2D(texOut, x, y);
        r = tex2D(texOut, x + 1, y);
        b = tex2D(texOut, x, y + 1);
    }
    // 更新公式：T_new = T_old + k * sum(T_neighbor - T_old)
    dst[offset] = c + SPEED * (t + b + l + r - 4 * c);
}

// 更新函数中需要的全局变量
struct DataBlock
{
    unsigned char* dev_bitmap;
    float* dev_inSrc;  // 输入缓冲区
    float* dev_outSrc;  // 输出缓冲区
    float* dev_constSrc;  // 初始化的热源
    CPUAnimBitmap* bitmap;

    cudaEvent_t start, stop;
    float totalTime;
    float frames;
};

// 每一帧动画将调用anim_gpu()
void anim_gpu(DataBlock* data, int ticks) {
    HANDLE_ERROR(cudaEventRecord(data->start, 0));

    // 每个Block有(16, 16)个Thread，(DIM/16, DIM/16)组织成一个Grid
    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    CPUAnimBitmap* bitmap = data->bitmap;

    // 每一帧动画都经过了90轮迭代计算，可以修改这个值
    // 由于tex是全局并且是有界的，因此需要通过一个标识来选择
    // 每次迭代中哪个是输入/输出
    volatile bool dstOut = true;
    for (int i = 0; i < 90; i++) {
        float* in, * out;
        if (dstOut) {
            in = data->dev_inSrc;
            out = data->dev_outSrc;
        }
        else {
            in = data->dev_outSrc;
            out = data->dev_inSrc;
        }

        // 为了简单，热源单元本身的温度将保持不变。但是，热量可以从更热的单元传导到更冷的单元
        copy_const_kernel<<<blocks, threads>>>(in);
        // 更新每一个单元
        blend_kernel<<<blocks, threads>>>(out, dstOut);
        // 交换输出和输入，将本次计算的输出作为下次计算的输入
        dstOut = !dstOut;
    }

    // 将温度转为颜色
    float_to_color<<<blocks, threads>>>(data->dev_bitmap, data->dev_inSrc);
    // 将结果复制回CPU
    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(),
        data->dev_bitmap,
        bitmap->image_size(),
        cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(data->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(data->stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, data->start, data->stop));  // 计算每一帧动画需要的时间

    data->totalTime += elapsedTime;
    data->frames++;
    printf("Average Time per frame:  %3.1f ms\n", data->totalTime / data->frames);
}

void anim_exit(DataBlock* data) {
    // 取消纹理内存的绑定
    HANDLE_ERROR(cudaUnbindTexture(texConstSrc));
    HANDLE_ERROR(cudaUnbindTexture(texIn));
    HANDLE_ERROR(cudaUnbindTexture(texOut));

    HANDLE_ERROR(cudaFree(data->dev_inSrc));
    HANDLE_ERROR(cudaFree(data->dev_outSrc));
    HANDLE_ERROR(cudaFree(data->dev_constSrc));

    HANDLE_ERROR(cudaEventDestroy(data->start));
    HANDLE_ERROR(cudaEventDestroy(data->stop));
}

int main() {
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;

    HANDLE_ERROR(cudaEventCreate(&data.start));
    HANDLE_ERROR(cudaEventCreate(&data.stop));

    HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));

    // 假设float类型的大小为4个字符(即rgba)
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size()));

    // 将三个内存绑定到之前声明的纹理应用
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR(cudaBindTexture2D(NULL, texConstSrc, data.dev_constSrc, desc, DIM, DIM, sizeof(float) * DIM));
    HANDLE_ERROR(cudaBindTexture2D(NULL, texIn, data.dev_inSrc, desc, DIM, DIM, sizeof(float) * DIM));
    HANDLE_ERROR(cudaBindTexture2D(NULL, texOut, data.dev_outSrc, desc, DIM, DIM, sizeof(float) * DIM));

    float* temp = (float*)malloc(bitmap.image_size());
    // 下面加入一些热源
    for (int i = 0; i < DIM * DIM; i++) {
        temp[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
            temp[i] = MAX_TEMP;
    }
    temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
    temp[DIM * 700 + 100] = MIN_TEMP;
    temp[DIM * 300 + 300] = MIN_TEMP;
    temp[DIM * 200 + 700] = MIN_TEMP;
    for (int y = 800; y < 900; y++) {
        for (int x = 400; x < 500; x++) {
            temp[x + y * DIM] = MIN_TEMP;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_constSrc,
        temp,
        bitmap.image_size(),
        cudaMemcpyHostToDevice));

    for (int y = 800; y < DIM; y++) {
        for (int x = 0; x < 200; x++) {
            temp[x + y * DIM] = MAX_TEMP;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_inSrc,
        temp,
        bitmap.image_size(),
        cudaMemcpyHostToDevice));
    free(temp);
    // 每次需要生成一帧图像，就调用一次anim_gpu，之后再调用anim_exit将分配的显存释放掉
    bitmap.anim_and_exit((void (*)(void*, int))anim_gpu,
        (void (*)(void*))anim_exit);

    return 0;
}