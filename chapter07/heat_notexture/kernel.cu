// heat_notexture

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"
#include "../../common/cpu_anim.h"

#define DIM 1024
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

__global__ void copy_const_kernel(float* iptr, const float* cptr) {
    // 将threadIdx/BlockIdx映射到像素位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * gridDim.x * blockDim.x + x;

    // 当温度不为0时，才会执行复制。这是为了维持非热源单位在上一次计算得到的温度值
    if (cptr[offset] != 0)
        iptr[offset] = cptr[offset];
}

__global__ void blend_kernel(float* outSrc, const float* inSrc) {
    // 将threadIdx/BlockIdx映射到像素位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * gridDim.x * blockDim.x + x;

    // 每个Thread都负责计算一个单元(一个像素)：读取对应单元及其相邻单元的温度值，
    // 然后执行更新运算，将得到的新值更新到对应的单元。
    int left = offset - 1;
    int right = offset + 1;
    if (x == 0)
        left++;  // 边缘处理，下同
    if (x == DIM - 1)
        right--;

    int top = offset - DIM;
    int bottom = offset + DIM;
    if (y == 0)
        top += DIM;
    if (y == DIM - 1)
        bottom -= DIM;

    // 更新公式：T_new = T_old + k * sum(T_neighbor - T_old)
    outSrc[offset] = inSrc[offset] + SPEED * (inSrc[left] + inSrc[right] 
                                            + inSrc[top] + inSrc[bottom]
                                            - inSrc[offset] * 4);
}

// 更新函数中需要的全局变量
struct DataBlock{
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
    for (int i = 0; i < 90; i++) {
        // 为了简单，热源单元本身的温度将保持不变。但是，热量可以从更热的单元传导到更冷的单元
        copy_const_kernel<<<blocks, threads>>>(data->dev_inSrc, data->dev_constSrc);
        // 更新每一个单元
        blend_kernel<<<blocks, threads>>>(data->dev_outSrc, data->dev_inSrc);
        // 交换输出和输入，将本次计算的输出作为下次计算的输入
        swap(data->dev_inSrc, data->dev_outSrc);
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

    float* temp = (float*)malloc(bitmap.image_size());
    // 下面加入一些热源
    for (int i = 0; i < DIM*DIM; i++) {
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