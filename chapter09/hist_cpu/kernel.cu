// hist_cpu

#include "../../common/book.h"
#include "time.h"

#define SIZE (100*1024*1024)

int main() {
    // 随机生成100MB的随机数据
    unsigned char* buffer = (unsigned char*)big_random_block(SIZE);
    // 每个字节的取值范围为0x00-0xFF,因此使用大小为256的数组来存储相应的值在buffer中出现的次数,
    // 用于计算直方图
    unsigned int histo[256] = { 0 };
    // 计算时间
    clock_t start, stop;
    start = clock();
    for (int i = 0; i < SIZE; i++) {
        histo[buffer[i]]++;
    }
    stop = clock();
    float elapsedTime = (float)(stop - start) / (float)CLOCKS_PER_SEC * 1000.0f;
    printf("Time to generate:  %3.1f ms\n", elapsedTime);
    // 验证直方图的所有元素加起来是否等于正确的值(应该等于SIZE)
    long histoCount = 0;
    for (int i = 0; i < 256; i++) {
        histoCount += histo[i];
    }
    printf("Histogram Sum:  %ld\n", histoCount);

    // 释放内存
    free(buffer);

    return 0;
}