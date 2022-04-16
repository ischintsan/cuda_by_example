//add_loop_cpu

#include <stdio.h>

#define N 10

void add(int* a, int* b, int* c)
{
    int tid = 0;  // 这是第0个CPU，因此索引从0开始
    while (tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += 1;  // 由于我们只有一个CPU，因此每次递增1
    }
}

int main()
{
    int a[N], b[N], c[N];

    // 初始化两个数组
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }
    add(a, b, c);
    // 打印结果
    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
}