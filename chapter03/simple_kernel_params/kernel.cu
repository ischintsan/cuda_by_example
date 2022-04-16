// simple_kernel_params

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"

__global__ void add(int a, int b, int* c) {
	*c = a + b;
}

int main() {
	int c;
	int* dev_c;
	// 分配显存
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));
	// 像调用C语言中的函数一样调用核函数
	add <<<1, 1 >>> (2, 7, dev_c);
	// 主机不能直接对dev_c所指的显存做操作，应该复制回主机内存
	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

	printf("2 + 7 = %d\n", c);
	// 最后要释放之前分配的显存
	cudaFree(dev_c);

	return 0;
}