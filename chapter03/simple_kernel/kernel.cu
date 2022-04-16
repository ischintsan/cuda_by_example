//simple_kernel

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel() {
}

int main() {
	kernel <<<1, 1>>> ();// 像调用C语言中的函数一样调用核函数
	printf("hello world!\n");

	return 0;
}