// set_gpu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "../../common/book.h"

int main()
{
    cudaDeviceProp prop;
    int dev;

    HANDLE_ERROR(cudaGetDevice(&dev));
    printf("ID of current CUDA device:  %d\n", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;//设置选择条件，算力>1.3
    prop.minor = 3;
    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));//返回最匹配的设备id(若所有设备都没达到条件，也会返回一个最匹配的)
    printf("ID of CUDA device closest to revision 1.3:  %d\n", dev);

    HANDLE_ERROR(cudaSetDevice(dev));//设置GPU设备，之后所有的设备操作都将在此设备上执行

    return 0;
}
