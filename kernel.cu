#include <cuda_runtime.h>
#include <cstdio>

__global__ void helloKernel()
{
    printf("Hello from the GPU!\n");
}

extern "C" void runCudaTest()
{
    helloKernel<<<1, 1 >>>();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n",
            cudaGetErrorString(error));
        return;
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        printf("Kernel execution failed: %s\n",
            cudaGetErrorString(error));
    }
}
