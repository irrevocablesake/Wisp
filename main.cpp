#include <cstdio>

extern "C" void runCudaTest();

int main()
{
    printf("Hello from the CPU!\n");

    runCudaTest();

    return 0;
}
