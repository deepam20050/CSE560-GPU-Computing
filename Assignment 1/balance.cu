#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void balanceLoadKernel(int *workLoad, int *threadLoad)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;

    int loadPerThread = (workLoad[0] + numThreads - 1) / numThreads;
    int start = tid * loadPerThread;
    int end = start + loadPerThread;
    if (end > workLoad[0])
    {
        end = workLoad[0];
    }

    int myLoad = end - start;
    threadLoad[tid] = myLoad;
}

int main()
{
    int workLoad = 100000;
    int *d_workLoad;
    int *d_threadLoad;
    int *threadLoad = (int *)malloc(sizeof(int) * BLOCK_SIZE * gridDim.x);

    cudaMalloc((void **)&d_workLoad, sizeof(int));
    cudaMalloc((void **)&d_threadLoad, sizeof(int) * BLOCK_SIZE * gridDim.x);

    cudaMemcpy(d_workLoad, &workLoad, sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = ceil((float)BLOCK_SIZE / (float)gridDim.x);

    balanceLoadKernel<<<numBlocks, BLOCK_SIZE>>>(d_workLoad, d_threadLoad);
    cudaDeviceSynchronize();
    cudaMemcpy(threadLoad, d_threadLoad, sizeof(int) * BLOCK_SIZE * gridDim.x, cudaMemcpyDeviceToHost);

    for (int i = 0; i < BLOCK_SIZE * gridDim.x; i++)
    {
        std::cout << "Thread " << i << ": " << threadLoad[i] << std::endl;
    }

    cudaFree(d_workLoad);
    cudaFree(d_threadLoad);
    free(threadLoad);

    return 0;
}
