/* Deepam Sarmah
 * 2020050
 * deepam20050@iiitd.ac.in
 */

#include <stdio.h>

__global__ void kernel1(float *a, int offset)
{
    int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
    float x, s, c, root;
    for (int j = 0; j < 1000; j ++){
        x = (float)i;
        s = sinf(x); 
        c = cosf(x);
        root = sqrtf(s*s+c*c);
    }
    a[i] = a[i] + 2*root;
}

 __global__ void kernel2(float *a, int offset)
{
    int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
    float x, s, c, root;
    for (int j = 0; j < 1000; j ++){
        x = (float)i;
        s = sinf(x); 
        c = cosf(x);
        root = sqrtf(s*s+c*c);
    }
    a[i] = a[i] - root;
}
 
 float difference(float *a, float *b, int n) 
{
    float diff = 0;
    for (int i = 0; i < n; i++) {
        diff += fabs(a[i]-b[i]);
    }
    return diff;
}


int main(){

    int blockSize = 256, nStreams = 4;
    int n = 4 * 1024 * blockSize * nStreams;
    int streamSize = n / nStreams;
    int streamBytes = streamSize * sizeof(float);
    int bytes = n * sizeof(float);


    float *a, *d_a, *b, *d_b;
    cudaMallocHost((void**)&a, bytes);      
    cudaMalloc((void**)&d_a, bytes);
    cudaMallocHost((void**)&b, bytes);      
    cudaMalloc((void**)&d_b, bytes);

    float ms;
   
    // create events
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&dummyEvent);

    // creating stream
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
        cudaStreamCreate(&stream[i]);

    // baseline method - using cudaMemcpy and calling kernels on default stream.
    memset(a, 0, bytes);
    cudaEventRecord(startEvent,0);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    kernel1<<<n/blockSize, blockSize>>>(d_a, 0);
    kernel2<<<n/blockSize, blockSize>>>(d_a, 0);
    cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("Time (baseline) : %f\n", ms);




    // using stream
    memset(b, 0, bytes);
    cudaEventRecord(startEvent,0);

    const int gridSize = streamSize / blockSize;
    for (int i = 0; i < nStreams; ++i) {
        cudaMemcpyAsync(d_b + i * streamSize, b + i * streamSize, streamBytes, cudaMemcpyHostToDevice, stream[i]);
        kernel1<<<gridSize, blockSize, 0, stream[i]>>>(d_b, i * streamSize);
        kernel2<<<gridSize, blockSize, 0, stream[i]>>>(d_b, i * streamSize);
        cudaMemcpyAsync(b + i * streamSize, d_b + i * streamSize, streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaDeviceSynchronize();
    // TODO : (You can refer to the lab document for the syntax or use CUDA programming guide)
    // 1. [DONE] Use cudaMemcpyAsync to copy data from host to device
    // 2. [DONE] Call the kernels on non-default streams.
    // 3. [DONE] Use cudaMemcpyAsync to copy data back from device to host
    // 4. [DONE] Some hints - You can divide the complete array into chunks (you can use a for loop for calling kernels for every chunk) where different streams can evaluate for different chunks of the array. You can also call different kernels on different streams. 
    // To obtain marks, you only have to show that using streams, you get some improvement in time over the basic model of using only default stream. 
    // If you would like, you can visualize the memory copies and kernel calls using Nvidia Visual Profiler tool to see how different kernels are running on different streams.

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    // TODO : [DONE] Mention this time and difference between array a and b in the report. 
    // If your implementation is correct, this time will be slightly less than the baseline method and the difference between array a aand b will be zero.
    printf("Time : %f\n", ms);
    printf("  difference : %e\n", difference(a, b, n));

    // cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(dummyEvent);

    // [DONE] TODO : Destroy the streams. (You can refer to the lab document for the same)
    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(stream[i]);
    }
      
    cudaFree(d_a);
    cudaFreeHost(a);
    cudaFree(d_b);
    cudaFreeHost(b);

    return 0;

}