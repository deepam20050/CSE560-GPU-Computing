/*
 * Deepam Sarmah
 * 2020050
 * deepam20050@iiitd.ac.in
 */

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel(int *a_d, int *max) {
  *a_d += 1;
  if (*a_d > *max) {
    *max = *a_d;
  }
}

int main() {
  int a = 0, maxVal = 0, *a_d, *max;
  
  cudaMalloc((void**)&a_d, sizeof(int));
  cudaMalloc((void**)&max, sizeof(int));
  cudaMemcpy(a_d, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(max, &maxVal, sizeof(int), cudaMemcpyHostToDevice);

  float milliseconds = 0.0f;
  cudaEvent_t start, stop;
  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  
  kernel<<<1000, 1000>>>(a_d, max);
  cudaDeviceSynchronize();
  
  cudaEventRecord(stop); 
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(&a, a_d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&maxVal, max, sizeof(int), cudaMemcpyDeviceToHost);
  
  printf("a = %d\nmax = %d\n", a, maxVal);
  cudaFree(a_d);
  cudaFree(max);
  printf("Time measured by GPU: %.10f milliseconds.\n", milliseconds);
}