/*
 * Deepam Sarmah
 * 2020050
 * deepam20050@iiitd.ac.in
 */

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <ctime>


#include <cuda_runtime.h>

#define ARRAY_SIZE 512

using namespace std;


struct SoA {
  int *keys;
  int *values;
};

__global__ void vector_add (SoA a, SoA b, SoA c) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < ARRAY_SIZE) {
    c.keys[i] = a.keys[i] + b.keys[i];
    c.values[i] = a.values[i] + b.values[i];
  }
}


int main(){
  struct SoA SoA_data1, SoA_data2, SoA_data3 ,d_SoA_data1, d_SoA_data2, d_SoA_data3;
  // malloc SoAdata1.keys, SoAdata1.values SoA_data2.keys, SoAdata3.keys etc
  SoA_data1.keys = (int *)malloc(ARRAY_SIZE * sizeof(int));
  SoA_data1.values = (int *)malloc(ARRAY_SIZE * sizeof(int));
  SoA_data2.keys = (int *)malloc(ARRAY_SIZE * sizeof(int));
  SoA_data2.values = (int *)malloc(ARRAY_SIZE * sizeof(int));
  SoA_data3.keys = (int *)malloc(ARRAY_SIZE * sizeof(int));
  SoA_data3.values = (int *)malloc(ARRAY_SIZE * sizeof(int));
  // initialize array keys, values
  for (int i = 0; i < ARRAY_SIZE; i++){
    SoA_data1.keys[i] = (i + 1);
    SoA_data1.values[i] = 2 * (i + 1);
    SoA_data2.keys[i] = 3 * (i + 1);
    SoA_data2.values[i] = 4 * (i + 1);
  }
  // cudaMalloc d_SoA_data1.keys, d_SoA_data1.values etc
  cudaMalloc((void **) &d_SoA_data1.keys, ARRAY_SIZE * sizeof(int));
  cudaMalloc((void **) &d_SoA_data1.values, ARRAY_SIZE * sizeof(int));
  cudaMalloc((void **) &d_SoA_data2.keys, ARRAY_SIZE * sizeof(int));
  cudaMalloc((void **) &d_SoA_data2.values, ARRAY_SIZE * sizeof(int));
  cudaMalloc((void **) &d_SoA_data3.keys, ARRAY_SIZE * sizeof(int));
  cudaMalloc((void **) &d_SoA_data3.values, ARRAY_SIZE * sizeof(int));
  // cudaMemcpy d_SoA_data1.keys, d_SoA_data1.values etc
	cudaMemcpy(d_SoA_data1.keys, SoA_data1.keys, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_SoA_data1.values, SoA_data1.values, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_SoA_data2.keys, SoA_data2.keys, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_SoA_data2.values, SoA_data2.values, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  // launching kernel and measuring time
  float gpu_elapsed = 0.0f;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start);
  vector_add<<<(ARRAY_SIZE/256)+1, 256>>>(d_SoA_data1, d_SoA_data2, d_SoA_data3);
  cudaDeviceSynchronize();
  cudaEventRecord(gpu_stop); 
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed, gpu_start, gpu_stop);
	printf("[SoA] Time measured: %.10f milliseconds.\n", gpu_elapsed);
  // copy back to host array
  cudaMemcpy(SoA_data3.keys, d_SoA_data3.keys, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(SoA_data3.values, d_SoA_data3.values, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  // Printing data
	// for (int i = 0; i < ARRAY_SIZE; ++i) {
	// 	printf("%d %d | %d %d | %d %d\n", SoA_data1.keys[i], SoA_data1.values[i], SoA_data2.keys[i], SoA_data2.values[i], SoA_data3.keys[i], SoA_data3.values[i]);
	// }
  // free the device memory
  cudaFree(d_SoA_data1.keys);
  cudaFree(d_SoA_data1.values);
  cudaFree(d_SoA_data2.keys);
  cudaFree(d_SoA_data2.values);
  cudaFree(d_SoA_data3.keys);
  cudaFree(d_SoA_data3.values);
  return 0;     
}