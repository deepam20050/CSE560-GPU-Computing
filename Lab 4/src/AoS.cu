/*
 * Deepam Sarmah
 * 2020050
 * deepam20050@iiitd.ac.in
 */

#include <iostream>
#include <stdio.h>
#include <ctime>
#include <cuda_runtime.h>

#define ARRAY_SIZE 512

using namespace std;

struct record {
  int key;
	int value;
};

__global__ void vector_add (record* a, record* b, record* c){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < ARRAY_SIZE) {
		c[i].key = a[i].key + b[i].key;
		c[i].value = a[i].value + b[i].value;
	}
}

int main() {
	struct record *AoS_data1, *AoS_data2, *AoS_data3 , *d_AoS_data1, *d_AoS_data2, *d_AoS_data3;
	// malloc AoSdata1, AoS_data2, AoSdata3
	AoS_data1 = (record *) malloc(ARRAY_SIZE * sizeof(record));
	AoS_data2 = (record *) malloc(ARRAY_SIZE * sizeof(record));
	AoS_data3 = (record *) malloc(ARRAY_SIZE * sizeof(record));
	// initialize array keys, values
	for (int i = 0; i < ARRAY_SIZE; i ++){
		AoS_data1[i].key = (i + 1);
		AoS_data1[i].value = 2 * (i + 1);
		AoS_data2[i].key = 3 * (i + 1);
		AoS_data2[i].value = 4 * (i + 1);
	}
	// cudaMalloc
	cudaMalloc((void **) &d_AoS_data1, ARRAY_SIZE * sizeof(record));
	cudaMalloc((void **) &d_AoS_data2, ARRAY_SIZE * sizeof(record));
	cudaMalloc((void **) &d_AoS_data3, ARRAY_SIZE * sizeof(record));
	// cudaMemcpy
	cudaMemcpy(d_AoS_data1, AoS_data1, ARRAY_SIZE * sizeof(record), cudaMemcpyHostToDevice);
	cudaMemcpy(d_AoS_data2, AoS_data2, ARRAY_SIZE * sizeof(record), cudaMemcpyHostToDevice);
	// launching kernel and measuring time
	float gpu_elapsed = 0.0f;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start);
	vector_add<<<(ARRAY_SIZE/256)+1, 256>>>(d_AoS_data1, d_AoS_data2, d_AoS_data3);
	cudaDeviceSynchronize();
	cudaEventRecord(gpu_stop); 
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed, gpu_start, gpu_stop);
	printf("[AoS] Time measured: %.10f milliseconds.\n", gpu_elapsed);
	// cudaMemcpy back to host array
	cudaMemcpy(AoS_data3, d_AoS_data3, ARRAY_SIZE * sizeof(record), cudaMemcpyDeviceToHost);
	// Printing data
	// for (int i = 0; i < ARRAY_SIZE; ++i) {
	// 	printf("%d %d | %d %d | %d %d\n", AoS_data1[i].key, AoS_data1[i].value, AoS_data2[i].key, AoS_data2[i].value, AoS_data3[i].key, AoS_data3[i].value);
	// }
	// free the device memory
	cudaFree(d_AoS_data1);
	cudaFree(d_AoS_data2);
	cudaFree(d_AoS_data3);
	return 0;
}