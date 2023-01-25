/*
 * Deepam Sarmah
 * 2020050
 * deepam20050@iiitd.ac.in
 */

#include <iostream>
#include <stdio.h>
#include <time.h>


#define LENGTH 100000000
using namespace std;

__global__ void vector_add_gpu (float *a, float *b, float *c) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] + b[i];
}

void vector_add_cpu(float *a, float *b, float *c){
    for (int i = 0; i < LENGTH; i ++){
        c[i] = a[i] + b[i];
    }
}

int main(){

    float *a_vec;
    float *b_vec;
    
    float *c_vec;
    float *d_a, *d_b, *d_c;
    float *h_c;


    h_c = (float*)malloc(LENGTH*sizeof(float));
    a_vec = (float*)malloc(LENGTH*sizeof(float));
    b_vec = (float*)malloc(LENGTH*sizeof(float));

    for (int i = 0; i < LENGTH; i ++){
        a_vec[i] = i;
        b_vec[i] = i;
    }

    timespec begin, end;

    clock_gettime(CLOCK_REALTIME, &begin);
    vector_add_cpu(a_vec, b_vec, h_c);
    clock_gettime(CLOCK_REALTIME, &end);

    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;

    double elapsed = seconds + nanoseconds*1e-9;

    // for(int i=0 ; i< LENGTH/100000 ; i++){
	// 	std::cout << h_c[i] << std::endl;
	// }

    printf("[CPU] Time measured: %.3f seconds.\n", elapsed);

    // -- GPU CODE --
    cudaMalloc((void **) &d_a, LENGTH * sizeof(float));
    cudaMalloc((void **) &d_b, LENGTH * sizeof(float));
    cudaMalloc((void **) &d_c, LENGTH * sizeof(float));
    
    cudaMemcpy(d_a, a_vec, LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_vec, LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    
    float gpu_elapsed = 0.0f;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);
    
    vector_add_gpu<<< 1 + LENGTH / 256, 256>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(gpu_stop); 
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed, gpu_start, gpu_stop);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    printf("[GPU] Time measured: %.10f milliseconds.\n", gpu_elapsed);
    return 0;
}