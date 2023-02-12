/**
Copyright (c) 2023 [Ojaswa Sharma, IIIT Delhi]. All rights reserved.

This computer code is proprietary and confidential. It is provided for the sole use of students enrolled in CSE 560 - GPU Computing. Any unauthorized use, reproduction, distribution or modification of this code, in whole or in part, is strictly prohibited.

By accessing and using the code, you agree to be bound by the terms and conditions of this notice. Unauthorized use may result in severe civil and criminal penalties, and will be prosecuted to the maximum extent possible under the law.

This notice constitutes an agreement between you and the author of the code, and may only be modified in writing signed by both parties.
**/

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

/*
 * Deepam Sarmah
 * 2020050
 * deepam20050@iiitd.ac.in
 */

/*
 References:
 [1] http://stackoverflow.com/questions/23711681/generating-custom-color-palette-for-julia-set
 [2] http://www.cs.rit.edu/~ncs/color/t_convert.html
 [3] https://thrust.github.io/doc/group__complex__numbers.html
*/

#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/complex.h>

#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

const int N = 1024;
const float SQRT_2 = 1.4142;
const int MAX_ITER = 512;

__host__ __device__ void HSVtoRGB( float *r, float *g, float *b, float h, float s, float v );
void saveImage(int width, int height, unsigned char * bitmap, complex<float> seed, int who);
void compute_julia_CPU(complex<float> c, unsigned char * image);
void compute_julia_GPU(complex<float> c, unsigned char * image);
bool compare_CPU_GPU(unsigned char *image_CPU, unsigned char *image_GPU);

__global__ void compute_julia_kernel(unsigned char *image, thrust::complex<float> c) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int index = y * N + x;
	if (x < N && y < N) {
		thrust::complex<float> z_old(0.0f, 0.0f);
		thrust::complex<float> z_new(4.0f * x / N - 2.0f, 4.0f * y / N - 2.0f);
		int iter = 0;
		while (iter < MAX_ITER) {
			z_old.real(z_new.real());
			z_old.imag(z_new.imag());
			z_new *= z_new;
			z_new += c;
			if (thrust::norm(z_new) > 4.0f) {
				break;
			}
			++iter;
		}
		float brightness = (iter < MAX_ITER) ? 1.0f : 0.0f;
		float hue = (iter % MAX_ITER) / float(MAX_ITER - 1);
		hue = (120 * sqrtf(hue) + 150);
		float r, g, b;
		HSVtoRGB(&r, &g, &b, hue, 1.0f, brightness);
		image[index * 3] = (unsigned char)(b * 255);
		image[index * 3 + 1] = (unsigned char)(g * 255);
		image[index * 3 + 2] = (unsigned char)(r * 255);
	}
}

__global__ void dummy_kernel()
{
	int tx = threadIdx.x + blockDim.x*blockIdx.x;
	tx++;
}

int main(int argc, char **argv)
{
	complex<float> c(0.285f, 0.01f);
	if(argc > 2)
	{
		c.real(atof(argv[1]));
		c.imag(atof(argv[2]));
	} else
		fprintf(stderr, "Usage: %s <real> <imag>\nWhere <real> and <imag> form the complex seed for the Julia set.\n", argv[0]);

	unsigned char *image_CPU_host = new unsigned char[N*N*3]; //RGB image
	unsigned char *image_GPU_host = new unsigned char[N*N*3]; //RGB image

	// Compute Julia set on CPU
	struct timespec start_cpu, end_cpu;
	float msecs_cpu;
	fprintf(stderr, "Performing Julia set computation on CPU... ");	
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_cpu);	

	compute_julia_CPU(c, image_CPU_host);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_cpu);
	msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
	fprintf(stderr, "done in %f milliseconds.\n", msecs_cpu);

	// Compute Julia set on GPU
	cudaEvent_t start_gpu, end_gpu;
	float msecs_gpu;
	fprintf(stderr, "Performing Julia set computation on GPU... ");	
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&end_gpu);
	cudaEventRecord(start_gpu, 0);

	compute_julia_GPU(c, image_GPU_host);

	cudaEventRecord(end_gpu, 0);
	cudaEventSynchronize(end_gpu);
	cudaEventElapsedTime(&msecs_gpu, start_gpu, end_gpu);
	cudaEventDestroy(start_gpu);
	cudaEventDestroy(end_gpu);
	fprintf(stderr, "done in %f milliseconds.\n", msecs_gpu);

	bool result = compare_CPU_GPU(image_CPU_host, image_GPU_host);
	fprintf(stderr, "CPU-GPU results do %smatch!\n", (result)?"":"not ");

	saveImage(N, N, image_CPU_host, c, 0); // for CPU image
	saveImage(N, N, image_GPU_host, c, 1); // for GPU image
	delete[] image_CPU_host;
	delete[] image_GPU_host;
}

void compute_julia_CPU(complex<float> c, unsigned char * image)
{
	complex<float> z_old(0.0f, 0.0f);
	complex<float> z_new(0.0f, 0.0f);
	for(int y=0; y<N; y++)
		for(int x=0; x<N; x++)
		{
			z_new.real(4.0f * x / (N) - 2.0f);
			z_new.imag(4.0f * y / (N) - 2.0f);
			int i;
			for(i=0; i<MAX_ITER; i++)
			{
				z_old.real(z_new.real());
				z_old.imag(z_new.imag());
				z_new = pow(z_new, 2);
				z_new += c;
				if(norm(z_new) > 4.0f) break;
			}
			float brightness = (i<MAX_ITER) ? 1.0f : 0.0f;
			float hue = (i % MAX_ITER)/float(MAX_ITER - 1);
			hue = (120*sqrtf(hue) + 150);
			float r, g, b;
			HSVtoRGB(&r, &g, &b, hue, 1.0f, brightness);
			image[(x + y*N)*3 + 0] = (unsigned char)(b*255);
			image[(x + y*N)*3 + 1] = (unsigned char)(g*255);
			image[(x + y*N)*3 + 2] = (unsigned char)(r*255);
		}
}

void compute_julia_GPU(complex<float> c, unsigned char * image) {
	int size = N * N * 3;
	unsigned char *device_image;
	cudaMalloc((void **)&device_image, size);
	dim3 block(32, 32);
	dim3 grid(N / block.x, N / block.y);
	thrust::complex <float> c_thrust(c.real(), c.imag()); 
	compute_julia_kernel<<<grid, block>>>(device_image, c_thrust);
	cudaDeviceSynchronize();
	cudaMemcpy(image, device_image, size, cudaMemcpyDeviceToHost);
	cudaFree(device_image);
}

//Returns true if GPU results match CPU results, else returns false
bool compare_CPU_GPU(unsigned char *image_CPU, unsigned char *image_GPU) {
	constexpr int nelem = N*N*3;
	float mean_cpu = 0.0f;
	for (int i = 0; i < nelem; ++i) {
		mean_cpu += image_CPU[i];
	}
	mean_cpu /= nelem;
	float sst = 0.0f, sse = 0.0f;
	for (int i = 0; i < nelem; ++i) {
		sst += (image_CPU[i] - mean_cpu) * (image_CPU[i] - mean_cpu);
		sse += (image_GPU[i] - mean_cpu) * (image_GPU[i] - mean_cpu);
	}
	return (sse / sst) >= 0.99f;
}

void saveImage(int width, int height, unsigned char * bitmap, complex<float> seed, int who)
{
	char imageName[256];
	sprintf(imageName, "%d %d Julia %.3f + i%.3f.png", who, N, seed.real(), seed.imag());
	stbi_write_png(imageName, width, height, 3, bitmap, width*3);
	fprintf(stderr, "Image saved as: %s\n", imageName);
}

// r,g,b values are from 0 to 1
// h = [0,360], s = [0,1], v = [0,1]
//		if s == 0, then h = -1 (undefined)
__host__ __device__ void HSVtoRGB( float *r, float *g, float *b, float h, float s, float v )
{
	int i;
	float f, p, q, t;
	if( s == 0 ) {
		// achromatic (grey)
		*r = *g = *b = v;
		return;
	}
	h /= 60;			// sector 0 to 5
	i = floor( h );
	f = h - i;			// factorial part of h
	p = v * ( 1 - s );
	q = v * ( 1 - s * f );
	t = v * ( 1 - s * ( 1 - f ) );
	switch( i ) {
		case 0:
			*r = v;
			*g = t;
			*b = p;
			break;
		case 1:
			*r = q;
			*g = v;
			*b = p;
			break;
		case 2:
			*r = p;
			*g = v;
			*b = t;
			break;
		case 3:
			*r = p;
			*g = q;
			*b = v;
			break;
		case 4:
			*r = t;
			*g = p;
			*b = v;
			break;
		default:		// case 5:
			*r = v;
			*g = p;
			*b = q;
			break;
	}
}
