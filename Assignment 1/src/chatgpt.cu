#include <cuda_runtime.h>
#include <iostream>

__global__ void julia_set_kernel(int *output, float *x, float *y, int width, int height, float real_c, float imag_c)
{
    __shared__ float s_x[16][16];
    __shared__ float s_y[16][16];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = i + j * width;

    if (i >= width || j >= height) return;

    s_x[threadIdx.y][threadIdx.x] = x[i];
    s_y[threadIdx.y][threadIdx.x] = y[j];

    __syncthreads();

    float real = s_x[threadIdx.y][threadIdx.x];
    float imag = s_y[threadIdx.y][threadIdx.x];
    int value = 0;

    for (int iter = 0; iter < 255; iter++)
    {
        float r2 = real * real;
        float i2 = imag * imag;

        if (r2 + i2 > 4.0f)
        {
            value = iter;
            break;
        }

        imag = 2.0f * real * imag + imag_c;
        real = r2 - i2 + real_c;
    }

    output[index] = value;
}

int main()
{
    int width = 800, height = 600;
    int size = width * height;
    float real_c = -0.8f, imag_c = 0.156f;
    float *x, *y;
    int *output;

    cudaMallocManaged(&x, width * sizeof(float));
    cudaMallocManaged(&y, height * sizeof(float));
    cudaMallocManaged(&output, size * sizeof(int));

    for (int i = 0; i < width; i++)
    {
        x[i] = (i / (float)width) * 4.0f - 2.0f;
    }

    for (int i = 0; i < height; i++)
    {
        y[i] = (i / (float)height) * 4.0f - 2.0f;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    julia_set_kernel<<<gridSize, blockSize>>>(output, x, y, width, height, real_c, imag_c);

    cudaDeviceSynchronize();

    // Convert the output array to color values and save the image

    cudaFree(x);
    cudaFree(y);
    cudaFree(output);

    return 0;
}
