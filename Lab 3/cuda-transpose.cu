#include <cuda_runtime.h>
#include <iostream>

__global__ void transpose(float *d_A, float *d_AT, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols)
    {
        int index_A = i * cols + j;
        int index_AT = j * rows + i;

        d_AT[index_AT] = d_A[index_A];
    }
}

int main()
{
    // Input matrix
    float h_A[4][5] = {{1, 2, 3, 4, 20}, {5, 6, 7, 8, 30}, {9, 10, 11, 12, 40}, {13, 14, 15, 16, 50}};

    int rows = 4;
    int cols = 5;

    // Allocate device memory
    float *d_A, *d_AT;
    cudaMalloc((void **)&d_A, rows * cols * sizeof(float));
    cudaMalloc((void **)&d_AT, rows * cols * sizeof(float));

    // Copy input matrix to device memory
    cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch transpose kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
    transpose<<<gridDim, blockDim>>>(d_A, d_AT, rows, cols);

    // Copy result back to host memory
    float h_AT[5][4];
    cudaMemcpy(h_AT, d_AT, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            std::cout << h_AT[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_AT);

    return 0;
}
