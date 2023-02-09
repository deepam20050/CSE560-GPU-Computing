#include <cuda_runtime.h>
#include <iostream>

__global__ void matmul(float *d_C, float *d_A, float *d_B, int rowsA, int colsA, int colsB)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rowsA && j < colsB)
    {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++)
        {
            int indexA = i * colsA + k;
            int indexB = k * colsB + j;

            sum += d_A[indexA] * d_B[indexB];
        }
        int indexC = i * colsB + j;
        d_C[indexC] = sum;
    }
}

int main()
{
    // Input matrices
    float h_A[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    float h_B[3][4] = {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}};

    int rowsA = 4;
    int colsA = 3;
    int colsB = 4;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, rowsA * colsA * sizeof(float));
    cudaMalloc((void **)&d_B, colsA * colsB * sizeof(float));
    cudaMalloc((void **)&d_C, rowsA * colsB * sizeof(float));

    // Copy input matrices to device memory
    cudaMemcpy(d_A, h_A, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, colsA * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // Launch matrix multiplication kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((rowsA + blockDim.x - 1) / blockDim.x, (colsB + blockDim.y - 1) / blockDim.y);
    matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, rowsA, colsA, colsB);

    // Copy result back to host memory
    float h_C[4][4];
    cudaMemcpy(h_C, d_C, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < colsB; j++) {
          std::cout << h_C[i][j] << " "; 
        }
        std::cout << '\n';
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
