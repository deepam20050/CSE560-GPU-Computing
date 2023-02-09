#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void matrixTransposeShared(float *d_out, float *d_in, int rows, int cols) {
    // 2D block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 2D thread index within a block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first element in the block
    int blockStart = bx * cols * BLOCK_SIZE + by * BLOCK_SIZE * rows;

    // Allocate shared memory to store elements of the block
    __shared__ float block[BLOCK_SIZE][BLOCK_SIZE];

    // Load the elements of the block into shared memory
    int row = blockStart + ty * rows + tx;
    if (row < rows * cols) {
        block[ty][tx] = d_in[row];
    }

    // Synchronize threads within the block
    __syncthreads();

    // Write the transposed block back to the output matrix
    int col = blockStart + ty * rows + tx;
    if (col < rows * cols) {
        d_out[col] = block[tx][ty];
    }
}

int main(int argc, char *argv[]) {
    float *d_in, *d_out;
    float h_A[4][5] = {{1, 2, 3, 4, 20}, {5, 6, 7, 8, 30}, {9, 10, 11, 12, 40}, {13, 14, 15, 16, 50}};

    int rows = 4;
    int cols = 5;

    // Allocate memory for the input and output matrices on the device
    cudaMalloc((void **)&d_in, rows * cols * sizeof(float));
    cudaMalloc((void **)&d_out, rows * cols * sizeof(float));

    // Copy the input matrix to the device
    cudaMemcpy(d_in, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the matrix transpose kernel
    dim3 grid((rows + BLOCK_SIZE - 1) / BLOCK_SIZE, (cols + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    matrixTransposeShared<<<grid, block>>>(d_out, d_in, rows, cols);
    float h_AT[5][4];
    // Copy the output matrix back to the host
    cudaMemcpy(h_AT, d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the transposed matrix
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            std::cout << h_AT[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
}
