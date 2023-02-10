/*
 * Deepam Sarmah
 * 2020050
 * deepam20050@iiitd.ac.in
 */
#include <cuda_runtime.h>
#include <cstdio>

const int ROWS = 1024;
const int COLS = 512;
const int THREADS = 16;

float A[ROWS][COLS], A_T[COLS][ROWS], A_T_A[COLS][COLS];

__global__ void global_transpose (float *d_A, float *d_A_T, int rows, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < rows && j < cols) {
    int idx_A = i * cols + j;
    int idx_A_T = j * rows + i;
    d_A_T[idx_A_T] = d_A[idx_A];
  }
}

__global__ void global_matmul (float *d_C, float *d_A, float *d_B, int rowsA, int colsA, int colsB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < rowsA && j < colsB) {
    float sum = 0.0f;
    for (int k = 0; k < colsA; k++) {
      int idxA = i * colsA + k;
      int idxB = k * colsB + j;
      sum += d_A[idxA] * d_B[idxB];
    }
    int idxC = i * colsB + j;
    d_C[idxC] = sum;
  }
}

__global__ void shared_transpose (float *d_A, float *d_A_T, int rows, int cols) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int blockStart = bx * cols * THREADS + by * THREADS * rows;
  __shared__ float block[THREADS][THREADS];
  int row = blockStart + ty * rows + tx;
  if (row < rows * cols) {
      block[ty][tx] = d_A[row];
  }
  __syncthreads();
  int col = blockStart + ty * rows + tx;
  if (col < rows * cols) {
    d_A_T[col] = block[tx][ty];
  }
}

// Ref: Lecture 6, 04CUDA_Memories.pdf, Slides 41-42
__global__ void shared_matmul (float *a, float *b, float *ab, int width) {
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;
  __shared__ float s_a[THREADS][THREADS];
  __shared__ float s_b[THREADS][THREADS];
  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;
  float result = 0;
  for(int p = 0; p < width/THREADS; ++p) {
    s_a[ty][tx] = a[row*width + (p*THREADS + tx)];
    s_b[ty][tx] = b[(p*THREADS + ty)*width + col];
    __syncthreads();
    for(int k = 0; k < THREADS; ++k)
      result += s_a[ty][k] * s_b[k][tx];
    __syncthreads();
  }
  ab[row * width + col] = result;
}

int main() {
  for (int i = 0; i < ROWS; ++i) {
    for (int j = 0; j < COLS; ++j) {
      A[i][j] = static_cast<float>((i + 1) * (j + 1));
    }
  }
  float *d_A, *d_A_T, *d_s_A_T, *d_A_T_A, *d_s_A_T_A;
  cudaMalloc((void **)&d_A, ROWS * COLS * sizeof(float));
  cudaMalloc((void **)&d_A_T, ROWS * COLS * sizeof(float));
  cudaMalloc((void **)&d_s_A_T, ROWS * COLS * sizeof(float));
  cudaMalloc((void **)&d_A_T_A, COLS * COLS * sizeof(float));
  cudaMalloc((void **)&d_s_A_T_A, COLS * COLS * sizeof(float));
  cudaMemcpy(d_A, A, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
  // global memory code
  dim3 blockGlobalDim(THREADS, THREADS);
  dim3 gridGlobalDim((ROWS + blockGlobalDim.x - 1) / blockGlobalDim.x, (COLS + blockGlobalDim.y - 1) / blockGlobalDim.y);
  float gpu_elapsed = 0.0f;
  cudaEvent_t gpu_start, gpu_stop;
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);
  cudaEventRecord(gpu_start);
  global_transpose<<<gridGlobalDim, blockGlobalDim>>>(d_A, d_A_T, ROWS, COLS);
  cudaDeviceSynchronize();
  cudaEventRecord(gpu_stop); 
  cudaEventSynchronize(gpu_stop);
  cudaEventElapsedTime(&gpu_elapsed, gpu_start, gpu_stop);
  printf("[GPU] Global Memory Transpose Time measured: %.9lf milliseconds.\n", gpu_elapsed);
  cudaMemcpy(A_T, d_A_T, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);
  gpu_elapsed = 0.0f;
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);
  cudaEventRecord(gpu_start);
  global_matmul<<<gridGlobalDim, blockGlobalDim>>>(d_A_T_A, d_A_T, d_A, COLS, ROWS, ROWS);
  cudaDeviceSynchronize();
  cudaEventRecord(gpu_stop); 
  cudaEventSynchronize(gpu_stop);
  cudaEventElapsedTime(&gpu_elapsed, gpu_start, gpu_stop);
  printf("[GPU] Global Memory A_T * A Time measured: %.9lf milliseconds.\n", gpu_elapsed);
  cudaMemcpy(A_T_A, d_A_T_A, COLS * COLS * sizeof(float), cudaMemcpyDeviceToHost);
  // shared memory code
  dim3 sharedGridDim((ROWS + THREADS - 1) / THREADS, (COLS + THREADS - 1) / THREADS);
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);
  cudaEventRecord(gpu_start);
  shared_transpose<<<sharedGridDim, blockGlobalDim>>>(d_A, d_s_A_T, ROWS, COLS);
  cudaDeviceSynchronize();
  cudaEventRecord(gpu_stop); 
  cudaEventSynchronize(gpu_stop);
  cudaEventElapsedTime(&gpu_elapsed, gpu_start, gpu_stop);
  printf("[GPU] Shared Memory Transpose Time measured: %.9lf milliseconds.\n", gpu_elapsed);
  cudaMemcpy(A_T, d_s_A_T, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);
  cudaEventRecord(gpu_start);
  shared_matmul<<<sharedGridDim, blockGlobalDim>>>(d_s_A_T, d_A, d_s_A_T_A, COLS);
  cudaDeviceSynchronize();
  cudaEventRecord(gpu_stop); 
  cudaEventSynchronize(gpu_stop);
  cudaEventElapsedTime(&gpu_elapsed, gpu_start, gpu_stop);
  printf("[GPU] Shared Memory A_T * A Time measured: %.9lf milliseconds.\n", gpu_elapsed);
  cudaMemcpy(A_T_A, d_s_A_T_A, COLS * COLS * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_A_T);
  cudaFree(d_s_A_T);
  cudaFree(d_A_T_A);
  cudaFree(d_s_A_T_A);
  return 0;
}