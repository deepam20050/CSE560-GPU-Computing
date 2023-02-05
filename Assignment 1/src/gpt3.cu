#include <cuda_runtime.h>

__global__ void compute_julia_GPU_optimized(float *d_output, int width, int height, float real_min, float real_max, float imag_min, float imag_max, int max_iter) {

    // block size
    const int block_size = 32;
    // block id and thread id
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // index of the output array
    int index = bx * block_size * height + by * block_size + ty * width + tx;
    // coordinate in the complex plane
    float real = real_min + (float)(tx + bx * block_size) * (real_max - real_min) / (float)(width - 1);
    float imag = imag_min + (float)(ty + by * block_size) * (imag_max - imag_min) / (float)(height - 1);
    // store values in shared memory
    __shared__ float s_real[block_size][block_size];
    __shared__ float s_imag[block_size][block_size];
    s_real[tx][ty] = real;
    s_imag[tx][ty] = imag;
    __syncthreads();
    // calculate the number of iterations for each pixel
    float c_real = -0.7f;
    float c_imag = 0.27015f;
    float z_real = real;
    float z_imag = imag;
    int iter = 0;
    for (; iter < max_iter; ++iter) {
        float z_real_squared = z_real * z_real;
        float z_imag_squared = z_imag * z_imag;
        if (z_real_squared + z_imag_squared > 4.0f) {
            break;
        }
        z_imag = 2.0f * z_real * z_imag + c_imag;
        z_real = z_real_squared - z_imag_squared + c_real;
    }
    d_output[index] = (float)iter / (float)max_iter;
}
