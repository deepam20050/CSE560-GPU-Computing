#define BLOCK_SIZE 1024

__global__ void compute_julia_GPU_kernel(complex<float> c, unsigned char *image, int width, int height)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= width || gy >= height)
        return;

    int i = gx + gy * width;

    complex<float> z(gx * 4.0f / width - 2.0f, gy * 4.0f / height - 2.0f);
    int iterations = 0;
    while (abs(z) <= SQRT_2 && iterations < MAX_ITER)
    {
        z = z * z + c;
        iterations++;
    }

    float r, g, b;
    HSVtoRGB(&r, &g, &b, (float)iterations / MAX_ITER, 1.0f, 1.0f);

    image[3 * i] = (unsigned char)(255.0f * r);
    image[3 * i + 1] = (unsigned char)(255.0f * g);
    image[3 * i + 2] = (unsigned char)(255.0f * b);
}

void compute_julia_GPU(complex<float> c, unsigned char *image)
{
    int width = N;
    int height = N;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    compute_julia_GPU_kernel<<<gridDim, blockDim>>>(c, image, width, height);
}
