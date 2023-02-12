void compute_julia_GPU(complex<float> c, unsigned char * image)
{
    int size = N * N * 3;
    unsigned char *device_image;
    cudaMalloc((void **)&device_image, size);

    dim3 block(16, 16);
    dim3 grid(N / block.x, N / block.y);

    // Launch CUDA kernel to compute Julia set on GPU
    compute_julia_kernel<<<grid, block>>>(device_image, c);

    cudaMemcpy(image, device_image, size, cudaMemcpyDeviceToHost);

    cudaFree(device_image);
}

__global__ void compute_julia_kernel(unsigned char *image, complex<float> c)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int index = y * N + x;

    if (x < N && y < N)
    {
        complex<float> z;
        z.real(SQRT_2 * (2.0f * x / N - 1.0f));
        z.imag(SQRT_2 * (2.0f * y / N - 1.0f));

        int iterations = 0;
        while (iterations < MAX_ITER && abs(z) < 2.0f)
        {
            z = z * z + c;
            iterations++;
        }

        float h = (float)iterations / MAX_ITER;
        float s = 1.0f;
        float v = 1.0f;
        float r, g, b;
        HSVtoRGB(&r, &g, &b, h, s, v);

        image[index * 3 + 0] = (unsigned char)(255.0f * r);
        image[index * 3 + 1] = (unsigned char)(255.0f * g);
        image[index * 3 + 2] = (unsigned char)(255.0f * b);
    }
}
