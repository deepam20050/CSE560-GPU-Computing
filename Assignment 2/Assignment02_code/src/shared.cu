__global__ void convKernel(unsigned char* inputImageData, unsigned char* outputImageData, int channels, int imageWidth, int imageHeight)
{
    __shared__ unsigned char tile[imageChannels][32][32];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * 30;
    int by = blockIdx.y * 30;
    int x = bx + tx;
    int y = by + ty;

    // Load input tile from global memory to shared memory
    for (int k = 0; k < channels; k++)
    {
        if (x < imageWidth && y < imageHeight)
            tile[k][ty][tx] = inputImageData[(y * imageWidth + x) * channels + k];
        else
            tile[k][ty][tx] = 0;
    }

    __syncthreads();

    float sum = 0;
    int kCenterX = kernelWidth / 2;
    int kCenter