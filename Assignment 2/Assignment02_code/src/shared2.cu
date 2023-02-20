__global__ void convKernel(const unsigned char * inputImageData, const float * __restrict__ kernel,
                           unsigned char* outputImageData, int channels, int imageWidth, int imageHeight,
                           int kernelSizeX, int kernelSizeY)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int blockWidth = blockDim.x;
    const int blockHeight = blockDim.y;
    const int imageChannels = channels;

    __shared__ unsigned char inputImageTile[(blockWidth + kernelSizeX - 1)][(blockHeight + kernelSizeY - 1)][imageChannels];
    
    const int kCenterX = kernelSizeX / 2;
    const int kCenterY = kernelSizeY / 2;
    
    const int x = bx * blockWidth + tx - kCenterX;
    const int y = by * blockHeight + ty - kCenterY;

    float sum = 0;
    
    if(x >= 0 && y >= 0 && x < imageWidth && y < imageHeight) {
        for(int k = 0; k < imageChannels; ++k) {
            inputImageTile[ty][tx][k] = inputImageData[(y * imageWidth + x) * imageChannels + k];
        }
    } else {
        for(int k = 0; k < imageChannels; ++k) {
            inputImageTile[ty][tx][k] = 0;
        }
    }

    __syncthreads();

    if(tx < blockWidth && ty < blockHeight && x < imageWidth && y < imageHeight) {
        for (int m = 0; m < kernelSizeY; ++m) {
            for (int n = 0; n < kernelSizeX; ++n) {
                const int mm = kernelSizeY - 1 - m;
                const int nn = kernelSizeX - 1 - n;
                const int yy = ty + m;
                const int xx = tx + n;

                for(int k = 0; k < imageChannels; ++k) {
                    sum += inputImageTile[yy][xx][k] * kernel[mm * kernelSizeX + nn];
                }
            }
        }
        for(int k = 0; k < imageChannels; ++k) {
            outputImageData[(y * imageWidth + x) * imageChannels + k] = sum;
        }
    }
}
