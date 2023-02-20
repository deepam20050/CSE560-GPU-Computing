__global__ void gpu2 (const unsigned char * InputImageData, const float * kernel, unsigned char* outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int blockWidth = blockDim.x;
    const int blockHeight = blockDim.y;

    __shared__ unsigned char inputImageTile[(blockWidth + kernelSizeX - 1)][(blockHeight + kernelSizeY - 1)][channels];
    
    const int kCenterX = kernelSizeX / 2;
    const int kCenterY = kernelSizeY / 2;
    
    const int x = bx * blockWidth + tx - kCenterX;
    const int y = by * blockHeight + ty - kCenterY;

    float sum = 0;
    
    if(x >= 0 && y >= 0 && x < dataSizeX && y < dataSizeY) {
        for(int k = 0; k < channels; ++k) {
            inputImageTile[ty][tx][k] = InputImageData[(y * dataSizeX + x) * channels + k];
        }
    } else {
        for(int k = 0; k < channels; ++k) {
            inputImageTile[ty][tx][k] = 0;
        }
    }

    __syncthreads();

    if(tx < blockWidth && ty < blockHeight && x < dataSizeX && y < dataSizeY) {
        for (int m = 0; m < kernelSizeY; ++m) {
            for (int n = 0; n < kernelSizeX; ++n) {
                const int mm = kernelSizeY - 1 - m;
                const int nn = kernelSizeX - 1 - n;
                const int yy = ty + m;
                const int xx = tx + n;

                for(int k = 0; k < channels; ++k) {
                    sum += inputImageTile[yy][xx][k] * kernel[mm * kernelSizeX + nn];
                }
            }
        }
        for(int k = 0; k < channels; ++k) {
            outputImageData[(y * dataSizeX + x) * channels + k] = sum;
        }
    }
}
