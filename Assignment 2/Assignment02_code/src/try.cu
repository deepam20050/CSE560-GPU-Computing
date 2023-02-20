#define imageChannels 1

__global__ void gpu2 (const unsigned char * InputImageData, const float * kernel, unsigned char* outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int kCenterX = kernelSizeX / 2, kCenterY = kernelSizeY / 2;
  __shared__ unsigned char tile[32][32][imageChannels];
  for (int k = 0; k < channels; ++k) {
    for (int m = 0; m < kernelSizeY; ++m) {
      for (int n = 0; n < kernelSizeX; ++n) {
        int yIndex = i + m - kCenterY;
        int xIndex = j + n - kCenterX;
        if( yIndex >= 0 && yIndex < dataSizeY && xIndex >= 0 && xIndex < dataSizeX) {
          tile[tx + m][ty + n][k] = InputImageData[(yIndex * dataSizeX + xIndex) * channels + k];
        } else {
          tile[tx + m][ty + n][k] = 0;
        }
      }
    }
  }
  __syncthreads();
  for (int k = 0; k < channels; ++k) {
    float sum = 0.0f;
    for (int m = 0; m < kernelSizeY; ++m) {
      int mm = kernelSizeY - 1 - m;
      for (int n = 0; n < kernelSizeX; ++n) {
        int nn = kernelSizeX - 1 - n;
        int yIndex = i + m - kCenterY;
        int xIndex = j + n - kCenterX;
        if(yIndex >= 0 && yIndex < dataSizeY && xIndex >= 0 && xIndex < dataSizeX) {
          sum += tile[ty + m][tx + n][k] * kernel[kernelSizeX * mm + nn];
        }
      }
    }
    outputImageData[(i * dataSizeX + j) * channels + k] = sum;
  }
}