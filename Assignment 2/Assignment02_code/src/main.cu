#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>

#define kernelWidth 3
#define kernelHeight 3
#define imageChannels 1

using namespace std;

__constant__ float imageKernel_c[kernelHeight * kernelWidth];

void sequentialConvolution(const unsigned char*inputImageData, const float *kernel ,unsigned char * outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels)
{
	int i, j, m, n, mm, nn;
	int kCenterX, kCenterY;
	float sum;
	int yIndex, xIndex;

	kCenterX = kernelSizeX / 2;
	kCenterY = kernelSizeY / 2;

	for (int k=0; k<channels; k++) 
	{
		for (i = 0; i < dataSizeY; ++i)
		{
			for (j = 0; j < dataSizeX; ++j)
			{
				sum = 0;
				for (m = 0; m < kernelSizeY; ++m)
				{
					mm = kernelSizeY - 1 - m;

					for (n = 0; n < kernelSizeX; ++n)
					{
						nn = kernelSizeX - 1 - n;

						yIndex = i + m - kCenterY;
						xIndex = j + n - kCenterX;

						if (yIndex >= 0 && yIndex < dataSizeY && xIndex >= 0 && xIndex < dataSizeX)
							sum += inputImageData[(dataSizeX * yIndex + xIndex)*channels + k] * kernel[kernelSizeX * mm + nn];
					}
				}
				outputImageData[(dataSizeX * i + j)*channels + k] = sum;
			}
		}
	}
}

__global__ void gpu1 (const unsigned char * InputImageData, const float * kernel, unsigned char* outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < dataSizeY && j < dataSizeX) {
		int kCenterX = kernelSizeX / 2, kCenterY = kernelSizeY / 2;
		for (int k = 0; k < channels; ++k) {
			float sum = 0.0f;
			for (int m = 0; m < kernelSizeY; ++m) {
				int mm = kernelSizeY - 1 - m;
				for (int n = 0; n < kernelSizeX; ++n) {
					int nn = kernelSizeX - 1 - n;
					int yIndex = i + m - kCenterY;
					int xIndex = j + n - kCenterX;
					if(yIndex >= 0 && yIndex < dataSizeY && xIndex >= 0 && xIndex < dataSizeX) {
						sum += InputImageData[(yIndex * dataSizeX + xIndex) * channels + k] * kernel[kernelSizeX * mm + nn];
					}
				}
			}
			outputImageData[(i * dataSizeX + j) * channels + k] = sum;
		}
	}
}

__global__ void gpu2 (const unsigned char * InputImageData, unsigned char* outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels) {
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
          tile[ty + m][tx + n][k] = InputImageData[(yIndex * dataSizeX + xIndex) * channels + k];
        } else {
          tile[ty + m][tx + n][k] = 0;
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
          sum += tile[ty + m][tx + n][k] * imageKernel_c[kernelSizeX * mm + nn];
        }
      }
    }
    outputImageData[(i * dataSizeX + j) * channels + k] = sum;
  }
}

int main(int argc, char* argv[]){
	if(argc < 3) {
		cout<<"Usage: "<<argv[0]<<" <image_in> <image_out>\n";
		return 0;
	}
	// Read input image on host
	int imageWidth, imageHeight, bpp;
	const unsigned char* image_in = stbi_load( argv[1], &imageWidth, &imageHeight, &bpp, imageChannels );
	if(bpp != 1 || imageChannels != 1) {
		cout<<"Input image must be 8 bits per pixel, and sigle channel (grayscale).\n";
		return 0;
	}
	cout << "Image size: " << imageHeight << " x " << imageWidth << std::endl; 
	
	// Allocate output image memory on host
	unsigned char *image_out = (unsigned char*) malloc(imageWidth*imageHeight*sizeof(unsigned char));
	unsigned char *image_gpu1 = (unsigned char*) malloc(imageWidth*imageHeight*sizeof(unsigned char));
	unsigned char *image_gpu2 = (unsigned char*) malloc(imageWidth*imageHeight*sizeof(unsigned char));

	// Setup image convolution kernel on host
	float imageKernel[kernelHeight*kernelWidth];
	for(int i=0; i< kernelWidth*kernelHeight; i++){
		imageKernel[i] = 1.0/(kernelHeight*kernelWidth);
	}

	// Perform image convolution
	sequentialConvolution(image_in, imageKernel, image_out, kernelWidth, 
	kernelHeight, imageWidth, imageHeight, imageChannels);

	// Write convolved image to disk
	stbi_write_png(argv[2], imageWidth, imageHeight, imageChannels, image_out, 0);

	// Add cuda code here
	
	// allocate memory on device
	int size = imageWidth * imageHeight;
	unsigned char *device_image_gpu1, *device_image_gpu2, *image_in_gpu;
	float *imageKernel_gpu;
	cudaMalloc((void **)&device_image_gpu1, size * sizeof(unsigned char));
	cudaMalloc((void **)&device_image_gpu2, size * sizeof(unsigned char));
	cudaMalloc((void **)&image_in_gpu, size * sizeof(unsigned char));
	cudaMalloc((void **)&imageKernel_gpu, kernelHeight * kernelWidth * sizeof(float));
	cudaMemcpy(image_in_gpu, image_in, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(imageKernel_gpu, imageKernel, kernelHeight * kernelWidth * sizeof(float), cudaMemcpyHostToDevice);

	// creating blocks and grids
	dim3 block(16, 16);
	dim3 grid((imageHeight + block.x - 1) / block.x, (imageWidth + block.y - 1) / block.y);

	// launching GPU1 kernel
	// TODO : Timing GPU1 Kernel
	gpu1<<<grid, block>>>(image_in_gpu, imageKernel_gpu, device_image_gpu1, kernelWidth, kernelHeight, imageWidth, imageHeight, imageChannels);
	cudaDeviceSynchronize();

	cudaMemcpy(image_gpu1, device_image_gpu1, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	string gpu1_png(argv[2]);
	gpu1_png.pop_back(); gpu1_png.pop_back(); gpu1_png.pop_back(); gpu1_png.pop_back();
	gpu1_png += "-GPU1.png";
	stbi_write_png(gpu1_png.c_str(), imageWidth, imageHeight, imageChannels, image_gpu1, 0);

	// copying imageKernel to imageKernel_c(constant memory)
	cudaMemcpyToSymbol(imageKernel_c, imageKernel, kernelHeight * kernelWidth * sizeof(float));

	// launching GPU2 kernel
	// TODO : Timing GPU2 Kernel
	gpu2<<<grid, block>>>(image_in_gpu, device_image_gpu2, kernelWidth, kernelHeight, imageWidth, imageHeight, imageChannels);
	cudaDeviceSynchronize();

	cudaMemcpy(image_gpu2, device_image_gpu2, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	string gpu2_png(argv[2]);
	gpu2_png.pop_back(); gpu2_png.pop_back(); gpu2_png.pop_back(); gpu2_png.pop_back();
	gpu2_png += "-GPU2.png";
	stbi_write_png(gpu2_png.c_str(), imageWidth, imageHeight, imageChannels, image_gpu2, 0);

	//Deallocate memory
	free(image_out);
	free(image_gpu1);
	free(image_gpu2);
	cudaFree(device_image_gpu1);
	cudaFree(device_image_gpu2);
	cudaFree(image_in_gpu);
	cudaFree(imageKernel_gpu);
	return 0;
}
