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

__global__ void gpu1 (const unsigned char * InputImageData, const float * kernel, unsigned char* outputImageData, int imageWidth, int imageHeight, int channels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < imageHeight && j < imageWidth) {
		int kCenterX = kernelWidth / 2, kCenterY = kernelHeight / 2;
		for (int k = 0; k < channels; ++k) {
			float sum = 0.0f;
			for (int m = 0; m < kernelHeight; ++m) {
				int mm = kernelHeight - 1 - m;
				for (int n = 0; n < kernelWidth; ++n) {
					int nn = kernelWidth - 1 - n;
					int yIndex = i + m - kCenterY;
					int xIndex = j + n - kCenterX;
					if(yIndex >= 0 && yIndex < imageHeight && xIndex >= 0 && xIndex < imageWidth){
						sum += InputImageData[(yIndex * imageWidth + xIndex) * channels + k] * kernel[kernelWidth * mm + nn];
					}
				}
			}
			outputImageData[(i * imageWidth + j) * channels + k] = sum;
		}
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
	// creating blocks and grids + allocating memory
	int size = imageWidth * imageHeight;
	unsigned char *device_image_gpu1;
	unsigned char *image_gpu1 = new unsigned char[size];
	unsigned char *image_gpu2 = new unsigned char[size];
	cudaMalloc((void **)&device_image_gpu1, size * sizeof(unsigned char));
	dim3 block(16, 16);
	dim3 grid((imageHeight + block.x - 1) / block.x, (imageWidth + block.y - 1) / block.y);
	
	// launching GPU1 kernel
	// TODO : Timing GPU1 Kernel
	gpu1<<<grid, block>>>(image_in, imageKernel, device_image_gpu1, kernelWidth, 
												kernelHeight, imageChannels);
	cudaDeviceSynchronize();

	cudaMemcpy(image_gpu1, device_image_gpu1, size, cudaMemcpyDeviceToHost);
	string gpu1_png(argv[2]);
	gpu1_png.pop_back();
	gpu1_png.pop_back();
	gpu1_png.pop_back();
	gpu1_png.pop_back();
	gpu1_png += "-GPU1.png";
	stbi_write_png(gpu1_png.c_str(), imageWidth, imageHeight, imageChannels, image_gpu1, 0);


	// launching GPU2 kernel
	// TODO : Writing GPU2 + Timing GPU2 Kernel

	//Deallocate memory
	free(image_out);
	delete[] image_gpu1;
	delete[] image_gpu2;
	cudaFree(device_image_gpu1);
	return 0;
}
