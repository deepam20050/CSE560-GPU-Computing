/* Deepam Sarmah
 * 2020050
 * deepam20050@iiitd.ac.in
 */
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


// Bonus Implementation
__global__ void gpu_tex (cudaTextureObject_t imageKernel_texture, unsigned char* outputImageData, int kernelSizeX, int kernelSizeY, int dataSizeX, int dataSizeY, int channels) {
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
						sum += tex2D < unsigned char > (imageKernel_texture, xIndex, yIndex) * imageKernel_c[kernelSizeX * mm + nn];
					}
				}
			}
			outputImageData[(i * dataSizeX + j) * channels + k] = sum;
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
	unsigned char *image_gpu1 = (unsigned char*) malloc(imageWidth*imageHeight*sizeof(unsigned char));
	unsigned char *image_gpu2 = (unsigned char*) malloc(imageWidth*imageHeight*sizeof(unsigned char));
	unsigned char *image_gpu_tex = (unsigned char*) malloc(imageWidth*imageHeight*sizeof(unsigned char));

	// Setup image convolution kernel on host
	float imageKernel[kernelHeight*kernelWidth];
	for(int i=0; i< kernelWidth*kernelHeight; i++){
		imageKernel[i] = 1.0/(kernelHeight*kernelWidth);
	}
	struct timespec start_cpu, end_cpu;
	float msecs_cpu;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_cpu);
	// Perform image convolution
	sequentialConvolution(image_in, imageKernel, image_out, kernelWidth, 
	kernelHeight, imageWidth, imageHeight, imageChannels);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_cpu);
	msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
	printf("[CPU] %f microseconds.\n", msecs_cpu * 1000.0);

	// Add cuda code here
	
	// allocate memory on device
	int size = imageWidth * imageHeight;
	unsigned char *device_image_gpu1, *device_image_gpu2, *image_in_gpu, *device_image_tex;
	float *imageKernel_gpu;
	cudaMalloc((void **)&device_image_gpu1, size * sizeof(unsigned char));
	cudaMalloc((void **)&device_image_gpu2, size * sizeof(unsigned char));
	cudaMalloc((void **)&device_image_tex, size * sizeof(unsigned char));
	cudaMalloc((void **)&image_in_gpu, size * sizeof(unsigned char));
	cudaMalloc((void **)&imageKernel_gpu, kernelHeight * kernelWidth * sizeof(float));
	cudaMemcpy(image_in_gpu, image_in, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(imageKernel_gpu, imageKernel, kernelHeight * kernelWidth * sizeof(float), cudaMemcpyHostToDevice);

	// creating blocks and grids
	dim3 block(16, 16);
	dim3 grid((imageHeight + block.x - 1) / block.x, (imageWidth + block.y - 1) / block.y);

	// launching GPU1 kernel
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
	gpu2<<<grid, block>>>(image_in_gpu, device_image_gpu2, kernelWidth, kernelHeight, imageWidth, imageHeight, imageChannels);
	cudaDeviceSynchronize();
	cudaMemcpy(image_gpu2, device_image_gpu2, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	string gpu2_png(argv[2]);
	gpu2_png.pop_back(); gpu2_png.pop_back(); gpu2_png.pop_back(); gpu2_png.pop_back();
	gpu2_png += "-GPU2.png";
	stbi_write_png(gpu2_png.c_str(), imageWidth, imageHeight, imageChannels, image_gpu2, 0);
	
	// Texture memory part
	cudaChannelFormatDesc channel_description = cudaCreateChannelDesc<unsigned char>();
	cudaArray_t cuArray;
	cudaMallocArray(&cuArray, &channel_description, imageWidth, imageHeight);
	const size_t spitch = imageWidth * sizeof(unsigned char);
	cudaMemcpy2DToArray(cuArray, 0, 0, image_in, spitch, imageWidth * sizeof(unsigned char), imageHeight, cudaMemcpyHostToDevice);
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.readMode = cudaReadModeElementType;
	cudaTextureObject_t imageKernel_texture = 0;
	cudaCreateTextureObject(&imageKernel_texture, &resDesc, &texDesc, NULL);
	// launching gpu_tex
	gpu_tex<<<grid, block>>>(imageKernel_texture, device_image_tex, kernelWidth, kernelHeight, imageWidth, imageHeight, imageChannels);
	cudaDeviceSynchronize();
	cudaMemcpy(image_gpu_tex, device_image_tex, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	string gputex_png(argv[2]);
	gputex_png.pop_back(); gputex_png.pop_back(); gputex_png.pop_back(); gputex_png.pop_back();
	gputex_png += "-GPU_TEX.png";
	stbi_write_png(gputex_png.c_str(), imageWidth, imageHeight, imageChannels, image_gpu_tex, 0);
	
	//Deallocate memory
	free(image_out);
	free(image_gpu1);
	free(image_gpu2);
	free(image_gpu_tex);
	cudaFree(device_image_gpu1);
	cudaFree(device_image_gpu2);
	cudaFree(device_image_tex);
	cudaFree(image_in_gpu);
	cudaFree(imageKernel_gpu);
	return 0;
}
