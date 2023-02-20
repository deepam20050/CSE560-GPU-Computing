__global__ void convKernel(unsigned char * inputImageData, const float * kernel,
        unsigned char* outputImageData, int channels, int imageWidth, int imageHeight, int kernelWidth, int kernelHeight){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < imageHeight && j < imageWidth){
        for(int k=0; k<channels; k++){
            float sum = 0.0;
            int kCenterX = kernelWidth/2;
            int kCenterY = kernelHeight/2;

            for(int m=0; m<kernelHeight; m++){
                int mm = kernelHeight - 1 - m;
                for(int n=0; n<kernelWidth; n++){
                    int nn = kernelWidth - 1 - n;

                    int yIndex = i + m - kCenterY;
                    int xIndex = j + n - kCenterX;

                    if(yIndex >= 0 && yIndex < imageHeight && xIndex >= 0 && xIndex < imageWidth){
                        sum += (float)inputImageData[(yIndex*imageWidth + xIndex)*channels + k] * kernel[mm*kernelWidth + nn];
                    }
                }
            }
            outputImageData[(i*imageWidth + j)*channels + k] = (unsigned char)sum;
        }
    }
}
