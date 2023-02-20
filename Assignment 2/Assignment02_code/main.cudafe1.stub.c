#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
extern void __device_stub__Z4gpu1PKhPKfPhiii(const unsigned char *, const float *, unsigned char *, int, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z4gpu1PKhPKfPhiii(const unsigned char *__par0, const float *__par1, unsigned char *__par2, int __par3, int __par4, int __par5){__cudaLaunchPrologue(6);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 28UL);__cudaSetupArgSimple(__par5, 32UL);__cudaLaunch(((char *)((void ( *)(const unsigned char *, const float *, unsigned char *, int, int, int))gpu1)));}
# 53 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code/src/main.cu"
void gpu1( const unsigned char *__cuda_0,const float *__cuda_1,unsigned char *__cuda_2,int __cuda_3,int __cuda_4,int __cuda_5)
# 53 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code/src/main.cu"
{__device_stub__Z4gpu1PKhPKfPhiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5);
# 74 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code/src/main.cu"
}
# 1 "main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T36) {  __nv_dummy_param_ref(__T36); __nv_save_fatbinhandle_for_managed_rt(__T36); __cudaRegisterEntry(__T36, ((void ( *)(const unsigned char *, const float *, unsigned char *, int, int, int))gpu1), _Z4gpu1PKhPKfPhiii, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
