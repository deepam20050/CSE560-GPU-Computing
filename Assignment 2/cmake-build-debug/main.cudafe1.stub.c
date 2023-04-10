#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
extern void __device_stub__Z4gpu1PKhPKfPhiiiii(const unsigned char *, const float *, unsigned char *, int, int, int, int, int);
extern void __device_stub__Z4gpu2PKhPhiiiii(const unsigned char *, unsigned char *, int, int, int, int, int);
extern void __device_stub__Z7gpu_texyPhiiiii(unsigned long long, unsigned char *, int, int, int, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z4gpu1PKhPKfPhiiiii(const unsigned char *__par0, const float *__par1, unsigned char *__par2, int __par3, int __par4, int __par5, int __par6, int __par7){__cudaLaunchPrologue(8);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 28UL);__cudaSetupArgSimple(__par5, 32UL);__cudaSetupArgSimple(__par6, 36UL);__cudaSetupArgSimple(__par7, 40UL);__cudaLaunch(((char *)((void ( *)(const unsigned char *, const float *, unsigned char *, int, int, int, int, int))gpu1)));}
# 59 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/src/main.cu"
void gpu1( const unsigned char *__cuda_0,const float *__cuda_1,unsigned char *__cuda_2,int __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6,int __cuda_7)
# 59 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/src/main.cu"
{__device_stub__Z4gpu1PKhPKfPhiiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 80 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/src/main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z4gpu2PKhPhiiiii( const unsigned char *__par0,  unsigned char *__par1,  int __par2,  int __par3,  int __par4,  int __par5,  int __par6) {  __cudaLaunchPrologue(7); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 28UL); __cudaSetupArgSimple(__par6, 32UL); __cudaLaunch(((char *)((void ( *)(const unsigned char *, unsigned char *, int, int, int, int, int))gpu2))); }
# 82 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/src/main.cu"
void gpu2( const unsigned char *__cuda_0,unsigned char *__cuda_1,int __cuda_2,int __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6)
# 82 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/src/main.cu"
{__device_stub__Z4gpu2PKhPhiiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 118 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/src/main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z7gpu_texyPhiiiii( unsigned long long __par0,  unsigned char *__par1,  int __par2,  int __par3,  int __par4,  int __par5,  int __par6) {  __cudaLaunchPrologue(7); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 20UL); __cudaSetupArgSimple(__par4, 24UL); __cudaSetupArgSimple(__par5, 28UL); __cudaSetupArgSimple(__par6, 32UL); __cudaLaunch(((char *)((void ( *)(unsigned long long, unsigned char *, int, int, int, int, int))gpu_tex))); }
# 122 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/src/main.cu"
void gpu_tex( unsigned long long __cuda_0,unsigned char *__cuda_1,int __cuda_2,int __cuda_3,int __cuda_4,int __cuda_5,int __cuda_6)
# 122 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/src/main.cu"
{__device_stub__Z7gpu_texyPhiiiii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 143 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/src/main.cu"
}
# 1 "main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T44) {  __nv_dummy_param_ref(__T44); __nv_save_fatbinhandle_for_managed_rt(__T44); __cudaRegisterEntry(__T44, ((void ( *)(unsigned long long, unsigned char *, int, int, int, int, int))gpu_tex), _Z7gpu_texyPhiiiii, (-1)); __cudaRegisterEntry(__T44, ((void ( *)(const unsigned char *, unsigned char *, int, int, int, int, int))gpu2), _Z4gpu2PKhPhiiiii, (-1)); __cudaRegisterEntry(__T44, ((void ( *)(const unsigned char *, const float *, unsigned char *, int, int, int, int, int))gpu1), _Z4gpu1PKhPKfPhiiiii, (-1)); __cudaRegisterVariable(__T44, __shadow_var(imageKernel_c,::imageKernel_c), 0, 36UL, 1, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
