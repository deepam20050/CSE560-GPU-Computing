#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
typedef thrust::complex<float>  _ZN6thrust7complexIfEE;
extern void __device_stub__Z20compute_julia_kernelPhN6thrust7complexIfEE(unsigned char *,  _ZN6thrust7complexIfEE&);
extern void __device_stub__Z12dummy_kernelv(void);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z20compute_julia_kernelPhN6thrust7complexIfEE(unsigned char *__par0,  _ZN6thrust7complexIfEE&__par1){__cudaLaunchPrologue(2);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArg(__par1, 8UL);__cudaLaunch(((char *)((void ( *)(unsigned char *,  _ZN6thrust7complexIfEE))compute_julia_kernel)));}
# 45 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 1/Assignment01_code/src/main.cu"
void compute_julia_kernel( unsigned char *__cuda_0, _ZN6thrust7complexIfEE __cuda_1)
# 45 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 1/Assignment01_code/src/main.cu"
{__device_stub__Z20compute_julia_kernelPhN6thrust7complexIfEE( __cuda_0,__cuda_1);
# 72 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 1/Assignment01_code/src/main.cu"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z12dummy_kernelv(void) {  __cudaLaunchPrologue(1); __cudaLaunch(((char *)((void ( *)(void))dummy_kernel))); }
# 74 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 1/Assignment01_code/src/main.cu"
void dummy_kernel(void)
# 75 "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 1/Assignment01_code/src/main.cu"
{__device_stub__Z12dummy_kernelv();


}
# 1 "main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T50) {  __nv_dummy_param_ref(__T50); __nv_save_fatbinhandle_for_managed_rt(__T50); __cudaRegisterEntry(__T50, ((void ( *)(void))dummy_kernel), _Z12dummy_kernelv, (-1)); __cudaRegisterEntry(__T50, ((void ( *)(unsigned char *,  _ZN6thrust7complexIfEE))compute_julia_kernel), _Z20compute_julia_kernelPhN6thrust7complexIfEE, (-1)); __cudaRegisterVariable(__T50, __shadow_var(_ZN37_INTERNAL_0fad6138_7_main_cu_1b34e0026thrust6system6detail10sequential3seqE,::thrust::system::detail::sequential::seq), 0, 1UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
