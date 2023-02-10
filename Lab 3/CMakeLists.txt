cmake_minimum_required(VERSION 3.18)

project(Lab03 LANGUAGES CUDA CXX)

find_package(CUDAToolkit)
# find_package(OpenMP)

set(SOURCES 
    src/cpu_code.cpp
    )

add_executable(gpu_exec src/gpu_code.cu)
add_executable(cpu_exec src/cpu_code.cpp)

target_include_directories(gpu_exec PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(gpu_exec PROPERTIES 
                      CUDA_SEPERABLE_COMPILATION ON
                      CUDA_ARCHITECTURES 70)
target_compile_options(gpu_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                      --keep # Keep PTX code 
                      >)
#if(OpenMP_CXX_FOUND)
#	target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
#endif()