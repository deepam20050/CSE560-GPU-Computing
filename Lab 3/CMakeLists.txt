cmake_minimum_required(VERSION 3.18)

project(Lab03 LANGUAGES CUDA CXX)

find_package(CUDAToolkit)

set(SOURCES 
		src/cpu_code.cpp
		src/gpu_code.cu
		)

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(${PROJECT_NAME} PROPERTIES 
											CUDA_SEPERABLE_COMPILATION ON
											CUDA_ARCHITECTURES 70)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
											--keep # Keep PTX code 
											>)