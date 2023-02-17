cmake_minimum_required(VERSION 3.18)

project(Lab04 LANGUAGES CUDA CXX)

find_package(CUDAToolkit)
# find_package(OpenMP)

set(SOURCES 
    src/AoS.cu
    src/SoA.cu
    )

add_executable(AoS src/AoS.cu)
add_executable(SoA src/SoA.cu)

target_include_directories(AoS PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(AoS PROPERTIES 
                        CUDA_SEPERABLE_COMPILATION ON
                        CUDA_ARCHITECTURES 60)
target_compile_options(AoS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --keep # Keep PTX code 
                        >)
target_include_directories(SoA PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(SoA PROPERTIES 
                        CUDA_SEPERABLE_COMPILATION ON
                        CUDA_ARCHITECTURES 60)
target_compile_options(SoA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --keep # Keep PTX code 
                        >)

# add_executable(${PROJECT_NAME} ${SOURCES})
# target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
# set_target_properties(${PROJECT_NAME} PROPERTIES 
#                       CUDA_SEPERABLE_COMPILATION ON
#                       CUDA_ARCHITECTURES 60)
# target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                       --keep # Keep PTX code 
#                       >)
# if(OpenMP_CXX_FOUND)
# 	target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
# endif()
