cmake_minimum_required(VERSION 3.24)
project(flash_attention_cuda LANGUAGES CXX CUDA)

# == cuda ==

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REPLACE "." "" GPU_SM "${GPU_COMPUTE_CAP}")
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_SM})
    message(STATUS "Auto-detected GPU architecture: sm_${GPU_SM}")
endif()

message(STATUS "Building for CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")

# == flags ==

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    -Xcompiler -Wall \
    -Xcompiler -Wextra \
    --use_fast_math \
")

set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG   "${CMAKE_CUDA_FLAGS_DEBUG}   -G -g")

# == targets ==

set(KERNEL_SOURCES
        src/kernels/attention_naive.cu
)

add_executable(flash_attn
    src/main.cu
    src/reference.cu
    include/benchmark.cuh
    ${KERNEL_SOURCES}
)

target_include_directories(flash_attn PRIVATE include/)

target_link_libraries(flash_attn
    CUDA::cublas
    CUDA::cudart
)
