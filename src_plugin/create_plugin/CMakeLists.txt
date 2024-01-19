
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)


set(TRT_LIB_DIR  /usr/tensorrt/TensorRT-8.4.1.5/targets/x86_64-linux-gnu/lib/)
set(TRT_INCLUDE_DIR /usr/tensorrt/TensorRT-8.4.1.5/include/)
set(CUDA_DIR /usr/local/cuda-10.2)



set(CMAKE_SKIP_BUILD_RPATH True)

project(pillarscatter-plugin
        LANGUAGES CXX CUDA
        VERSION "8.4.0")

# C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(CUDA  REQUIRED)
# endif()
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_ROOT_DIR}/include
)

if(NOT CMAKE_CUDA_DEVICE_LINK_LIBRARY)
  set(CMAKE_CUDA_DEVICE_LINK_LIBRARY
    "<CMAKE_CUDA_COMPILER> ${CMAKE_CUDA_HOST_FLAGS} <LANGUAGE_COMPILE_FLAGS> ${CMAKE_CUDA_COMPILE_OPTIONS_PIC} ${_CMAKE_CUDA_EXTRA_DEVICE_LINK_FLAGS} -shared -dlink <OBJECTS> -o <TARGET> <LINK_LIBRARIES>${__IMPLICT_DLINK_FLAGS}")
endif()
if(NOT CMAKE_CUDA_DEVICE_LINK_EXECUTABLE)
  set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE
    "<CMAKE_CUDA_COMPILER> ${CMAKE_CUDA_HOST_FLAGS} <FLAGS> ${CMAKE_CUDA_COMPILE_OPTIONS_PIC} ${_CMAKE_CUDA_EXTRA_DEVICE_LINK_FLAGS} -shared -dlink <OBJECTS> -o <TARGET> <LINK_LIBRARIES>${__IMPLICT_DLINK_FLAGS}")
endif()
set(CUDA_NVCC_FLAGS "-std=c++14 ${CUDA_NVCC_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-fno-gnu-unique")

find_library(CUDNN_LIB cudnn HINTS
    ${CUDA_DIR}  PATH_SUFFIXES lib64 lib)
find_library(CUBLAS_LIB cublas HINTS
    ${CUDA_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUBLASLT_LIB cublasLt HINTS
    ${CUDA_DIR} PATH_SUFFIXES lib64 lib lib/stubs)

find_library(nvinfer nvinfer SHARED ${TRT_LIB_DIR})
find_library(nvuffparser nvparsers SHARED ${TRT_LIB_DIR})

find_library(CUDART_LIB cudart HINTS ${CUDA_DIR} PATH_SUFFIXES lib lib64)
find_library(RT_LIB rt)

set(CUDA_LIBRARIES ${CUDART_LIB})

############################################################################################
# CUDA targets

set(GPU_ARCHS 70)
if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
else()
    # Xavier (SM72) only supported for aarch64.
    list(APPEND GPU_ARCHS 72)
    message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()
set(BERT_GENCODES)
# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
    if (${arch} GREATER_EQUAL 70)
        set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
    endif()
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
if (${LATEST_SM} GREATER_EQUAL 70)
    set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")

############################################################################################

add_subdirectory(plugin)
