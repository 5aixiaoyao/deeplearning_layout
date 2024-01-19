add_custom_target(plugin)

set(TRT_LIB_DIR  "/usr/tensorrt/TensorRT-8.4.1.5/targets/x86_64-linux-gnu/lib/")
find_library(nvinfer nvinfer SHARED ${TRT_LIB_DIR})
set(TARGET_NAME nvinfer_plugin)
set(SHARED_TARGET ${TARGET_NAME})

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PLUGIN_EXPORT_MAP ${TARGET_DIR}/exports.map)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

include_directories(${TARGET_DIR}/PillarScatter)
include_directories(${TARGET_DIR}/ScatterND_MoGo)
include_directories(${TARGET_DIR}/Grid2Points)
add_subdirectory(PillarScatter)
add_subdirectory(ScatterND_MoGo)
add_subdirectory(Grid2Points)
include_directories(${TARGET_DIR}/common)
add_subdirectory(common)

# Set gencodes
set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS ${GENCODES})
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")


list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/InferPlugin.cpp")
#list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/InferPlugin.cpp")

################################## SHARED LIBRARY #######################################

add_library(${SHARED_TARGET} SHARED
    ${PLUGIN_SOURCES}
)
set(TRT_INCLUDE_DIR /usr/tensorrt/TensorRT-8.4.1.5/include/)
target_include_directories(nvinfer_plugin PUBLIC ${TRT_INCLUDE_DIR})

target_include_directories(${SHARED_TARGET}
    # PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${TARGET_DIR}/common
    # PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${TARGET_DIR}/PillarScatter
    PRIVATE ${TARGET_DIR}/ScatterND_MoGo
    PRIVATE ${TARGET_DIR}/Grid2Points
)


set_property(TARGET ${SHARED_TARGET} PROPERTY CUDA_STANDARD 14)

target_link_libraries(${SHARED_TARGET}
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CUDART_LIB}
    ${CUDNN_LIB}
    nvinfer
    ${CMAKE_DL_LIBS}
)

#########################################################################################

add_dependencies(plugin ${SHARED_TARGET} )
