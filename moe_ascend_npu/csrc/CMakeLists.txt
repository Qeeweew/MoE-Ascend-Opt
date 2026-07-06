# set the library output dir to the python dir for wheel package build
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/moe_ascend_npu/lib)

# host side files
FILE(GLOB OP_SRCS
    ${PROJECT_OP_SRC_BASE}/pytorch_extensions.cpp
    ${PROJECT_OP_SRC_BASE}/op_host/grouped_gemv_w4a16_moe.cpp
    )

# set the so name
set(OP_PLUGIN_NAME moe_ascend_npu_kernels)

# kernel side files without workspace
set(NO_WORKSPACE_KERNEL_SRCS
    ${PROJECT_OP_SRC_BASE}/op_kernel/stub_pipe.cpp
    ${PROJECT_OP_SRC_BASE}/op_kernel/grouped_gemv_w4a16_moe.cpp
)

ascendc_library(no_workspace_kernel STATIC ${NO_WORKSPACE_KERNEL_SRCS})
ascendc_include_directories(no_workspace_kernel PRIVATE
    ${PROJECT_OP_SRC_BASE}/utils/kernel
)

# create shared library libmoe_ascend_npu_kernels.so
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        no_workspace_kernel
        torch_npu
        ascendcl
        tiling_api
        register
        platform
        ascendalog
        dl
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
        ${PROJECT_OP_SRC_BASE}/utils
        ${PROJECT_SOURCE_DIR}/include
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include
        ${ASCEND_INCLUDE_DIR}
        ${ASCEND_INCLUDE_DIR}/external
        ${ASCEND_INCLUDE_DIR}/experiment/platform
        ${ASCEND_INCLUDE_DIR}/experiment/runtime
)
