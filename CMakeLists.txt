cmake_minimum_required(VERSION 3.26)
project(he_example LANGUAGES CXX CUDA)

# 保留原有基础配置，完全兼容
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")

find_package(CUDAToolkit REQUIRED)
find_package(HEonGPU 1.1 CONFIG REQUIRED)

# ===================== 修正：定义测试编译开关（核心） =====================
# 为每个测试添加独立的CMake选项，默认值为ON（启用）
option(BUILD_MODEL_TEST "Whether to build modelTest executable" ON)
option(BUILD_LINEAR_TEST "Whether to build linearTest executable" OFF)
option(BUILD_NONLINEAR_TEST "Whether to build nonlinearTest executable" OFF)
option(BUILD_BENCHMARK_TEST "Whether to build benchmark executable" OFF)
option(BUILD_SWINMODEL_TEST "Whether to build SwinModelTest executable" OFF)

set(TEST_CONFIGS
    # 格式：每一行是 [测试名, 编译开关名]
    modelTest      BUILD_MODEL_TEST
    linearTest     BUILD_LINEAR_TEST
    nonlinearTest  BUILD_NONLINEAR_TEST
    benchmarkTest  BUILD_BENCHMARK_TEST
    SwinModelTest  BUILD_SWINMODEL_TEST
)

# 1. 获取公共源码（保留原有逻辑）
file(GLOB COMMON_CU_FILES ${CMAKE_SOURCE_DIR}/src/lib/*.cu)

# 2. 遍历测试配置，按需编译（修正拆分逻辑）
# 每次取2个元素：测试名 + 编译开关名
list(LENGTH TEST_CONFIGS TEST_CONFIGS_LEN)
math(EXPR TEST_CONFIGS_LOOP_LEN "${TEST_CONFIGS_LEN} - 1")
foreach(IDX RANGE 0 ${TEST_CONFIGS_LOOP_LEN} 2)
    # 取出当前测试名和对应的编译开关
    list(GET TEST_CONFIGS ${IDX} EXE_NAME)
    math(EXPR OPTION_IDX "${IDX} + 1")
    list(GET TEST_CONFIGS ${OPTION_IDX} BUILD_OPTION)

    # 拼接测试文件路径（和原有逻辑一致）
    set(TEST_FILE ${CMAKE_SOURCE_DIR}/src/${EXE_NAME}.cu)

    # ===================== 核心条件判断：仅当开关为ON时编译 =====================
    if(${BUILD_OPTION})
        # 检查测试文件是否存在（增加鲁棒性）
        if(NOT EXISTS ${TEST_FILE})
            message(WARNING "Test file ${TEST_FILE} not found, skip building ${EXE_NAME}")
            continue()
        endif()

        # 生成可执行文件：公共源码 + 当前测试文件
        add_executable(${EXE_NAME} ${COMMON_CU_FILES} ${TEST_FILE})
        
        # 头文件包含路径（保留原有逻辑）
        target_include_directories(${EXE_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/include
            ${HEonGPU_INCLUDE_DIRS}
        )
        
        # 链接库配置（保留原有逻辑）
        target_link_libraries(${EXE_NAME} PRIVATE
            HEonGPU::heongpu
            CUDA::cudart
        )
        
        # CUDA属性配置（保留原有逻辑）
        set_target_properties(${EXE_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )

        message(STATUS "Enabled building ${EXE_NAME} (${BUILD_OPTION}=${${BUILD_OPTION}})")
    else()
        message(STATUS "Disabled building ${EXE_NAME} (${BUILD_OPTION}=${${BUILD_OPTION}})")
    endif()
endforeach()