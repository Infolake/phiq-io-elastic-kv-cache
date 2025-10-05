cmake_minimum_required(VERSION 3.18)

# ============================================================================
#  ΦQ™ PHIQ.IO Elastic KV Cache – Build Configuration – GOE Nucleus
#  Author: Dr. Guilherme de Camargo
#  Organization: PHIQ.IO Quantum Technologies (ΦQ™)
#  © 2025 PHIQ.IO Quantum Technologies. All rights reserved.
# ============================================================================

project(PHIQElasticKVCache
    VERSION 1.0.0
    LANGUAGES CXX CUDA
    DESCRIPTION "ΦQ™ Production-Grade Elastic Key-Value Cache for LLM Inference"
)

set(PROJECT_VENDOR "PHIQ.IO Quantum Technologies")
set(PROJECT_CONTACT "https://phiq.io")
set(PROJECT_DESCRIPTION "High-Performance Elastic Key-Value Cache for Large Language Models")

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect CUDA architecture or use multi-arch build
if(NOT DEFINED CUDA_ARCH)
    # Try to detect GPU compute capability
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(GPU_COMPUTE_CAP)
        string(REPLACE "." "" CUDA_ARCH "${GPU_COMPUTE_CAP}")
        message(STATUS "Auto-detected CUDA architecture: SM ${CUDA_ARCH}")
    else()
        # Build for multiple common architectures (fat binary - Pascal through Hopper)
        set(CUDA_ARCH "61;75;80;86;89;90")  # Pascal, Turing, Ampere, Ada, Hopper
        message(STATUS "No GPU detected, building for common architectures: ${CUDA_ARCH}")
    endif()
else()
    message(STATUS "Using specified CUDA architecture: SM ${CUDA_ARCH}")
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Project options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test programs" ON)
option(ENABLE_CUDA_GRAPHS "Enable CUDA Graphs optimization" ON)
option(ENABLE_FAST_MATH "Enable fast math operations" ON)

# Find packages
find_package(CUDAToolkit REQUIRED)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MD")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /fp:fast /GS- /favor:INTEL64")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
endif()

# CUDA compiler options with multi-architecture support (audit-ready)
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --maxrregcount=64 -lineinfo")

if(ENABLE_FAST_MATH)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
endif()

# Compile definitions
add_compile_definitions(
    OPTIMAL_BLOCK_SIZE=256
    VECTOR_WIDTH=4
    THREADS_PER_BLOCK=256
)

if(ENABLE_CUDA_GRAPHS)
    add_compile_definitions(ENABLE_CUDA_GRAPHS=1)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Shared library for open-source minimal implementation
add_library(elastic_kv_core SHARED
    src/elastic_kv_core.cu
)

# Set CUDA architectures for the library (optimized for Colab: T4, V100, A100)
set_target_properties(elastic_kv_core PROPERTIES
    CUDA_ARCHITECTURES "70;75;80"  # V100, T4, A100
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "elastic_kv_core"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Link libraries for the shared library
target_link_libraries(elastic_kv_core
    CUDA::cudart
    CUDA::cuda_driver
)

# Main executable
add_executable(elastic_kv_cli
    src/elastic_kv_cli.cu
)

# Set CUDA architectures for the target (production multi-arch)
set_target_properties(elastic_kv_cli PROPERTIES
    CUDA_ARCHITECTURES "${CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "elastic_kv_cli"
)

# Link libraries
target_link_libraries(elastic_kv_cli
    CUDA::cudart
    CUDA::cuda_driver
)

# Installation
install(TARGETS elastic_kv_cli elastic_kv_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Documentation files
install(FILES
    README.md
    LICENSE
    DESTINATION share/doc/phiq-elastic-kv-cache
)

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Display configuration summary
message(STATUS "")
message(STATUS "PHIQ Elastic KV Cache Configuration Summary")
message(STATUS "===========================================")
message(STATUS "Project Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architecture: SM_${CUDA_ARCH}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Fast Math: ${ENABLE_FAST_MATH}")
message(STATUS "CUDA Graphs: ${ENABLE_CUDA_GRAPHS}")
message(STATUS "Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "")