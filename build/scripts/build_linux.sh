#!/bin/bash
# ============================================================================
# PHIQ Elastic KV Cache - Linux Build Script
# Production-Grade CUDA Compilation for GTX 1070 (SM 6.1)
# ============================================================================

echo "PHIQ Elastic KV Cache - Linux Build Script"
echo "==========================================="

# Check requirements
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA Toolkit not found. Please install CUDA 11.8 or higher."
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake 3.18 or higher."
    exit 1
fi

# Set build configuration
BUILD_TYPE="${1:-Release}"
CUDA_ARCH="${2:-61}"
OUTPUT_DIR="build"

echo "Build Type: $BUILD_TYPE"
echo "CUDA Architecture: SM_$CUDA_ARCH"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Create build directory
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
         -DCUDA_ARCH=$CUDA_ARCH \
         -DCMAKE_CUDA_COMPILER=$(which nvcc)

if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed"
    exit 1
fi

# Build
echo "Building..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

echo ""
echo "Build completed successfully!"
echo "Executable: $OUTPUT_DIR/elastic_kv_cli"
echo ""

# Run quick test
if [ -f "elastic_kv_cli" ]; then
    echo "Running quick test..."
    ./elastic_kv_cli --seq=1024 --compress=2 --reps=10 --json
    echo ""
    echo "Build and test completed successfully!"
else
    echo "Warning: Executable not found"
fi