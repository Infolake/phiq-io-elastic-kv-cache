#!/bin/bash
# ============================================================================
#  ΦQ™ PHIQ.IO Elastic KV Cache - Universal Build Script
#  Author: Dr. Guilherme de Camargo
#  Organization: PHIQ.IO Quantum Technologies (ΦQ™)
#  https://phiq.io
#  © 2025 PHIQ.IO Quantum Technologies. All rights reserved.
#
#  Description: Automatic GPU detection and multi-architecture build
#  Camargo Constant: Δ = φ + π = 4.759627
# ============================================================================

set -e

echo "============================================================================"
echo "  PHIQ.IO   _____ _   _ _____ ___        _____ ____                      "
echo "           |  _  | | | |_   _/ _ \\      |  _  |_   |    Elastic KV Cache "
echo "           | |_| | |_| | | || | | |     | |_| | | |                     "
echo "           |  ___| ___ | | || |_| | ___ |  _  | | |  Golden Ticket Ed.  "
echo "           |_|   |_| |_|_|_| \\___(_)___)|_| |_|___|                     "
echo "                                                                          "
echo "       PHIQ.IO Quantum Technologies - GOE Nucleus - https://phiq.io     "
echo "============================================================================"
echo ""

# Detect CUDA and GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits | head -1)
    echo "[OK] Detected GPU: $GPU_INFO"

    # Extract compute capability
    COMPUTE_CAP=$(echo "$GPU_INFO" | cut -d',' -f2 | tr -d ' ')
    ARCH=$(echo "$COMPUTE_CAP" | tr -d '.')
    echo "[OK] Building for architecture: SM $ARCH"

    CMAKE_ARGS="-DCUDA_ARCH=$ARCH"
else
    echo "[WARNING] No GPU detected - building for common architectures"
    CMAKE_ARGS="-DCUDA_ARCH='61;75;80;86;89'"
fi

# Detect number of cores for parallel build
if [[ "$OSTYPE" == "darwin"* ]]; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=$(nproc 2>/dev/null || echo 4)
fi

echo "[OK] Using $CORES CPU cores for parallel build"
echo ""

# Create build directory
mkdir -p build
cd build

# Configure
echo "[CONFIG] Configuring CMake..."
cmake .. $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release

# Build
echo "[BUILD] Building..."
cmake --build . -j$CORES

echo ""
echo "============================================================================"
echo "                          BUILD COMPLETE!"
echo "============================================================================"
echo ""
echo "Binary location: ./build/elastic_kv_cli"
echo ""
echo "Quick test:"
echo "  ./build/elastic_kv_cli --seq=1024 --compress=2 --reps=10"
echo ""
echo "Full benchmark:"
echo "  ./build/elastic_kv_cli --seq=4096 --compress=4 --paired-baseline --json"
echo ""
echo "For more options: ./build/elastic_kv_cli --help"
echo ""
echo "============================================================================"
echo "PHIQ.IO Quantum Technologies - Camargo Constant: Delta = phi + pi = 4.759627"
echo "============================================================================"
