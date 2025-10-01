@echo off
REM ============================================================================
REM  ΦQ™ PHIQ.IO Elastic KV Cache - Windows Build Script
REM  Author: Dr. Guilherme de Camargo
REM  Organization: PHIQ.IO Quantum Technologies (ΦQ™)
REM  https://phiq.io
REM  © 2025 PHIQ.IO Quantum Technologies. All rights reserved.
REM
REM  Camargo Constant: Δ = φ + π = 4.759627
REM ============================================================================

echo ============================================================================
echo   PHIQ.IO Elastic KV Cache - Windows Build
echo   PHIQ.IO Quantum Technologies - GOE Nucleus - https://phiq.io
echo ============================================================================
echo.

REM Detect GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits > gpu_info.txt 2>nul
if %ERRORLEVEL% EQU 0 (
    set /p GPU_INFO=<gpu_info.txt
    echo [OK] Detected GPU: %GPU_INFO%
    for /f "tokens=2 delims=," %%a in (gpu_info.txt) do set COMPUTE_CAP=%%a
    set COMPUTE_CAP=%COMPUTE_CAP: =%
    set COMPUTE_CAP=%COMPUTE_CAP:.=%
    echo [OK] Building for architecture: SM %COMPUTE_CAP%
    set CMAKE_ARGS=-DCUDA_ARCH=%COMPUTE_CAP%
    del gpu_info.txt
) else (
    echo [WARNING] No GPU detected - building for common architectures
    set CMAKE_ARGS=-DCUDA_ARCH="61;75;80;86;89"
)

REM Create build directory
if not exist build mkdir build
cd build

REM Configure
echo.
echo [CONFIG] Configuring CMake...
cmake .. %CMAKE_ARGS% -DCMAKE_BUILD_TYPE=Release

REM Build
echo.
echo [BUILD] Building...
cmake --build . --config Release

echo.
echo ============================================================================
echo                          BUILD COMPLETE!
echo ============================================================================
echo.
echo Binary location: .\build\Release\elastic_kv_cli.exe
echo.
echo Quick test:
echo   .\build\Release\elastic_kv_cli.exe --seq=1024 --compress=2 --reps=10
echo.
echo For more options: .\build\Release\elastic_kv_cli.exe --help
echo.
echo ============================================================================
echo PHIQ.IO Quantum Technologies - Camargo Constant: Delta = phi + pi = 4.759627
echo ============================================================================

cd ..
pause
